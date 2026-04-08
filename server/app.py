from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from fastapi import Body, FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.types import (
    EnvironmentMetadata,
    HealthResponse,
    HealthStatus,
    ResetRequest,
    ResetResponse,
    SchemaResponse,
    StepRequest,
    StepResponse,
    WSCloseMessage,
    WSErrorCode,
    WSErrorResponse,
    WSObservationResponse,
    WSResetMessage,
    WSStateMessage,
    WSStateResponse,
    WSStepMessage,
)

from server.environment import ModGuardEnvironment
from server.models import ModGuardAction, ModGuardObservation, ModGuardState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"
UI_DIR = PROJECT_ROOT / "ui"
SESSION_COOKIE = "modguard_session_id"

app = FastAPI(
    title="OpenEnv Environment HTTP API",
    version="2.0.0",
    description=(
        "HTTP and WebSocket API for the ModGuard-RL content moderation "
        "triage environment."
    ),
)

if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="modguard_ui")

_HTTP_SESSIONS: dict[str, ModGuardEnvironment] = {}
_HTTP_SESSION_LOCK = RLock()


def _web_enabled() -> bool:
    return os.getenv("ENABLE_WEB_INTERFACE", "true").lower() in {"1", "true", "yes"}


def _serialize_observation(observation: ModGuardObservation) -> dict[str, Any]:
    return {
        "observation": observation.model_dump(exclude={"reward", "done"}),
        "reward": None if observation.reward is None else float(observation.reward),
        "done": bool(observation.done),
    }


def _read_readme() -> Optional[str]:
    if not README_PATH.is_file():
        return None
    return README_PATH.read_text(encoding="utf-8")


def _metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="ModGuard-RL",
        description=(
            "A multi-stage moderation triage environment with escalation budgeting, "
            "adversarial confidence signals, and post-decision audits."
        ),
        version="2.0.0",
        author="Codex + user collaboration",
        documentation_url="https://huggingface.co/spaces",
        readme_content=_read_readme(),
    )


def _coerce_action(action_data: dict[str, Any]) -> ModGuardAction:
    try:
        return ModGuardAction.model_validate(action_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def _resolve_session_id(
    request: Request,
    session_id_header: Optional[str],
) -> Optional[str]:
    return (
        session_id_header
        or request.query_params.get("session_id")
        or request.cookies.get(SESSION_COOKIE)
    )


def _get_or_create_http_env(session_id: str) -> ModGuardEnvironment:
    with _HTTP_SESSION_LOCK:
        env = _HTTP_SESSIONS.get(session_id)
        if env is None:
            env = ModGuardEnvironment()
            _HTTP_SESSIONS[session_id] = env
        return env


def _require_http_env(session_id: Optional[str]) -> ModGuardEnvironment:
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No active session. Call /reset first.",
        )
    with _HTTP_SESSION_LOCK:
        env = _HTTP_SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session expired or missing. Call /reset first.",
        )
    return env


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    if _web_enabled() and UI_DIR.is_dir():
        return RedirectResponse(url="/ui/", status_code=302)
    return RedirectResponse(url="/health", status_code=302)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse(status=HealthStatus.HEALTHY)


@app.get("/metadata", response_model=EnvironmentMetadata, tags=["Environment Info"])
def metadata() -> EnvironmentMetadata:
    return _metadata()


@app.get("/schema", response_model=SchemaResponse, tags=["Schema"])
def schema() -> SchemaResponse:
    return SchemaResponse(
        action=ModGuardAction.model_json_schema(),
        observation=ModGuardObservation.model_json_schema(),
        state=ModGuardState.model_json_schema(),
    )


@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
def reset(
    request: Request,
    response: Response,
    payload: ResetRequest = Body(default_factory=ResetRequest),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
) -> ResetResponse:
    session_id = _resolve_session_id(request, x_session_id) or str(uuid.uuid4())
    env = _get_or_create_http_env(session_id)
    observation = env.reset(**payload.model_dump(exclude_unset=True))
    response.set_cookie(key=SESSION_COOKIE, value=session_id, httponly=False, samesite="lax")
    return ResetResponse(**_serialize_observation(observation))


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
def step(
    request: Request,
    response: Response,
    payload: StepRequest,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
) -> StepResponse:
    session_id = _resolve_session_id(request, x_session_id)
    env = _require_http_env(session_id)
    observation = env.step(_coerce_action(payload.action))
    response.set_cookie(key=SESSION_COOKIE, value=session_id, httponly=False, samesite="lax")
    return StepResponse(**_serialize_observation(observation))


@app.get("/state", response_model=ModGuardState, tags=["State Management"])
def state(
    request: Request,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
) -> ModGuardState:
    session_id = _resolve_session_id(request, x_session_id)
    env = _require_http_env(session_id)
    return env.get_state()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    env = ModGuardEnvironment()
    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError as exc:
                await websocket.send_text(
                    WSErrorResponse(
                        data={
                            "message": f"Invalid JSON: {exc}",
                            "code": WSErrorCode.INVALID_JSON,
                        }
                    ).model_dump_json()
                )
                continue

            msg_type = message.get("type")
            try:
                if msg_type == "reset":
                    reset_msg = WSResetMessage(**message)
                    observation = env.reset(**reset_msg.data)
                    await websocket.send_text(
                        WSObservationResponse(
                            data=_serialize_observation(observation)
                        ).model_dump_json()
                    )
                elif msg_type == "step":
                    step_msg = WSStepMessage(**message)
                    observation = env.step(_coerce_action(step_msg.data))
                    await websocket.send_text(
                        WSObservationResponse(
                            data=_serialize_observation(observation)
                        ).model_dump_json()
                    )
                elif msg_type == "state":
                    WSStateMessage(**message)
                    await websocket.send_text(
                        WSStateResponse(data=env.get_state().model_dump()).model_dump_json()
                    )
                elif msg_type == "close":
                    WSCloseMessage(**message)
                    break
                else:
                    await websocket.send_text(
                        WSErrorResponse(
                            data={
                                "message": f"Unknown message type: {msg_type}",
                                "code": WSErrorCode.UNKNOWN_TYPE,
                            }
                        ).model_dump_json()
                    )
            except HTTPException as exc:
                await websocket.send_text(
                    WSErrorResponse(
                        data={
                            "message": str(exc.detail),
                            "code": WSErrorCode.VALIDATION_ERROR,
                        }
                    ).model_dump_json()
                )
            except Exception as exc:
                await websocket.send_text(
                    WSErrorResponse(
                        data={
                            "message": str(exc),
                            "code": WSErrorCode.EXECUTION_ERROR,
                        }
                    ).model_dump_json()
                )
    except WebSocketDisconnect:
        pass
    finally:
        env.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    main()
