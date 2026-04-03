from openenv.core import create_fastapi_app

from server.environment import ModGuardEnvironment
from server.models import ModGuardAction, ModGuardObservation

_ENV_INSTANCE = ModGuardEnvironment()

app = create_fastapi_app(
    env=lambda: _ENV_INSTANCE,
    action_cls=ModGuardAction,
    observation_cls=ModGuardObservation,
)


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
