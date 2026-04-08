from __future__ import annotations

from typing import Any

from openenv.core import GenericEnvClient, SyncEnvClient

from server.models import ActionType, ModGuardAction, ModGuardObservation, ModGuardState


class ModGuardClient:
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url
        self._sync_client: SyncEnvClient | None = None

    def __enter__(self) -> "ModGuardClient":
        async_client = GenericEnvClient(base_url=self._base_url)
        self._sync_client = SyncEnvClient(async_client)
        self._sync_client.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._sync_client:
            self._sync_client.disconnect()

    def _require_client(self) -> SyncEnvClient:
        if self._sync_client is None:
            raise RuntimeError("ModGuardClient is not connected. Use it within a with-block.")
        return self._sync_client

    def _to_observation(self, payload: Any) -> ModGuardObservation:
        if isinstance(payload, ModGuardObservation):
            return payload
        if isinstance(payload, dict):
            return ModGuardObservation.model_validate(payload)
        if hasattr(payload, "model_dump"):
            return ModGuardObservation.model_validate(payload.model_dump())
        if hasattr(payload, "__dict__"):
            return ModGuardObservation.model_validate(payload.__dict__)
        return ModGuardObservation.model_validate(payload)

    def _result_to_observation(self, result: Any) -> ModGuardObservation:
        if isinstance(result.observation, dict):
            obs_dict = dict(result.observation)
            obs_dict["done"] = result.done
            obs_dict["reward"] = result.reward
            return ModGuardObservation.model_validate(obs_dict)
        observation = self._to_observation(result.observation)
        observation.done = result.done
        observation.reward = result.reward
        return observation

    def reset(
        self,
        seed: int | None = None,
        difficulty: str | None = None,
        task_name: str | None = None,
    ) -> ModGuardObservation:
        kwargs = {"seed": seed} if seed is not None else {}
        if difficulty is not None:
            kwargs["difficulty"] = difficulty
        if task_name is not None:
            kwargs["task_name"] = task_name
        result = self._require_client().reset(**kwargs)
        return self._result_to_observation(result)

    def step(self, action: ModGuardAction) -> ModGuardObservation:
        result = self._require_client().step(action)
        return self._result_to_observation(result)

    def get_state(self) -> ModGuardState:
        payload = self._require_client().state()
        if isinstance(payload, ModGuardState):
            return payload
        if isinstance(payload, dict):
            return ModGuardState.model_validate(payload)
        if hasattr(payload, "model_dump"):
            return ModGuardState.model_validate(payload.model_dump())
        if hasattr(payload, "__dict__"):
            return ModGuardState.model_validate(payload.__dict__)
        return ModGuardState.model_validate(payload)


if __name__ == "__main__":
    with ModGuardClient() as client:
        observation = client.reset()
        observation = client.step(ModGuardAction(action=ActionType.approve))
        state = client.get_state()

    print(observation)
    print(f"Final reward: {observation.reward}")
    print(state)
