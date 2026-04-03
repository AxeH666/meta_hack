from openenv.core import create_fastapi_app

from server.environment import ModGuardEnvironment
from server.models import ModGuardAction, ModGuardObservation

app = create_fastapi_app(
    env=lambda: ModGuardEnvironment(),
    action_cls=ModGuardAction,
    observation_cls=ModGuardObservation,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
