import uvicorn

from pulsarlm.checkpoint import PulsarLMModelLoadingError, model_path
from pulsarlm.server.app import create_app, load_server_state
from pulsarlm.settings import Settings

__all__ = ["PulsarLMModelLoadingError", "serve"]


def serve(settings: Settings, *, host: str, port: int) -> None:
    """Serves the model until interrupted. Raises PulsarLMModelLoadingError,
    before starting, when no data path holds it."""
    path = model_path(settings.model, settings.data_dir)
    uvicorn.run(
        create_app(lambda: load_server_state(settings, path)), host=host, port=port
    )
