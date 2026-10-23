from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRouter

from pulsarlm.server.openai import ServerState

router = APIRouter()


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request) -> str:
    """The engine's stat in the Prometheus text format."""
    state: ServerState = request.app.state.server_state
    stat = state.engine.stat()
    labels = f'model_name="{state.model_id}"'
    return (
        "# HELP pulsarlm:kv_cache_bytes Bytes of the KV cache that running requests use.\n"
        "# TYPE pulsarlm:kv_cache_bytes gauge\n"
        f"pulsarlm:kv_cache_bytes{{{labels}}} {stat.kv_cache_bytes}\n"
    )
