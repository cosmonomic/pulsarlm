from fastapi import Request, Response
from fastapi.routing import APIRouter
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from pulsarlm.server.openai import ServerState

router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """The server's OpenTelemetry metrics in the Prometheus text format."""
    state: ServerState = request.app.state.server_state
    return Response(
        generate_latest(state.telemetry.registry), media_type=CONTENT_TYPE_LATEST
    )
