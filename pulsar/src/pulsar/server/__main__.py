"""Serves a model: python -m pulsar.server <model>. Other settings come from PULSAR_*
environment variables (see Settings.from_env)."""

import argparse

import uvicorn

from pulsar.server.app import Settings, create_app, load_server_state


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pulsar.server")
    parser.add_argument("model", help="the name of a .pulsar model in the data paths")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings.from_env(args.model)
    uvicorn.run(
        create_app(lambda: load_server_state(settings)), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
