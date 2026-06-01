"""
HTTP adapter — exposes your env via the BenchAnything four-endpoint protocol.

Local dev:
    python adapter.py
    python adapter.py --port 9000
"""

import argparse

from bench_common.env_sdk import serve
from env import TwisterEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"TwisterEnv adapter → http://{args.host}:{args.port}")
    serve(TwisterEnv, host=args.host, port=args.port)
