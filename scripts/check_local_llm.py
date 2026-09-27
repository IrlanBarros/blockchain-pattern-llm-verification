#!/usr/bin/env python3
"""Manual connectivity/structured-output probe; excluded from pytest."""
from __future__ import annotations

import json
import time

from llm_pipeline.client import call_sync, close_client, create_client, parse_response_payload
from llm_pipeline.providers import ProviderConfig


def main() -> int:
    config = ProviderConfig.from_environment("openai_compatible")
    client = create_client(config)
    started = time.perf_counter()
    try:
        server = client.probe()
        params = {
            "model": config.model,
            "contents": [{"role": "user", "parts": [{"text": "Return status ok."}]}],
            "config": {
                "system_instruction": "Return only the requested structured JSON.",
                "temperature": 0.0,
                "seed": 42,
                "max_output_tokens": 32,
                "response_json_schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "enum": ["ok"]}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        }
        response = call_sync(client, params, attempts=1)
        payload, metadata = parse_response_payload(response)
    except Exception as exc:
        print(json.dumps({"status": "offline_or_invalid", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    finally:
        close_client(client)
    print(json.dumps({
        "status": "ok", "server": server, "payload": payload,
        "usage": metadata.get("usage"), "model": metadata.get("response_model"),
        "latency_seconds": time.perf_counter() - started,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
