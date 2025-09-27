from __future__ import annotations
from sn1.boot import entrypoint
import sn1

@entrypoint("solve")
def solve(*, observation: dict) -> str:
    prompt = observation.get("prompt", "")
    return sn1.rpc("chutes", prompt=prompt) or "<Answer>NaN</Answer>"


