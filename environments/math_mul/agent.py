from __future__ import annotations
from sn1.boot import entrypoint
import sn1

@entrypoint("solve")
def solve(prompt:str) -> str:
    return sn1.tools.chutes(prompt)