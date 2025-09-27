from __future__ import annotations
import os
import random
import asyncio
import aiohttp
import logging
from typing import Any
from sn1 import declare_tool, get_conf

TERMINAL = {400, 404, 410}

@declare_tool
async def chutes(*, prompt: str, model: str = "unsloth/gemma-3-12b-it", timeout: int = 150, retries: int = 0, backoff: float = 1) -> str | None:
    url = f"https://llm.chutes.ai/v1/chat/completions"
    hdr = {"Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as client:
        sem = asyncio.Semaphore(int(os.getenv("SN1_HTTP_CONCURRENCY", "16")))
        for attempt in range(1, retries + 2):
            try:
                payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
                async with sem, client.post(url, json=payload, headers=hdr, timeout=timeout) as r:
                    _ = await r.text(errors="ignore")
                    if r.status in TERMINAL:
                        return None
                    r.raise_for_status()
                    return (await r.json())["choices"][0]["message"]["content"]
            except Exception:
                if attempt > retries:
                    return None
                await asyncio.sleep(backoff * 2 ** (attempt - 1) * (1 + random.uniform(-0.1, 0.1)))

