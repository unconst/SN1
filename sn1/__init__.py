# sn1/__init__.py
from __future__ import annotations
import os
import uuid
import time
import click
import random
import aiohttp
import uvicorn
import asyncio
import logging
import traceback
import bittensor as bt 
from dotenv import load_dotenv
from typing import Any
load_dotenv(override=True)

# ---------------- Config ----------------
def get_conf(key, default=None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        raise ValueError(f"{key} not set.\nYou must set env var: {key} in .env")
    return v or default

# ---------------- Logging ----------------
TRACE = 5
logging.addLevelName(TRACE, "TRACE")
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("affine")
def setup_logging(verbosity: int):
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore", "uvicorn.access"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
def info(): setup_logging(1)
def debug(): setup_logging(2)
def trace(): setup_logging(3)

# ---------------- Subtensor ----------------
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR is None:
        logger.trace("Making Bittensor connection...")
        if bt is None:
            raise RuntimeError("bittensor not installed")
        SUBTENSOR = bt.async_subtensor(get_conf('SUBTENSOR_ENDPOINT', default='wss://lite.sub.latent.to:443'))
        try:
            await SUBTENSOR.initialize()
            logger.trace("Connected")
        except Exception as e:
            os._exit(1)
    return SUBTENSOR

# ---------------- CHUTES ----------------
TERMINAL = {400, 404, 410}
async def CHUTES(prompt, model: str = "unsloth/gemma-3-12b-it", slug: str = "llm", timeout=150, retries=0, backoff=1) -> str | None:
    url = f"https://{slug}.chutes.ai/v1/chat/completions"
    hdr = {"Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
    if aiohttp is None:
        return None
    client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    sem = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "16")))
    for attempt in range(1, retries + 2):
        try:
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            async with sem, client.post(url, json=payload, headers=hdr, timeout=timeout) as r:
                txt = await r.text(errors="ignore")
                if r.status in TERMINAL:
                    return None
                r.raise_for_status()
                content = (await r.json())["choices"][0]["message"]["content"]
                return content
        except Exception:
            if attempt > retries:
                return None
            await asyncio.sleep(backoff * 2 ** (attempt - 1) * (1 + random.uniform(-0.1, 0.1)))
            
async def SEARCH(query: str) -> str:
    return "results from sn13"

# ---------------- Tools ----------------
from .docker import *
    
# -------------- Host communication helpers --------------
def _base_url(base_url: Optional[str] = None) -> str:
    if base_url: return base_url.rstrip("/")
    env = os.getenv("RUNNER_BASE_URL")
    if env:
        return env.rstrip("/")
    try:
        if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
            return "http://host.docker.internal:5005"
    except Exception:
        pass
    return "http://127.0.0.1:5005"

def call_host(path: str, payload: dict, *, base_url: Optional[str] = None, timeout: int = 60):
    root = _base_url(base_url)
    url = f"{root}{path if path.startswith('/') else '/' + path}"
    headers = {"x-sn1-token": os.getenv("SN1_TOKEN", "")}
    resp = requests.post(url, json=payload, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.json()

def rpc(method: str, *args, base_url: Optional[str] = None, timeout: int = 60, **kwargs):
    payload = {"method": method, "args": list(args), "kwargs": kwargs}
    data = call_host("/rpc", payload, base_url=base_url, timeout=timeout)
    if isinstance(data, dict) and data.get("ok") is True:
        return data.get("result")
    raise RuntimeError((isinstance(data, dict) and data.get("error")) or "remote error")

def out_func(*, x: int = 1) -> int:
    return x + 1

async def llm_tool(*, prompt: str, model: str = "unsloth/gemma-3-12b-it", timeout: int = 150):
    return await CHUTES(prompt, model=model, timeout=timeout)

async def search(*, query: str):
    return await SEARCH( query = query)

class tools:
    @staticmethod
    def out_func(*, x: int = 1, base_url: Optional[str] = None, timeout: int = 60):
        return rpc("out_func", x=x, base_url=base_url, timeout=timeout)
    
    @staticmethod
    def llm(*, prompt: str, model: str = "unsloth/gemma-3-12b-it", base_url: Optional[str] = None, timeout: int = 60):
        return rpc("llm", prompt=prompt, model=model, base_url=base_url, timeout=timeout)
    
    @staticmethod
    def search(*, query: str = None):
        return rpc("search", query=query)

# Register tools.
def _register_default_methods() -> None:
    from .server import register
    register("out_func", out_func)
    register("llm", llm_tool)
    register("searc", search)
_register_default_methods()

# ---------------- CLI ----------------
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    setup_logging(verbose)

# ---------------- Watchdog ----------------
HEARTBEAT = time.monotonic()
async def watchdog(timeout: int = 300):
    global HEARTBEAT
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)

# ---------------- Runner ----------------
@cli.command("runner")
def runner():
    async def run_server():
        # You can switch to a Unix socket by using uds="/tmp/sn1.sock" and mounting it into containers
        config = uvicorn.Config("sn1:app", host="0.0.0.0", port=5005, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    async def _run():
        # Example loop that creates a container and calls into it
        from .docker import Container
        while True:
            global HEARTBEAT
            try:
                HEARTBEAT = time.monotonic()
                # Demo: call functions on arbitrary script
                with Container("gen.py") as c:
                    # These will work if gen.py defines matching functions via boot.py @entrypoint or raw defs
                    print(c.__list__() if hasattr(c, "__list__") else [])
                    print(c.func(z="cat"))
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"runner error: {e}; retrying...")
                await asyncio.sleep(5)

    async def main():
        await asyncio.gather(_run(), watchdog(timeout=60 * 10), run_server())

    asyncio.run(main())

@cli.command("validator")
def validator():
    async def _run():
        while True:
            global HEARTBEAT
            try:
                HEARTBEAT = time.monotonic()
                logging.debug('debug')
                await asyncio.sleep(3)
            except asyncio.CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"runner error: {e}; retrying...")
                await asyncio.sleep(5)

    async def main():
        await asyncio.gather(_run(), watchdog(timeout=60 * 10))
    asyncio.run(main())
