# sn1/__init__.py
from __future__ import annotations
import os
import re
import uuid
import time
import click
import random
import aiohttp
import uvicorn
import asyncio
import logging
import tempfile
import aiofiles
import traceback
import bittensor as bt 
from pathlib import Path
from dotenv import load_dotenv
from typing import Any
load_dotenv(override=True)

NETUID = 1

# ---------------- Config ----------------
def get_conf(key, default=None, except_input: bool = False) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        if except_input:
            v = input(f"Enter value for {key}: ")
            os.environ[key] = v
            return v
        raise ValueError(f"{key} not set.\nYou must set env var: {key} in .env")
    return v or default

# ---------------- Logging ----------------
TRACE = 5
logging.addLevelName(TRACE, "TRACE")
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("sn1")
def setup_logging(verbosity: int):
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore", "uvicorn.access"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level, format="[%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
    sem = asyncio.Semaphore(int(os.getenv("SN1_HTTP_CONCURRENCY", "16")))
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

def func(*, x: int = 1) -> int:
    return x + 1

async def llm_tool(*, prompt: str, model: str = "unsloth/gemma-3-12b-it", timeout: int = 150):
    return await CHUTES(prompt, model=model, timeout=timeout)

class tools:
    @staticmethod
    def func(*, x: int = 1, base_url: Optional[str] = None, timeout: int = 60):
        return rpc("out_func", x=x, base_url=base_url, timeout=timeout)
    
    @staticmethod
    def llm(*, prompt: str, model: str = "unsloth/gemma-3-12b-it", base_url: Optional[str] = None, timeout: int = 60):
        return rpc("llm", prompt=prompt, model=model, base_url=base_url, timeout=timeout)
    
# Register tools.
def _register_default_methods() -> None:
    from .server import register
    register("func", func)
    register("llm", llm_tool)
_register_default_methods()

# ---------------- Get Agent. ----------------
async def pull_agent(uid: int) -> str:
    try:
        logger.info(f"Starting to pull agent for uid: {uid}")
        sub = await get_subtensor()
        commit = await sub.get_revealed_commitment(netuid = NETUID, uid = uid)
        g = commit[0][1]
        block = commit[0][0]
        if g.startswith("http") and "api.github.com" not in g:
            g = f"https://api.github.com/gists/{g.rstrip('/').split('/')[-1]}"
            logger.debug(f"Converted to gist URL: {g}")
        if not g.startswith("http"):
            g = f"https://api.github.com/gists/{g}"
            logger.debug(f"Added gist prefix: {g}")
        logger.info(f"Final gist URL: {g}")
        async with aiohttp.ClientSession() as s:
            logger.debug(f"Making request to gist URL: {g}")
            async with s.get(g) as r:
                data = await r.json()
            logger.debug(f"Got gist data: {list(data.get('files', {}).keys())}")
            meta = next(iter(data["files"].values()))
            logger.debug(f"Got file metadata: {meta.keys()}")
            content = meta.get("content")
            if content is None or meta.get("truncated"):
                logger.debug(f"Content is None or truncated, fetching raw content from: {meta['raw_url']}")
                async with s.get(meta["raw_url"]) as r:
                    content = await r.text()
            logger.debug(f"Got content, length: {len(content) if content else 0}")
        dir = f"agents/{uid}/{block}/" 
        Path(dir).mkdir(parents=True, exist_ok=True)
        name = f"{dir}agent.py"
        logger.debug(f"Writing agent to: {name}")
        async with aiofiles.open(name, "w", encoding="utf-8") as f:
            await f.write(content or "")
        resolved_path = str(Path(name).resolve())
        logger.info(f"Successfully pulled agent to: {resolved_path}")
        return resolved_path
    except Exception as e:
        logger.warning(f'Failed pulling agent on uid: {uid} with error: {e}')
        return None
    
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
    
    
@cli.command("push")
@click.argument("path", default="agents/base_agent.py")
def push( path:str ):
    coldkey = get_conf("BT_WALLET_COLD", except_input=True)
    hotkey = get_conf("BT_WALLET_HOT", except_input=True)
    github_token = get_conf("GITHUB_TOKEN", except_input=True)
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)
    async def main():
        logger.info('Loading chain state ...')
        sub = await get_subtensor()
        metagraph = await sub.metagraph(NETUID)
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            logger.warning(f"Not registered, first register your wallet `btcli subnet register --netuid {NETUID} --wallet.name {coldkey} --hotkey {hotkey}`")
            os._exit(1)
        logger.info(f'UID: {metagraph.hotkeys.index(wallet.hotkey.ss58_address)}')

        with open(path, 'r') as f:
            content = f.read()
        scheme = "token" if github_token.startswith(("ghp_", "github_pat_")) else "Bearer"
        headers = {
            "Authorization": f"{scheme} {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sn1-cli"
        }
        gist_data = {"description": "Agent code", "public": True, "files": {os.path.basename(path): {"content": content}}}
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.github.com/gists", json=gist_data, headers=headers) as resp:
                if resp.status != 201:
                    try:
                        error_json = await resp.json()
                        error_msg = error_json.get("message") or str(error_json)
                    except Exception:
                        error_msg = await resp.text()
                    raise RuntimeError(
                        f"Failed to create gist ({resp.status}): {error_msg}. Ensure your GITHUB_TOKEN is valid and has 'gist' scope, visit: https://github.com/settings/tokens/new"
                    )
                gist_url = (await resp.json())["html_url"]
                logger.info(f"Created gist: {gist_url}")
        
        await sub.set_reveal_commitment(wallet=wallet, netuid=NETUID, data=gist_url, blocks_until_reveal = 1)
        logger.info(f"Committed gist URL to blockchain.")
    
    asyncio.run(main())
    
@cli.command("pull")
@click.argument("uid", type=int, required=False)
def pull( uid:int = None):
    if uid is not None:
        asyncio.run(pull_agent(uid))
    else:
        async def pull_all():
            sub = await get_subtensor()
            metagraph = await sub.metagraph(NETUID)
            for uid in metagraph.uids:
                await pull_agent(int(uid))
        asyncio.run(pull_all())

# ---------------- Runner ----------------
@cli.command("validator")
def validator():
    coldkey = get_conf("BT_WALLET_COLD", except_input=True)
    hotkey = get_conf("BT_WALLET_HOT", except_input=True)
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)
    logger.debug(f"Validator initialized with wallet: {coldkey}/{hotkey}")

    async def _run():
        # Example loop that creates a container and calls into it
        from .docker import Container
        logger.debug("Starting validator main loop")
        while True:
            global HEARTBEAT
            try:
                
                SAMPLES = 10                
                HEARTBEAT = time.monotonic()
                logger.debug(f"Heartbeat updated: {HEARTBEAT}")
                sub = await get_subtensor()
                logger.debug("Subtensor connection established")
                
                metagraph = await sub.metagraph(NETUID)
                uids = [ int(uid) for uid in metagraph.uids]
                weights = [ 0 for _ in metagraph.uids ]
                logger.debug(f"Loaded metagraph with {len(uids)} UIDs: {uids}")
                
                for uid in uids:
                    logger.debug(f"Processing UID {uid}")
                    gen_tmp_file: str = await pull_agent( uid )
                    logger.debug(f"Retrieved agent file for UID {uid}: {gen_tmp_file}")
                    gen_tmp_file = "gen.py"
                    logger.debug(f"Using hardcoded agent file: {gen_tmp_file}")
                    with Container( gen_tmp_file ) as c:
                        logger.debug(f"Created container for UID {uid}")
                        success = 0
                        for sample_idx in range(SAMPLES):
                            try:
                                x = random.random()
                                y = random.random()
                                z = x * y
                                prompt = f"what is {x} * {y}?, return you answer like <Answer>12.232</Answer>"
                                logger.debug(f"UID {uid} sample {sample_idx}: testing {x} * {y} = {z}")
                                response = c.llm( prompt = prompt )
                                logger.debug(f"UID {uid} sample {sample_idx}: got response: {response}")
                                match = re.search(r'<Answer>(.*?)</Answer>', response)
                                if match:
                                    parsed_answer = float(match.group(1))
                                    if abs(parsed_answer - z) <= 1e-6:
                                        success += 1
                                        logger.debug(f"UID {uid} sample {sample_idx}: correct answer {parsed_answer}")
                                    else:
                                        logger.debug(f"UID {uid} sample {sample_idx}: incorrect answer {parsed_answer}, expected {z}")
                                else:
                                    logger.debug(f"UID {uid} sample {sample_idx}: no answer found in response")
                            except Exception as e: 
                                logger.debug(f"UID {uid} sample {sample_idx}: error - {e}")
                        weights[uid] = float(success)/SAMPLES
                        logger.debug(f"UID {uid}: scored {success}/{SAMPLES} = {weights[uid]}")
                                                
                logger.debug(f"Setting weights: UIDs={uids}, weights={weights}")
                await sub.set_weights( 
                    wallet=wallet, 
                    netuid=NETUID, 
                    weights=weights, 
                    uids=uids,
                    wait_for_inclusion=False,
                    wait_for_finalization=False
                )
                logger.debug("Weights successfully set")
                
            except asyncio.CancelledError:
                logger.debug("Validator loop cancelled")
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"runner error: {e}; retrying...")
                await asyncio.sleep(5)

    async def main():
        logger.debug("Starting validator with watchdog")
        await asyncio.gather(_run(), watchdog(timeout=60 * 10))

    asyncio.run(main())


