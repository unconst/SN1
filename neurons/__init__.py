from __future__ import annotations
import os
import re
import time
import click
import random
import aiohttp
import asyncio
import aiofiles
import traceback
import logging
import bittensor as bt
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from sn1 import Container

logger = logging.getLogger("neurons")

NETUID = 1

# ---------------- Subtensor ----------------
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR is None:
        logger.debug("Making Bittensor connection...")
        if bt is None:
            raise RuntimeError("bittensor not installed")
        SUBTENSOR = bt.async_subtensor(os.getenv('SUBTENSOR_ENDPOINT', 'wss://lite.sub.latent.to:443'))
        try:
            await SUBTENSOR.initialize()
            logger.debug("Connected")
        except Exception as e:
            os._exit(1)
    return SUBTENSOR


# ---------------- Get Agent. ----------------
async def pull_agent(uid: int) -> Optional[str]:
    try:
        logger.info(f"Starting to pull agent for uid: {uid}")
        sub = await get_subtensor()
        commit = await sub.get_revealed_commitment(netuid=NETUID, uid=uid)
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
            async with s.get(g) as r:
                data = await r.json()
            meta = next(iter(data["files"].values()))
            content = meta.get("content")
            if content is None or meta.get("truncated"):
                async with s.get(meta["raw_url"]) as r:
                    content = await r.text()
        dir = f"agents/{uid}/{block}/"
        Path(dir).mkdir(parents=True, exist_ok=True)
        name = f"{dir}agent.py"
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
@click.option('--log-level', type=click.Choice(['CRITICAL','ERROR','WARNING','INFO','DEBUG'], case_sensitive=False), default=None, help='Logging level (or set LOG_LEVEL env)')
def cli(log_level: Optional[str]):
    load_dotenv(override=True)
    level_name = (log_level or os.getenv('LOG_LEVEL') or 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


@cli.command("push")
@click.argument("path", default="agents/base_agent.py")
def push(path: str):
    def require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value
    coldkey = require_env("BT_WALLET_COLD")
    hotkey = require_env("BT_WALLET_HOT")
    github_token = require_env("GITHUB_TOKEN")
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
            "User-Agent": "neurons-cli"
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

        await sub.set_reveal_commitment(wallet=wallet, netuid=NETUID, data=gist_url, blocks_until_reveal=1)
        logger.info(f"Committed gist URL to blockchain.")

    asyncio.run(main())


@cli.command("pull")
@click.argument("uid", type=int, required=False)
def pull(uid: int = None):
    if uid is not None:
        asyncio.run(pull_agent(uid))
    else:
        async def pull_all():
            sub = await get_subtensor()
            metagraph = await sub.metagraph(NETUID)
            for uid in metagraph.uids:
                await pull_agent(int(uid))
        asyncio.run(pull_all())


# ---------------- Watchdog ----------------
HEARTBEAT = time.monotonic()
async def watchdog(timeout: int = 300):
    global HEARTBEAT
    while True:
        await asyncio.sleep(max(1, timeout // 3))
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)

@cli.command("validator")
def validator():
    def require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value
    coldkey = require_env("BT_WALLET_COLD")
    hotkey = require_env("BT_WALLET_HOT")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)
    logger.debug(f"Validator initialized with wallet: {coldkey}/{hotkey}")

    async def _run():
        global HEARTBEAT
        logger.debug("Starting validator main loop")
        while True:
            try:
                HEARTBEAT = time.monotonic()
                SAMPLES = 10
                sub = await get_subtensor()
                logger.debug("Subtensor connection established")

                metagraph = await sub.metagraph(NETUID)
                uids = [int(uid) for uid in metagraph.uids]
                weights = [0 for _ in metagraph.uids]
                logger.debug(f"Loaded metagraph with {len(uids)} UIDs: {uids}")

                for uid in uids:
                    update_heartbeat()
                    logger.debug(f"Processing UID {uid}")
                    gen_tmp_file: str = await pull_agent(uid)
                    logger.debug(f"Retrieved agent file for UID {uid}: {gen_tmp_file}")
                    gen_tmp_file = "gen.py"
                    logger.debug(f"Using hardcoded agent file: {gen_tmp_file}")
                    with Container(gen_tmp_file) as c:
                        logger.debug(f"Created container for UID {uid}")
                        success = 0
                        for sample_idx in range(SAMPLES):
                            try:
                                x = random.random()
                                y = random.random()
                                z = x * y
                                prompt = f"what is {x} * {y}?, return you answer like <Answer>12.232</Answer>"
                                logger.debug(f"UID {uid} sample {sample_idx}: testing {x} * {y} = {z}")
                                response = c.llm(prompt=prompt)
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
                        weights[uid] = float(success) / SAMPLES
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
        await asyncio.gather(
            _run(),
            watchdog(),
            return_exceptions=True
        )
    asyncio.run(main())



