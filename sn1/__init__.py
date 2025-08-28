
import os
import time
import click
import asyncio
import logging
import traceback
import bittensor as bt
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable

# --------------------------------------------------------------------------- #
#                               Config                                       #
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
def get_conf(key, default=None) -> Any:
    v = os.getenv(key); 
    if not v and default is None:
        raise ValueError(f"{key} not set.\nYou must set env var: {key} in .env")
    return v or default
# --------------------------------------------------------------------------- #
#                               Logging                                       #
# --------------------------------------------------------------------------- #
TRACE  = 5
logging.addLevelName(TRACE, "TRACE")
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("affine")
def setup_logging(verbosity: int):
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
def info():setup_logging(1)
def debug():setup_logging(2)
def trace():setup_logging(3)

# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.trace("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor( get_conf('SUBTENSOR_ENDPOINT', default='wss://lite.sub.latent.to:443') )
        try:
            await SUBTENSOR.initialize()
            logger.trace("Connected")
        except Exception as e:
            os.exit(1)
    return SUBTENSOR

# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    setup_logging(verbose)
    
# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #
HEARTBEAT = time.monotonic()
async def watchdog(timeout: int = 300):
    global HEARTBEAT
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)

# --------------------------------------------------------------------------- #
#                               Runner                                        #
# --------------------------------------------------------------------------- #
@cli.command("runner")
def runner():
    async def _run():
        while True:
            global HEARTBEAT
            try:
                # Get the latest scripts
                # Iterate over the scripts
                # Create a docker for the script
                # Inject the script into the docker
                # Create a query
                # Query the script to run the generator
                HEARTBEAT = time.monotonic()
                logging.debug('debug')
                await asyncio.sleep(3)
            except asyncio.CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"runner error: {e}; retrying...")
                subtensor = await get_subtensor()
                await asyncio.sleep(5)

    async def main():
        await asyncio.gather(_run(), watchdog(timeout=60*10))
    asyncio.run(main())