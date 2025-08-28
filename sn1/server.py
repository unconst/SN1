# sn1/server.py
import time
import uvicorn
import asyncio 
import secrets 
import requests
import threading
from pydantic import BaseModel
from typing import Any, Callable, Dict
from fastapi import FastAPI, Request, HTTPException, Depends

# ---------------- Registry & Tokens ----------------
_METHODS: Dict[str, Callable[..., Any]] = {}
_TOKENS: Dict[str, float] = {}  # token -> expiry (epoch seconds)
_PER_TOKEN_LIMIT: Dict[str, asyncio.Semaphore] = {}
_GLOBAL_LIMIT = asyncio.Semaphore(200)  # global backpressure, tune as needed

def register(name: str, fn: Callable[..., Any]):
    _METHODS[name] = fn

def issue_token(ttl_s: int = 3600, per_token_limit: int = 16) -> str:
    tok = secrets.token_urlsafe(24)
    _TOKENS[tok] = time.time() + ttl_s
    _PER_TOKEN_LIMIT[tok] = asyncio.Semaphore(per_token_limit)
    return tok

def validate_token(req: Request) -> str:
    tok = req.headers.get("x-sn1-token")
    if not tok or tok not in _TOKENS or _TOKENS[tok] < time.time():
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return tok

class RpcIn(BaseModel):
    method: str
    args: list[Any] = []
    kwargs: dict[str, Any] = {}

app = FastAPI()

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/rpc")
async def rpc_call(payload: RpcIn, tok: str = Depends(validate_token)):
    fn = _METHODS.get(payload.method)
    if not fn:
        raise HTTPException(status_code=404, detail=f"unknown method {payload.method}")
    async with _GLOBAL_LIMIT, _PER_TOKEN_LIMIT[tok]:
        try:
            res = fn(*payload.args, **payload.kwargs)
            if asyncio.iscoroutine(res):
                res = await res
            return {"ok": True, "result": res}
        except Exception as e:
            return {"ok": False, "error": str(e)}

# ---------------- Local server bootstrap ----------------
_SERVER_BOOT_LOCK = threading.Lock()

def _healthz_local_ok(host: str = "127.0.0.1", port: int = 5005, timeout: float = 0.5) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/healthz", timeout=timeout)
        return r.ok
    except Exception:
        return False

def ensure_server_running(*, host: str = "0.0.0.0", port: int = 5005, startup_timeout_s: float = 10.0) -> None:
    """
    Ensure the FastAPI server is running in this process. If not, start
    a uvicorn server in a background thread bound to host:port.
    """
    if _healthz_local_ok(port=port):
        return
    with _SERVER_BOOT_LOCK:
        if _healthz_local_ok(port=port):
            return
        def _run_server():
            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            asyncio.run(server.serve())

        t = threading.Thread(target=_run_server, daemon=True)
        t.start()

    # Wait until the server is up
    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if _healthz_local_ok(port=port):
            return
        time.sleep(0.1)
    raise RuntimeError("Failed to start local SN1 server")
