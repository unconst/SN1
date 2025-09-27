from __future__ import annotations
import requests
from pathlib import Path
from argparse import Namespace
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
import os, sys, time, uuid, json, asyncio, logging, shutil, subprocess, shlex, secrets, threading, importlib, importlib.util, inspect, types, io, contextlib

# Library logging: expose a named logger without configuring handlers/levels.
logger = logging.getLogger("sn1")

# ---------------- Host RPC client ----------------
def _base_url(base_url: Optional[str] = None) -> str:
    if base_url:
        return base_url.rstrip("/")
    env = os.getenv("RUNNER_BASE_URL")
    if env:
        return env.rstrip("/")
    try:
        if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
            return "http://127.0.0.1:5005"
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
        # Surface any stdout/stderr captured on the server side
        std_out = data.get("stdout")
        std_err = data.get("stderr")
        try:
            if std_out:
                print(std_out, end="")
            if std_err:
                print(std_err, end="", file=sys.stderr)
        except Exception:
            pass
        return data.get("result")
    raise RuntimeError((isinstance(data, dict) and data.get("error")) or "remote error")

class _ToolsProxy:
    def __getattr__(self, method: str):
        def _call(*args, **kwargs):
            base_url = kwargs.pop("base_url", None)
            call_timeout = kwargs.pop("timeout", 60)
            if len(args) == 1 and "prompt" not in kwargs:
                kwargs["prompt"] = args[0]
                args = ()
            return rpc(method, *args, base_url=base_url, timeout=call_timeout, **kwargs)
        return _call

tools = _ToolsProxy()
# Expose a decorator for registering host-callable tools
# Usage:
#   @sn1.tool              -> registers function under its name
#   @sn1.tool(name="foo") -> registers under custom name
def tool(_fn: Callable | None = None, *, name: str | None = None):
    return declare_tool(_fn, name=name)

# Entrypoints: decorator and compatibility alias for sn1.boot
ENTRYPOINTS: dict[str, Callable[..., Any]] = {}
# Ensure user code can `from sn1.boot import entrypoint`
sys.modules.setdefault("sn1.boot", sys.modules[__name__])

def entrypoint(_fn: Callable | None = None, *, name: str | None = None):
    def _decorator(fn: Callable) -> Callable:
        # Register entrypoints under a dedicated namespace to avoid colliding with tools
        ep_name = f"entry:{name or fn.__name__}"
        register(ep_name, fn)
        ENTRYPOINTS[name or fn.__name__] = fn
        return fn
    if _fn is None:
        return _decorator
    return _decorator(_fn)

# ---------------- Env loader ----------------
def load_env(env_or_path: str) -> Namespace:
    p = Path(env_or_path)
    if p.exists():
        base_dir = p if p.is_dir() else p.parent
        tools_file = base_dir / "tools.py"
        if not tools_file.exists():
            raise RuntimeError(f"Invalid environment path: {base_dir}. Expected tools.py")

        def _load_module_from_file(name: str, file_path: Path):
            spec = importlib.util.spec_from_file_location(name, str(file_path))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed loading module from {file_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        tools_module = _load_module_from_file(f"sn1_env_{uuid.uuid4().hex}_tools", tools_file)
    else:
        tools_module = importlib.import_module(f"environments.{env_or_path}.tools")

    if hasattr(tools_module, "register_tools"):
        tools_module.register_tools()

    allowed_methods = set(getattr(tools_module, "ALLOWED_METHODS", set()))
    docker_image = getattr(tools_module, "DOCKER_IMAGE", "python:3.11-slim")
    entrypoint = getattr(tools_module, "ENTRYPOINT", "solve")
    defaults = getattr(tools_module, "TOOL_DEFAULTS", None)

    return Namespace(
        docker_image=docker_image,
        entrypoint=entrypoint,
        allowed_methods=allowed_methods,
        defaults=defaults,
    )


# ---------------- RPC server (FastAPI) ----------------
_METHODS: dict[str, Any] = {}
_TOKEN_META: dict[str, dict[str, Any]] = {}
_GLOBAL_LIMIT = asyncio.Semaphore(200)

@dataclass
class Context:
    token: str
    method: str
    request_id: str
    created_at: float
    headers: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

def register(name: str, fn: Any) -> None:
    _METHODS[name] = fn

def declare_tool(_fn: Callable | None = None, *, name: str | None = None):
    """Decorator to register a function as a callable tool via RPC.

    Usage:
        @declare_tool
        def my_tool(...): ...

        or with explicit name:
        @declare_tool(name="custom")
        def my_tool(...): ...
    """
    def _decorator(fn: Callable) -> Callable:
        register(name or fn.__name__, fn)
        return fn
    if _fn is None:
        return _decorator
    return _decorator(_fn)

def issue_token(ttl_s: int = 3600, per_token_limit: int = 16, allowed_methods: set[str] | None = None) -> str:
    tok = secrets.token_urlsafe(24)
    _TOKEN_META[tok] = {
        "expiry": time.time() + ttl_s,
        "allowed": set(allowed_methods or []),
        "sem": asyncio.Semaphore(per_token_limit),
    }
    return tok

def _validate_token(req: Request) -> str:
    tok = req.headers.get("x-sn1-token")
    meta = _TOKEN_META.get(tok)
    if not tok or not meta or meta["expiry"] < time.time():
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

@app.get("/methods")
async def methods():
    return {"methods": sorted(_METHODS.keys())}

def _fn_wants_ctx(fn: Callable) -> bool:
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if not params:
            return False
        p0 = params[0]
        if p0.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return False
        anno = p0.annotation
        if isinstance(anno, type):
            anno_name = getattr(anno, "__name__", "")
        else:
            anno_name = str(anno)
        wants = (p0.name in {"ctx", "_ctx", "context"}) or (anno_name.lower() == "context" or "sn1.Context" in anno_name)
        return wants
    except Exception:
        return False

@app.post("/rpc")
async def rpc_call(payload: RpcIn, req: Request, tok: str = Depends(_validate_token)):
    fn = _METHODS.get(payload.method)
    if not fn:
        raise HTTPException(status_code=404, detail=f"unknown method {payload.method}")
    meta = _TOKEN_META[tok]
    allowed = meta["allowed"]
    if allowed and payload.method not in allowed:
        raise HTTPException(status_code=403, detail=f"method {payload.method} not allowed for this token")
    async with _GLOBAL_LIMIT, meta["sem"]:
        out_buf, err_buf = io.StringIO(), io.StringIO()
        try:
            ctx = Context(
                token=tok,
                method=payload.method,
                request_id=uuid.uuid4().hex,
                created_at=time.time(),
                headers={k: v for k, v in req.headers.items()},
            )
            # Merge client-sent context into ctx.meta (namespaced under __ctx in kwargs)
            extra_ctx = payload.kwargs.pop("__ctx", None)
            if isinstance(extra_ctx, dict):
                try:
                    ctx.meta.update(extra_ctx)
                except Exception:
                    pass
            wants_ctx = _fn_wants_ctx(fn)
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                if inspect.iscoroutinefunction(fn):
                    if wants_ctx:
                        res = await fn(ctx, *payload.args, **payload.kwargs)
                    else:
                        res = await fn(*payload.args, **payload.kwargs)
                else:
                    if wants_ctx:
                        res = await run_in_threadpool(fn, ctx, *payload.args, **payload.kwargs)
                    else:
                        res = await run_in_threadpool(fn, *payload.args, **payload.kwargs)
            return {"ok": True, "result": res, "stdout": out_buf.getvalue(), "stderr": err_buf.getvalue()}
        except Exception as e:
            logger.error(f"rpc error in {payload.method}: {e}")
            return {"ok": False, "error": str(e), "stdout": out_buf.getvalue(), "stderr": err_buf.getvalue()}

_SERVER_BOOT_LOCK = threading.Lock()

def _healthz_local_ok(host: str = "127.0.0.1", port: int = 5005, timeout: float = 0.5) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/healthz", timeout=timeout)
        return r.ok
    except Exception:
        return False

def ensure_server_running(*, host: str = "0.0.0.0", port: int = 5005, startup_timeout_s: float = 10.0) -> None:
    if _healthz_local_ok(port=port):
        return
    with _SERVER_BOOT_LOCK:
        if _healthz_local_ok(port=port):
            return
        def _run_server():
            import uvicorn
            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            asyncio.run(server.serve())
        t = threading.Thread(target=_run_server, daemon=True)
        t.start()
    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if _healthz_local_ok(port=port):
            return
        time.sleep(0.1)
    raise RuntimeError("Failed to start local SN1 server")

# --- Auto-load tools/entrypoints when running in a container ---
def _load_module_from_file(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _autoregister_from_env() -> None:
    try:
        # Support both module names and file paths via env vars
        mods = (os.getenv("SN1_IMPORT_MODULES") or "").strip()
        paths = (os.getenv("SN1_IMPORT_PATHS") or "").strip()
        if mods:
            for m in [x.strip() for x in mods.split(",") if x.strip()]:
                importlib.import_module(m)
                logger.info(f"sn1: imported module {m}")
        if paths:
            for i, p in enumerate([x.strip() for x in paths.split(":" ) if x.strip()]):
                _p = Path(p)
                if _p.exists():
                    _load_module_from_file(f"sn1_autoload_{i}_{_p.stem}", _p)
                    logger.info(f"sn1: loaded file {_p}")
        # Fallback to common /app layout
        if not mods and not paths:
            for candidate in ["/app/tools.py", "/app/agent.py"]:
                if os.path.exists(candidate):
                    _load_module_from_file(f"sn1_autoload_{Path(candidate).stem}", Path(candidate))
                    logger.info(f"sn1: loaded file {candidate}")
    except Exception as e:
        logger.warning(f"autoregister failed: {e}")

def _bootstrap_container_token() -> None:
    fixed = os.getenv("SN1_TOKEN")
    if fixed and fixed not in _TOKEN_META:
        # Accept this token for all methods with a long expiry
        _TOKEN_META[fixed] = {
            "expiry": time.time() + 365 * 24 * 3600,
            "allowed": set(),
            "sem": asyncio.Semaphore(64),
        }
        logger.info("sn1: accepted container SN1_TOKEN for RPC")

@app.on_event("startup")
async def _on_startup():
    _bootstrap_container_token()
    _autoregister_from_env()
    logger.info(f"sn1: startup complete; registered {len(_METHODS)} methods")

# Always register a builtin lister
def _list_methods() -> list[str]:
    return sorted(list(_METHODS.keys()))

register("__list__", _list_methods)

# ---------------- Docker helpers and Container ----------------
def _get_docker_bin() -> str:
    docker_path = os.getenv("DOCKER_BIN") or "/usr/bin/docker"
    if os.path.exists(docker_path):
        return docker_path
    fallback = shutil.which("docker")
    if fallback:
        return fallback
    raise RuntimeError("docker binary not found. Ensure docker is installed and mounted.")

def _run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    logger.debug(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=capture_output, text=True, check=check)

def _docker(*args: str, capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    docker_bin = _get_docker_bin()
    return _run([docker_bin, *args], capture_output=capture_output, check=check)

def create_running_container(image: str, name: str, env: dict[str, str] | None = None, extra_args: list[str] | None = None) -> str:
    env = env or {}
    extra_args = extra_args or []
    try:
        _docker("pull", image)
    except Exception as e:
        logger.warning(f"docker pull failed (continuing): {e}")
    args = [
        "create", "--entrypoint", "/bin/sh", "--name", name,
        "--add-host=host.docker.internal:host-gateway",
    ]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    args += extra_args
    args += [image, "-c", "sleep infinity"]
    create = _docker(*args)
    container_id = create.stdout.strip() or name
    _docker("start", container_id)
    return container_id

def copy_into_container(container_id: str, src_path: str, dest_path: str):
    _docker("cp", src_path, f"{container_id}:{dest_path}")

def exec_in_container(container_id: str, command: str) -> tuple[int, str, str]:
    proc = _docker("exec", container_id, "/bin/sh", "-lc", command, capture_output=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr

def exec_in_container_detach(container_id: str, command: str) -> None:
    _docker("exec", "-d", container_id, "/bin/sh", "-lc", command, capture_output=True, check=False)

def stop_and_remove_container(container_id: str):
    _docker("rm", "-f", container_id, check=False)

 
def _container_host_port(container_id: str, internal_port: int = 5005) -> int:
    try:
        out = _docker("port", container_id, f"{internal_port}/tcp").stdout.strip()
        # Expected formats like: "0.0.0.0:49153" or ":::49153". Take the last colon segment.
        if out:
            last = out.split()[-1]
            port = int(last.split(":")[-1])
            return port
    except Exception as e:
        logger.warning(f"failed to discover published port: {e}")
    raise RuntimeError("could not determine published host port")

def run_detached_container(image: str, name: str, env: dict[str, str] | None = None, extra_args: list[str] | None = None) -> str:
    env = env or {}
    extra_args = extra_args or []
    try:
        _docker("pull", image)
    except Exception as e:
        logger.warning(f"docker pull failed (continuing): {e}")
    args = [
        "run", "-d", "--name", name,
        "--add-host=host.docker.internal:host-gateway",
        "-p", "0:5005",
    ]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    args += extra_args
    args += [image]
    run = _docker(*args)
    container_id = run.stdout.strip() or name
    return container_id

class Container:
    def __init__(
        self,
        agent: str,
        image: str | None = None,
        *,
        spec: Any | None = None,
        python_path: str = "/opt/venv/bin/python",
        base_url: Optional[str] = None,
        token_ttl: int = 3600,
        allowed_methods: set[str] | None = None,
        ctx: Any | None = None,
    ) -> None:
        if spec is not None:
            image = getattr(spec, "docker_image", image)
            if allowed_methods is None:
                allowed_methods = set(getattr(spec, "allowed_methods", set()))
        self.image = image or "thebes1618/sn1:latest"
        self.local_script_path = os.path.abspath(agent) if agent else None
        self.in_container_script_path = f"/app/{os.path.basename(self.local_script_path)}" if self.local_script_path else None
        self.python_path = python_path
        self.container_name = f"sn1-{os.path.splitext(os.path.basename(self.local_script_path))[0]}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

        # Generate a shared token used by host<->container HTTP
        self.token = secrets.token_urlsafe(24)
        self.base_url = (base_url or "").rstrip("/")
        # User-provided call context (exposed locally and sent with RPC requests)
        self._ctx_source = ctx or {}
        try:
            self.ctx = types.SimpleNamespace(**(self._ctx_source if isinstance(self._ctx_source, dict) else {"value": self._ctx_source}))
        except Exception:
            self.ctx = types.SimpleNamespace()

        # Ensure local server if pointing to host
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            host = (parsed.hostname or "").lower()
            port = parsed.port or 5005
            if host in {"host.docker.internal", "localhost", "127.0.0.1"}:
                ensure_server_running(host="0.0.0.0", port=port)
        except Exception as _e:
            logger.warning(f"ensure_server_running failed: {_e}")

        logger.info(f"Preparing container {self.container_name} from {self.image}")

        if self.base_url:
            # External server; we don't manage a container
            self.container_id = ""
        else:
            if self.local_script_path and os.path.exists(self.local_script_path):
                # Develop-from-file mode: prepare a container, copy files, and start uvicorn
                self.container_id = create_running_container(
                    self.image,
                    self.container_name,
                    env={"SN1_TOKEN": self.token},
                    extra_args=["-p", "0:5005"],
                )
                # Ensure working directory exists
                _ = exec_in_container(self.container_id, "mkdir -p /app")
                # Copy agent and tools
                copy_into_container(self.container_id, self.local_script_path, "/app/agent.py")
                tools_candidate = os.path.join(os.path.dirname(self.local_script_path), "tools.py")
                if os.path.exists(tools_candidate):
                    copy_into_container(self.container_id, tools_candidate, "/app/tools.py")
                # Copy our package into /app so `import sn1` works without pip install
                pkg_src = os.path.dirname(__file__)
                copy_into_container(self.container_id, pkg_src, "/app")
                # Ensure runtime deps exist (venv + fastapi + uvicorn)
                rc, out, err = exec_in_container(
                    self.container_id,
                    "python -m venv /opt/venv || true; "
                    "if ! /opt/venv/bin/python -c 'import uvicorn,fastapi,requests' 2>/dev/null; then "
                    "/opt/venv/bin/pip install --no-cache-dir --upgrade pip && "
                    "/opt/venv/bin/pip install --no-cache-dir fastapi uvicorn requests; fi",
                )
                if rc != 0:
                    tail = (err or "").strip().splitlines()[-10:]
                    snippet = ("\n".join(tail)).strip()
                    raise RuntimeError(f"Container dependency install failed:\n{snippet}")
                # Start server in background
                exec_in_container_detach(
                    self.container_id,
                    "cd /app && SN1_IMPORT_PATHS=/app/tools.py:/app/agent.py /opt/venv/bin/python -m uvicorn sn1:app --host 0.0.0.0 --port 5005",
                )
            else:
                # Image-only mode: assume image starts uvicorn sn1:app
                self.container_id = run_detached_container(
                    self.image,
                    self.container_name,
                    env={"SN1_TOKEN": self.token},
                )

            # Discover mapped host port and set base_url
            host_port = _container_host_port(self.container_id, 5005)
            self.base_url = f"http://127.0.0.1:{host_port}"
            logger.info(f"sn1: container {self.container_id[:12]} listening at {self.base_url}")

            # Wait for health before proceeding
            deadline = time.time() + 120.0
            while time.time() < deadline:
                try:
                    r = requests.get(f"{self.base_url}/healthz", timeout=0.5)
                    if r.ok:
                        break
                except Exception:
                    pass
                time.sleep(0.2)
            else:
                try:
                    r = requests.get(f"{self.base_url}/methods", timeout=0.5)
                    logger.warning(f"sn1: /methods before fail -> {getattr(r,'text',None)}")
                except Exception as e:
                    logger.warning(f"sn1: /methods request failed: {e}")
                raise RuntimeError("Agent HTTP server did not become ready in time")

        self._destroyed = False

    def _ensure_active(self) -> None:
        if self._destroyed:
            raise RuntimeError("Container has been destroyed")

    def _ctx_payload(self) -> dict[str, Any]:
        try:
            if isinstance(self._ctx_source, dict):
                return dict(self._ctx_source)
            if hasattr(self.ctx, "__dict__"):
                return dict(self.ctx.__dict__)
        except Exception:
            pass
        return {}

    def _call(self, entry: str, *args, **kwargs):
        self._ensure_active()
        prev = os.environ.get("SN1_TOKEN")
        os.environ["SN1_TOKEN"] = self.token
        try:
            method = f"entry:{entry}"
            # Include container-level ctx; allow per-call override via __ctx
            call_kwargs = dict(kwargs)
            call_ctx = self._ctx_payload()
            if "__ctx" in call_kwargs and isinstance(call_kwargs["__ctx"], dict):
                try:
                    merged = dict(call_ctx)
                    merged.update(call_kwargs["__ctx"])  # per-call takes precedence
                    call_ctx = merged
                except Exception:
                    pass
            call_kwargs["__ctx"] = call_ctx
            return rpc(method, *args, base_url=self.base_url, **call_kwargs)
        finally:
            if prev is None:
                os.environ.pop("SN1_TOKEN", None)
            else:
                os.environ["SN1_TOKEN"] = prev

    def __getattr__(self, name: str):
        def _caller(*args, **kwargs):
            return self._call(name, *args, **kwargs)
        return _caller

    def entries(self) -> list[str]:
        try:
            res = self._call("__list__")
            if isinstance(res, list):
                return [str(x) for x in res]
        except Exception:
            pass
        return []

    def destroy(self) -> None:
        if not self._destroyed:
            stop_and_remove_container(self.container_id)
            self._destroyed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.destroy()
        except Exception:
            pass
        return False

 