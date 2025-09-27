from __future__ import annotations
import requests
from pathlib import Path
from argparse import Namespace
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
import os, sys, time, uuid, asyncio, logging, shutil, subprocess, secrets, threading, importlib, importlib.util, inspect, types
from contextvars import ContextVar

# Minimal inline logging setup
logger = logging.getLogger("sn1")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [sn1] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(getattr(logging, (os.getenv("SN1_LOGLEVEL") or "INFO").upper(), logging.INFO))
def _lvl(x):
    return x if isinstance(x, int) else getattr(logging, str(x).upper(), logging.INFO)
def set_log_level(level: str | int) -> None:
    logger.setLevel(_lvl(level))

# ---------------- Host RPC client ----------------
def _base_url(base_url: Optional[str] = None) -> str:
    if base_url:
        return base_url.rstrip("/")
    env = os.getenv("RUNNER_BASE_URL")
    if env:
        return env.rstrip("/")
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

def rpc_raw(method: str, *args, base_url: Optional[str] = None, timeout: int = 60, **kwargs) -> dict[str, Any]:
    payload = {"method": method, "args": list(args), "kwargs": kwargs}
    return call_host("/rpc", payload, base_url=base_url, timeout=timeout)

class _ToolsProxy:
    def __getattr__(self, method: str):
        def _call(*args, **kwargs):
            base_url = kwargs.pop("base_url", None)
            call_timeout = kwargs.pop("timeout", 60)
            # If running inside container with an active request context, dispatch directly
            try:
                active_ctx = _CURRENT_CTX.get()
            except Exception:
                active_ctx = None
            if active_ctx is not None:
                fn = _METHODS.get(method)
                if fn is None:
                    raise RuntimeError(f"unknown method {method}")
                wants_ctx = _fn_wants_ctx(fn)
                if inspect.iscoroutinefunction(fn):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop is None:
                        if wants_ctx:
                            # Tools are trusted writers; temporarily enable writes
                            active_ctx._trusted_writer = True
                            try:
                                return asyncio.run(fn(active_ctx, *args, **kwargs))
                            finally:
                                active_ctx._trusted_writer = False
                        return asyncio.run(fn(*args, **kwargs))
                    # Running inside an event loop; fall back to HTTP path
                if wants_ctx:
                    active_ctx._trusted_writer = True
                    try:
                        return fn(active_ctx, *args, **kwargs)
                    finally:
                        active_ctx._trusted_writer = False
                return fn(*args, **kwargs)
            if len(args) == 1 and "prompt" not in kwargs:
                kwargs["prompt"] = args[0]
                args = ()
            # Forward server-execution timeout as meta to avoid colliding with user function params
            kwargs["__timeout"] = call_timeout
            return rpc(method, *args, base_url=base_url, timeout=call_timeout, **kwargs)
        return _call

tools = _ToolsProxy()
def tool(_fn: Callable | None = None, *, name: str | None = None):
    return declare_tool(_fn, name=name)

# Entrypoints: decorator and compatibility alias for sn1.boot
ENTRYPOINTS: dict[str, Callable[..., Any]] = {}
sys.modules.setdefault("sn1.boot", sys.modules[__name__])

def entrypoint(_fn: Callable | None = None, *, name: str | None = None):
    def _decorator(fn: Callable) -> Callable:
        # Mark the function so class-based agents can auto-register bound methods later
        try:
            setattr(fn, "__sn1_entrypoint__", True)
            setattr(fn, "__sn1_entry_name__", name or fn.__name__)
        except Exception:
            pass
        # Register entrypoints under a dedicated namespace to avoid colliding with tools
        ep_name = f"entry:{name or fn.__name__}"
        register(ep_name, fn)
        ENTRYPOINTS[name or fn.__name__] = fn
        return fn
    if _fn is None:
        return _decorator
    return _decorator(_fn)

# ---------------- Lifecycle: Agent base, and state store ----------------
AGENT_CLASSES: list[type] = []
AGENT_INSTANCES: list[Any] = []
_INIT_LOCK = threading.Lock()

class State(dict):
    pass
state = State()

class Agent:
    def __init_subclass__(cls, **kwargs):  # type: ignore[no-untyped-def]
        super().__init_subclass__(**kwargs)
        try: AGENT_CLASSES.append(cls)
        except Exception: pass
    def init(self, ctx: "Context") -> None: return None
    def shutdown(self) -> None: return None

def _agent_entry_methods(instance: Any):
    try:
        for attr_name, attr in getattr(instance.__class__, "__dict__", {}).items():
            if callable(attr) and getattr(attr, "__sn1_entrypoint__", False):
                ep_name = getattr(attr, "__sn1_entry_name__", attr_name)
                bound = getattr(instance, attr_name)
                yield str(ep_name), bound
    except Exception:
        return

# ---------------- Env loader ----------------
def load_env(env_or_path: str) -> Namespace:
    p = Path(env_or_path)
    if p.exists():
        base_dir = p if p.is_dir() else p.parent
        tools_file = base_dir / "tools.py"
        if not tools_file.exists():
            raise RuntimeError(f"Invalid environment path: {base_dir}. Expected tools.py")
        tools_module = _load_module_from_file(f"sn1_env_{uuid.uuid4().hex}_tools", tools_file)
    else:
        tools_module = importlib.import_module(f"environments.{env_or_path}.tools")

    if hasattr(tools_module, "register_tools"): tools_module.register_tools()
    docker_image = getattr(tools_module, "DOCKER_IMAGE", "python:3.11-slim")
    entrypoint = getattr(tools_module, "ENTRYPOINT", "solve")
    defaults = getattr(tools_module, "TOOL_DEFAULTS", None)
    return Namespace(
        docker_image=docker_image,
        entrypoint=entrypoint,
        defaults=defaults,
    )


# ---------------- RPC server (FastAPI) ----------------
_METHODS: dict[str, Any] = {}
_TOKEN_META: dict[str, dict[str, Any]] = {}
_GLOBAL_LIMIT = asyncio.Semaphore(200)
_CURRENT_CTX: ContextVar["Context | None"] = ContextVar("sn1_current_ctx", default=None)

@dataclass
class Context:
    token: str
    method: str
    request_id: str
    created_at: float
    headers: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    _trusted_writer: bool = False
    def get(self, key: str, default: Any = None) -> Any:
        try: return self.meta.get(key, default)
        except Exception: return default
    def set(self, key: str, value: Any) -> None:
        try: 
            if self._trusted_writer: self.meta[key] = value
        except Exception:pass

def register(name: str, fn: Any) -> None:
    _METHODS[name] = fn

def declare_tool(_fn: Callable | None = None, *, name: str | None = None):
    def _decorator(fn: Callable) -> Callable:
        register(name or fn.__name__, fn)
        return fn
    if _fn is None:
        return _decorator
    return _decorator(_fn)

def issue_token(ttl_s: int = 3600, per_token_limit: int = 16) -> str:
    tok = secrets.token_urlsafe(24)
    _TOKEN_META[tok] = {
        "expiry": time.time() + ttl_s,
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
async def healthz(): return {"ok": True}
@app.get("/methods")
async def methods(): return {"methods": sorted(_METHODS.keys())}
def _fn_wants_ctx(fn: Callable) -> bool:
    try:
        return any(
            (p.name in {"ctx", "context"}) or (
                p.annotation in (Context, "Context", "sn1.Context")
            )
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    except Exception:
        return False

@app.post("/rpc")
async def rpc_call(payload: RpcIn, req: Request, tok: str = Depends(_validate_token)):
    fn = _METHODS.get(payload.method)
    if not fn:
        raise HTTPException(status_code=404, detail=f"unknown method {payload.method}")
    meta = _TOKEN_META[tok]
    async with _GLOBAL_LIMIT, meta["sem"]:
        try:
            ctx = Context(
                token=tok,
                method=payload.method,
                request_id=uuid.uuid4().hex,
                created_at=time.time(),
                headers={k: v for k, v in req.headers.items()},
            )
            extra_ctx = payload.kwargs.pop("__ctx", None)
            if isinstance(extra_ctx, dict):
                try:
                    ctx.meta.update(extra_ctx)
                except Exception:
                    pass
            wants_ctx = _fn_wants_ctx(fn)
            # Extract optional per-call execution timeout (seconds)
            exec_timeout: Optional[float] = None
            try:
                t = payload.kwargs.pop("__timeout", None)
                if t is not None:
                    exec_timeout = float(t)
            except Exception:
                exec_timeout = None
            token = _CURRENT_CTX.set(ctx)
            try:
                async def _invoke():
                    if inspect.iscoroutinefunction(fn):
                        if wants_ctx:
                            ctx._trusted_writer = True
                            try:
                                return await fn(ctx, *payload.args, **payload.kwargs)
                            finally:
                                ctx._trusted_writer = False
                        return await fn(*payload.args, **payload.kwargs)
                    else:
                        if wants_ctx:
                            ctx._trusted_writer = True
                            try:
                                return await run_in_threadpool(fn, ctx, *payload.args, **payload.kwargs)
                            finally:
                                ctx._trusted_writer = False
                        return await run_in_threadpool(fn, *payload.args, **payload.kwargs)
                if exec_timeout and exec_timeout > 0:
                    res = await asyncio.wait_for(_invoke(), timeout=exec_timeout)
                else:
                    res = await _invoke()
            finally:
                _CURRENT_CTX.reset(token)
            return {"ok": True, "result": res, "ctx": dict(ctx.meta)}
        except Exception as e:
            # On timeout, asyncio raises TimeoutError from wait_for
            if isinstance(e, asyncio.TimeoutError):
                return {"ok": True, "result": None, "timeout": True, "ctx": dict(ctx.meta)}
            logger.error("rpc_error method=%s error=%s", payload.method, e)
            return {"ok": False, "error": str(e)}

def _instantiate_agents_and_register() -> dict[str, Any]:
    with _INIT_LOCK:
        if AGENT_INSTANCES:
            return {"initialized": True, "agents": len(AGENT_INSTANCES)}
        ctx = Context(
            token=os.getenv("SN1_TOKEN", ""),
            method="startup",
            request_id=uuid.uuid4().hex,
            created_at=time.time(),
            headers={},
        )
        for cls in list(AGENT_CLASSES):
            try:
                instance = cls()  # type: ignore[call-arg]
            except Exception as e:
                logger.error("failed to instantiate Agent %s: %s", getattr(cls, "__name__", str(cls)), e)
                continue
            AGENT_INSTANCES.append(instance)
            try:
                init_maybe = getattr(instance, "init", None)
                if callable(init_maybe):
                    if inspect.iscoroutinefunction(init_maybe):
                        # run init synchronously via loop
                        asyncio.get_event_loop().create_task(init_maybe(ctx))
                    else:
                        try:
                            init_maybe(ctx)
                        except TypeError:
                            init_maybe()
            except Exception as e:
                logger.error("Agent.init failed for %s: %s", getattr(instance.__class__, "__name__", "Agent"), e)
            for ep_name, bound in _agent_entry_methods(instance):
                register(f"entry:{ep_name}", bound)
                ENTRYPOINTS[ep_name] = bound
        return {"initialized": True, "agents": len(AGENT_INSTANCES)}

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

def _load_default_app_files() -> None:
    try:
        for candidate in ["/app/tools.py", "/app/agent.py"]:
            if os.path.exists(candidate):
                _load_module_from_file(f"sn1_autoload_{Path(candidate).stem}", Path(candidate))
                logger.info(f"sn1: loaded file {candidate}")
    except Exception as e:
        logger.warning(f"autoload failed: {e}")

def _bootstrap_container_token() -> None:
    fixed = os.getenv("SN1_TOKEN")
    if fixed and fixed not in _TOKEN_META:
        # Accept this token for all methods with a long expiry
        _TOKEN_META[fixed] = {
            "expiry": time.time() + 365 * 24 * 3600,
            "sem": asyncio.Semaphore(64),
        }
        logger.info("sn1: accepted container SN1_TOKEN for RPC")

@app.on_event("startup")
async def _on_startup():
    _bootstrap_container_token()
    _load_default_app_files()
    try:
        res = _instantiate_agents_and_register()
        logger.info(
            "sn1: startup complete; registered %d methods; init=%s",
            len(_METHODS),
            res,
        )
    except Exception as e:
        logger.error("sn1: initialization failed: %s", e)
        raise

@app.on_event("shutdown")
async def _on_shutdown():
    # Call Agent.shutdown hooks
    for instance in list(AGENT_INSTANCES):
        try:
            sd = getattr(instance, "shutdown", None)
            if callable(sd) and getattr(getattr(sd, "__func__", sd), "__qualname__", "").split(".")[0] != "Agent":
                if inspect.iscoroutinefunction(sd):
                    await sd()
                else:
                    await run_in_threadpool(sd)
        except Exception as e:
            logger.warning("shutdown handler failed for %s: %s", getattr(instance.__class__, "__name__", "Agent"), e)

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

def start_container(
    image: str,
    name: str,
    env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    copy_mode: bool = False,
) -> str:
    env = env or {}
    extra_args = extra_args or []
    try:
        _docker("pull", image)
    except Exception as e:
        logger.warning(f"docker pull failed (continuing): {e}")
    if copy_mode:
        args = [
            "run", "-d", "--entrypoint", "/bin/sh", "--name", name,
            "--add-host=host.docker.internal:host-gateway",
        ]
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        args += extra_args
        args += [image, "-c", "sleep infinity"]
        run = _docker(*args)
        container_id = run.stdout.strip() or name
        return container_id
    else:
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

def copy_into_container(container_id: str, src_path: str, dest_path: str):
    _docker("cp", src_path, f"{container_id}:{dest_path}")

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
        ctx: Any | None = None,
    ) -> None:
        if spec is not None:
            image = getattr(spec, "docker_image", image)
        self.image = image or "thebes1618/sn1:latest"
        self.local_script_path = os.path.abspath(agent) if agent else None
        self.in_container_script_path = f"/app/{os.path.basename(self.local_script_path)}" if self.local_script_path else None
        self.python_path = python_path
        self.container_name = f"sn1-{os.path.splitext(os.path.basename(self.local_script_path))[0]}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

        # Generate a shared token used by host<->container HTTP
        self.token = secrets.token_urlsafe(24)
        self.base_url = (base_url or "").rstrip("/")
        # Runner-visible context holder; use explicit get/set
        self._runner_ctx: dict[str, Any] = dict(ctx or {})
        self.ctx = Context(
            token=self.token,
            method="runner",
            request_id=uuid.uuid4().hex,
            created_at=time.time(),
            headers={},
            meta=dict(self._runner_ctx),
            _trusted_writer=True,
        )

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
                self.container_id = start_container(
                    self.image,
                    self.container_name,
                    env={"SN1_TOKEN": self.token},
                    extra_args=["-p", "0:5005"],
                    copy_mode=True,
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
                    "cd /app && /opt/venv/bin/python -m uvicorn sn1:app --host 0.0.0.0 --port 5005",
                )
            else:
                # Image-only mode: assume image starts uvicorn sn1:app
                self.container_id = start_container(
                    self.image,
                    self.container_name,
                    env={"SN1_TOKEN": self.token},
                    copy_mode=False,
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
            return dict(self.ctx.meta)
        except Exception:
            return {}

    def _call(self, entry: str, *args, **kwargs):
        self._ensure_active()
        prev = os.environ.get("SN1_TOKEN")
        os.environ["SN1_TOKEN"] = self.token
        try:
            method = f"entry:{entry}"
            # Include container-level ctx; allow per-call override via __ctx
            call_kwargs = dict(kwargs)
            # Extract timeout for both HTTP request and server-execution
            call_timeout = float(call_kwargs.pop("timeout", 60) or 60)
            call_ctx = self._ctx_payload()
            if "__ctx" in call_kwargs and isinstance(call_kwargs["__ctx"], dict):
                try:
                    merged = dict(call_ctx)
                    merged.update(call_kwargs["__ctx"])  # per-call takes precedence
                    call_ctx = merged
                except Exception:
                    pass
            call_kwargs["__ctx"] = call_ctx
            # Forward server-execution timeout without colliding with user params
            call_kwargs["__timeout"] = call_timeout
            try:
                data = rpc_raw(method, *args, base_url=self.base_url, timeout=call_timeout, **call_kwargs)
            except requests.exceptions.Timeout:
                return None
            if not isinstance(data, dict):
                return None
            # Update local ctx from server, if provided
            try:
                returned_ctx = data.get("ctx")
                if isinstance(returned_ctx, dict):
                    try:
                        self.ctx.meta.update(returned_ctx)
                    except Exception:
                        pass
            except Exception:
                pass
            if data.get("ok") is True:
                # Gracefully treat server-side timeouts as None
                if data.get("timeout") is True:
                    return None
                return data.get("result")
            return None
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

 