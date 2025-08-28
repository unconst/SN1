# sn1/docker.py
from __future__ import annotations
import os, sn1, shutil, subprocess, time, uuid, shlex, json, requests, asyncio
from typing import Optional

# ---------------- Docker helpers ----------------
def _get_docker_bin() -> str:
    docker_path = os.getenv("DOCKER_BIN") or "/usr/bin/docker"
    if os.path.exists(docker_path):
        return docker_path
    fallback = shutil.which("docker")
    if fallback:
        return fallback
    raise RuntimeError("docker binary not found. Ensure docker is installed and mounted.")

def _run(cmd: list[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    sn1.logger.debug(f"Running command: {' '.join(cmd)}")
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
        sn1.logger.warning(f"docker pull failed (continuing): {e}")

    args = [
        "create",
        "--entrypoint", "/bin/sh",
        "--name", name,
        "--add-host=host.docker.internal:host-gateway",  # Linux-friendly host mapping
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

def stop_and_remove_container(container_id: str):
    try:
        _docker("stop", container_id, check=False)
    finally:
        _docker("rm", container_id, check=False)

# ---------------- Container API ----------------
class Container:
    """
    Runs an arbitrary script inside a container and dispatches functions
    via the sn1/boot.py bootstrapper. Inside the container, scripts can call
    host tools via Container.rpc/llm/out_func, authenticated by SN1_TOKEN.
    """
    def __init__(
        self,
        path_to_script: str,
        image: str = "thebes1618/sn1:latest",
        *,
        python_path: str = "/opt/venv/bin/python",
        base_url: Optional[str] = None,
        token_ttl: int = 3600
    ) -> None:
        self.image = image
        self.local_script_path = os.path.abspath(path_to_script)
        self.in_container_script_path = f"/app/{os.path.basename(self.local_script_path)}"
        self.python_path = python_path
        self.container_name = f"sn1-{os.path.splitext(os.path.basename(self.local_script_path))[0]}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

        # per-container token and base URL
        from sn1.server import issue_token, ensure_server_running  # local import to avoid cycles
        self.token = issue_token(ttl_s=token_ttl)
        # For processes running inside the container, 127.0.0.1 is the container itself.
        # Default to host.docker.internal so the container can reach the host services.
        self.base_url = (base_url or "http://host.docker.internal:5005").rstrip("/")

        # Start local server if we're pointing at the host
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            host = (parsed.hostname or "").lower()
            port = parsed.port or 5005
            if host in {"host.docker.internal", "localhost", "127.0.0.1"}:
                ensure_server_running(host="0.0.0.0", port=port)
        except Exception as _e:
            sn1.logger.warning(f"ensure_server_running failed: {_e}")

        sn1.logger.info(f"Preparing container {self.container_name} from {self.image}")
        self.container_id = create_running_container(
            self.image, self.container_name,
            env={
                "RUNNER_BASE_URL": self.base_url,
                "SN1_TOKEN": self.token,
            }
        )

        # Copy user script and our package/bootstrapper into the container
        copy_into_container(self.container_id, self.local_script_path, self.in_container_script_path)
        # Copy sn1 package into /app so `import sn1` resolves to our current code
        pkg_src = os.path.dirname(__file__)
        copy_into_container(self.container_id, pkg_src, "/app/sn1")
        # Also copy boot.py explicitly to a well-known path
        boot_src = os.path.join(pkg_src, "boot.py")
        copy_into_container(self.container_id, boot_src, "/app/boot.py")

        self._destroyed = False

    def _ensure_active(self) -> None:
        if self._destroyed:
            raise RuntimeError("Container has been destroyed")

    def _call(self, entry: str, *args, **kwargs):
        """
        Call a function inside the container using the bootstrapper.
        """
        self._ensure_active()
        payload = {"args": list(args), "kwargs": kwargs}
        payload_json = json.dumps(payload, separators=(",", ":"))
        cmd_parts = [
            shlex.quote(self.python_path),
            "/app/boot.py",
            "--script", shlex.quote(self.in_container_script_path),
            "--entry", shlex.quote(entry),
            "--payload", shlex.quote(payload_json),
        ]
        cmd = " ".join(cmd_parts)

        rc, out, err = exec_in_container(self.container_id, cmd)
        if err:
            for line in err.splitlines():
                sn1.logger.warning(f"[script][stderr] {line}")

        parsed = None
        if out and out.strip():
            try:
                parsed = json.loads(out.strip())
            except json.JSONDecodeError:
                parsed = None

        if isinstance(parsed, dict) and parsed.get("ok") is False:
            message = parsed.get("error", "remote error")
            err_type = parsed.get("type")
            if err_type:
                message = f"{err_type}: {message}"
            raise RuntimeError(message)

        if rc != 0:
            tail = (err or "").strip().splitlines()[-5:]
            snippet = ("\n".join(tail)).strip()
            if snippet:
                raise RuntimeError(f"script exited with code {rc}:\n{snippet}")
            raise RuntimeError(f"script exited with code {rc}")

        if isinstance(parsed, dict) and "result" in parsed:
            return parsed["result"]
        if parsed is not None:
            return parsed
        return out

    def __getattr__(self, name: str):
        def _caller(*args, **kwargs):
            return self._call(name, *args, **kwargs)
        return _caller

    def entries(self) -> list[str]:
        # best effort: try calling __list__ if provided by the script
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


