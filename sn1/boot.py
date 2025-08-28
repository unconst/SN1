# sn1/boot.py
import sys, os, json, argparse, importlib.util, types
from typing import Any, Callable, Dict, Optional

ENTRYPOINTS: Dict[str, Callable[..., Any]] = {}

# Make this module importable as 'sn1.boot' so user scripts can do
# 'from sn1.boot import entrypoint' inside the container.
sys.modules.setdefault("sn1.boot", sys.modules[__name__])

def entrypoint(name: Optional[str] = None):
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        ENTRYPOINTS[name or fn.__name__] = fn
        return fn
    return _decorator

def _load_module_from_path(path: str) -> types.ModuleType:
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("sn1_user_module", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sn1_user_module"] = mod
    spec.loader.exec_module(mod)  # executes user code (which may use @entrypoint)
    return mod

def _json_default(o: Any):
    return repr(o)

def main() -> int:
    parser = argparse.ArgumentParser(description="SN1 bootstrapper")
    parser.add_argument("--script", required=True, help="Path to user script")
    parser.add_argument("--entry", required=True, help="Function name to call")
    parser.add_argument("--payload", default="{}", help='JSON {"args": [], "kwargs": {}}')
    ns = parser.parse_args()

    try:
        payload = json.loads(ns.payload or "{}")
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ValueError("payload must contain list 'args' and dict 'kwargs'")

        _load_module_from_path(ns.script)

        target = ENTRYPOINTS.get(ns.entry)
        if target is None:
            # fallback: call raw attribute on module if not decorated
            mod = sys.modules["sn1_user_module"]
            target = getattr(mod, ns.entry, None)
        if target is None:
            raise ValueError(f"unknown entry '{ns.entry}'")

        result = target(*args, **kwargs)
        if hasattr(result, "__await__"):
            # allow async defs too
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(result)
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, default=_json_default))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e), "type": e.__class__.__name__}, ensure_ascii=False))
        return 1

if __name__ == "__main__":
    sys.exit(main())
