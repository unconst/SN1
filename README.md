# Gödelian Incentives

A template for open source agent incentives on a Bittensor subnet.

# Features
- Agents which pull/push from public Github Gists
- Run agents in secure sandboxes (docker containers)
- Provide agents with env-local tools via RPC (`sn1.tools.<name>` or `sn1.tool("name")`) enforced by per-token allowlists
- Docker orchestration with watchtower allows you to make direct commitments to running validators

## Build Agent
```python
import sn1
from sn1.boot import entrypoint

# Return something
@entrypoint()
def anything( z:str, y: int = 1 ) -> str:
    return y * z

# Query an llm through the tools (two equivalent styles)
@entrypoint()
def llm(prompt: str):
    return sn1.tools.llm(prompt=prompt)
    # or: return sn1.tool("llm", prompt=prompt)
```

## Run an Agent
```python
from sn1 import Container
with Container("gen.py") as s:
    print(s.anything( z = 'cat', y = 2))   # -> catcat
    print(s.llm(prompt="what is the capital of texas"))   # -> queries host via allowlisted tool

```

## Environments and Tools

All tools are defined per-environment in `environments/<env>/tools.py`. The host issues a token with an allowlist derived from that file and enforces it at `/rpc`.

Run an environment against an agent:

```bash
sn1 env run math_mul --agent gen.py --samples 20
```

### Create a minimal environment (simple guide)

1) Create a folder and `tools.py`:

```bash
mkdir -p environments/my_env
```

```python
# environments/my_env/tools.py
from typing import Any, Dict, Set
from sn1 import register

async def echo(*, prompt: str) -> str:
    return prompt

ALLOWED_METHODS: Set[str] = {"echo"}
TOOL_DEFAULTS: Dict[str, Any] = {"echo": {}}

def register_tools() -> None:
    register("echo", echo)
```

2) Add a tiny agent using the tool:

```python
# environments/my_env/agent.py
import sn1
from sn1.boot import entrypoint

@entrypoint()
def solve(prompt: str):
    return sn1.tools.echo(prompt=prompt)
```

3) Run it:

```bash
sn1 env run environments/my_env --agent environments/my_env/agent.py --entry solve --prompt "hello"
```

Notes:
- `ALLOWED_METHODS` restricts what the agent can call via RPC.
- `TOOL_DEFAULTS` provides per-tool defaults (optional).
- You can add more async tools in `tools.py` and register each with `register("name", fn)`.

## (TODO) Validating
```bash
# Copy .env and fill out validator items
cp .env.example .env
```
Run the validator with docker and watchtower autoupdate.
```bash
# Run the validator with watchtower.
docker-compose down && docker-compose pull && docker-compose up -d && docker-compose logs -f
```
