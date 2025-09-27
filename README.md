# Gödelian Incentives

A template for open source agent incentives on a Bittensor subnet.

## Build Tools (tools.py)
```python
import sn1
@sn1.tool
async def multiply(ctx: sn1.Context, x:float, y:float) -> float:
    ctx.set("tool_calls", ctx.get("tool_calls", 0) + 1)
    return x * y
```

## Build Agent (agent.py)
```python
import sn1
class MyAgent(sn1.Agent):
    def init(self, ctx: sn1.Context):
        self.calls = 0
        print ('init agent')

    @sn1.entrypoint
    def multiply(self, ctx: sn1.Context, x: float, y: float) -> float:
        self.calls += 1
        print("challenge:", ctx.get("challenge_id"))
        print("tool calls so far:", ctx.get("tool_calls", 0))
        return sn1.tools.multiply(x=x, y=y)
```

### Run Agent
```python
import os
import sn1
spec = sn1.load_env(os.path.dirname(__file__))
agent = os.path.join(os.path.dirname(__file__), "agent.py")
with sn1.Container(agent = agent, spec = spec, ctx={'foo': 'bar', 'n_tool_calls': 0}) as s:
    print(s.multiply(x = 2, y = 5, timeout = 1))
    print("calls:", s.ctx.get("tool_calls", 0))
```

## Subnet example
```bash
# Copy .env and fill out validator items
cp .env.example .env
```
Run the validator with docker and watchtower autoupdate.
```bash
# Run the validator with watchtower.
docker-compose down && docker-compose pull && docker-compose up -d && docker-compose logs -f
```
