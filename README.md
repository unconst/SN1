# Gödelian Incentives

A template for open source agent incentives on a Bittensor subnet.

## Build Tools (tools.py)
```python
import sn1
@sn1.tool
async def multiply(x:float, y:float) -> float:
    return x * y
```

## Build Agent (agent.py)
```python
import sn1
@sn1.entrypoint
def multiply(x:float, y:float) -> float:
    return sn1.tools.multiply(x = x, y = y)
```

## Run an Agent in container (run.py)
```python
spec = load_env(os.path.dirname(__file__))
agent = os.path.join(os.path.dirname(__file__), "agent.py")
with Container(agent = agent, spec = spec) as s:
    print(s.multiply(x = x, y = y))

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
