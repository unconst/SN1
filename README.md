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

### Run Agent
```python
import os
import sn1
spec = sn1.load_env(os.path.dirname(__file__))
agent = os.path.join(os.path.dirname(__file__), "agent.py")
with sn1.Container(agent=agent, spec=spec) as c:
    print(c.multiply(x=2, y=5))
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
