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

## Run the agent as an HTTP microservice

Container images now run a FastAPI app and expose RPC over HTTP.

### Build and run
```bash
# From the repository root
docker build -t sn1-multiply -f envs/multiply/Dockerfile .
docker run -p 5005:5005 sn1-multiply
```

### Call from Python
```python
from sn1 import Container, load_env
import os

spec = load_env(os.path.dirname(__file__))
agent = os.path.join(os.path.dirname(__file__), "agent.py")

with Container(agent=agent, spec=spec) as c:
    print(c.multiply(x=2, y=5))
```

You can also connect to an already running service by providing a base URL:
```python
from sn1 import Container
with Container(agent="sn1-multiply", base_url="http://127.0.0.1:5005") as c:
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
