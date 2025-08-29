# Gödelian Incentives

A template for open source agent incentives on a Bittensor subnet.

## Build Agent
```python
import sn1
from sn1.boot import entrypoint

# Return something
@entrypoint()
def anything( z:str, y: int = 1 ) -> str:
    return y * z

# Call a function from tools
@entrypoint()
def in_func( x: int = 1 ):
    return sn1.tools.out_func(x=x)

# Query an llm through the tools.
@entrypoint()
def llm( prompt:str ):
    return sn1.tools.llm( prompt = prompt )
```

## Run an Agent
```python
from sn1 import Container
with Container("gen.py") as s:
    print(s.anything( z = 'cat', y = 2))   # -> catcat
    print(s.in_func( 2 ))   # -> 5  (direct function inside container)
    print(s.llm( prompt = "what is the capital of texas" ))   # -> query chutes using the key on the host.
```

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
