from sn1 import Container
with Container("gen.py") as s:
    print(s.llm(prompt = "what is the capital of texas?" ))   # ->  query chutes using the key on the host.
