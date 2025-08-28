from sn1 import Container
with Container("gen.py") as s:
    print(s.in_func( x = 41))   # -> 42 (via host out_func)
    print(s.raw_add(2, 3))   # -> 5  (direct function inside container)
    print(s.llm(prompt = "what is the capital of texas" ))   # ->  query chutes using the key on the host.
    print(s.search(query = "where is japan?" ))   # ->  query chutes using the key on the host.
