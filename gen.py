import sn1
from sn1.boot import entrypoint

@entrypoint()
def anything( z:str, y: int = 1 ) -> str:
    return y * z

@entrypoint()
def in_func( x: int = 1 ):
    return sn1.tools.out_func(x=x)

@entrypoint()
def llm( prompt:str ):
    return sn1.tools.llm( prompt = prompt )
