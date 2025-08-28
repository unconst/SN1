import sn1
from sn1.boot import entrypoint

@entrypoint()   # optional; without it, raw def name also works
def in_func( x: int = 1 ):
    # call host tool; host returns x+1
    return sn1.tools.out_func(x=x)

@entrypoint()   # optional; without it, raw def name also works
def llm( prompt:str ):
    # call host tool; host returns x+1
    return sn1.tools.llm( prompt = prompt )

@entrypoint()   # optional; without it, raw def name also works
def search( search:str ):
    return sn1.tools.search( search = search )
