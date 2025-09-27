import sn1
@sn1.entrypoint
def multiply(x:float, y:float) -> float:
    return sn1.tools.multiply(x = x, y = y)