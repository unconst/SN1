import sn1
@sn1.tool
async def multiply(ctx: sn1.Context, x:float, y:float) -> float:
    ctx.n_tools += 1
    return x * y

