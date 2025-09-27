import sn1
@sn1.tool
async def multiply(ctx: sn1.Context, x:float, y:float) -> float:
    ctx.n_tool_calls += 1
    return x * y

