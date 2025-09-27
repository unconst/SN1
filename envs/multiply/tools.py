import sn1
@sn1.tool
async def multiply(ctx: sn1.Context, x:float, y:float) -> float:
    ctx.set("tool_calls", ctx.get("tool_calls", 0) + 1)
    return x * y

