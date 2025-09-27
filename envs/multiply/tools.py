import sn1
@sn1.tool
async def multiply(ctx: sn1.Context, x:float, y:float) -> float:
    # Access caller context if needed, e.g., ctx.token
    return x * y

