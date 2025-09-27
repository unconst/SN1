import sn1

class MyAgent(sn1.Agent):
    def init(self, ctx: sn1.Context):
        self.calls = 0
        print ('init agent')

    @sn1.entrypoint
    def multiply(self, ctx: sn1.Context, x: float, y: float) -> float:
        self.calls += 1
        print("challenge:", ctx.get("challenge_id"))
        print("tool calls so far:", ctx.get("tool_calls", 0))
        return sn1.tools.multiply(x=x, y=y)