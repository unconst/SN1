import sn1

class MyAgent(sn1.Agent):
    def init(self, ctx: sn1.Context):
        self.calls = 0
        print ('init agent')

    @sn1.entrypoint
    def multiply(self, x: float, y: float) -> float:
        self.calls += 1
        return sn1.tools.multiply(x=x, y=y)