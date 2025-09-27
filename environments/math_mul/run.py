import os
import sys
from sn1 import load_env, Container

def main():
    spec = load_env(os.path.dirname(__file__))
    agent = os.path.join(os.path.dirname(__file__), "agent.py")
    prompt = "what is 0.2 * 0.4?, return you answer like <Answer>12.232</Answer>"
    with Container(agent, spec=spec) as c:
        print( c.solve(prompt=prompt) )

if __name__ == "__main__":
    main()


