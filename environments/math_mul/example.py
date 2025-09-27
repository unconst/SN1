import os
import sys

# Ensure repo root is on sys.path so 'import sn1' works when run directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sn1 import load_env, Container


def main():
    spec = load_env(os.path.dirname(__file__))
    agent_path = os.path.join(os.path.dirname(__file__), "agent.py")
    prompt = "what is 0.2 * 0.4?, return you answer like <Answer>12.232</Answer>"
    with Container(agent_path, spec=spec) as c:
        print(getattr(c, spec.entrypoint)(observation={"prompt": prompt}))


if __name__ == "__main__":
    main()


