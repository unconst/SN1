import os
import sn1
sn1.set_log_level("DEBUG")

def main():
    spec = sn1.load_env(os.path.dirname(__file__))
    agent = os.path.join(os.path.dirname(__file__), "agent.py")
    with sn1.Container(agent = agent, spec = spec, ctx={'my_thing': 'cat'}) as s:
        print(s.multiply(x = 2, y = 5))

if __name__ == "__main__":
    main()


