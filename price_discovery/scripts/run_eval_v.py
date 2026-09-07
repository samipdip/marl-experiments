import os
import sys

for v in [1001, 1002, 1003, 1004, 1005]:
    print(f"===== eval v={v} =====")
    for ep in range(100):
        seed = 10000 + ep
        d = f"/home/samip/price_discovery/results/ppo_hjb_v{v}/ep_{ep}"
        os.makedirs(d, exist_ok=True)
        cfg = f"/tmp/ppo_hjb_eval_v{v}.yaml"
        script = (
            "import sys,os;"
            'sys.path.insert(0,os.path.expanduser("~/SimExchangeSandbox"));'
            'sys.path.insert(0,os.path.expanduser("~/Trading_agents"));'
            "from phase3_multi_agent.src.runner import run_simulation;"
            f'run_simulation("{cfg}",device="cpu",seed={seed},results_dir="{d}")'
        )
        r = os.system(f"cd /home/samip/marl-experiments && .venv/bin/python -c '{script}'")
        if r != 0:
            print(f"  FAILED ep {ep} v={v}")
            sys.exit(1)
    print(f"===== v={v} done =====")
