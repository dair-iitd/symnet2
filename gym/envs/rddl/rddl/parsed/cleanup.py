import os
files = os.listdir("/scratch/cse/dual/cs5180404/uai2022/deep-rl-transfer7/rddl/parsed")

postfix = [f'__{i}' for i in range(1, 11)]
postfix += [f'__{i}' for i in range(5000, 5004)]
# postfix += [f'_mdp.dot']
for file in files:
    found = False
    for p in postfix:
        if file.endswith(p):
            found = True
            break
    if not found:
        if file != "cleanup.py":
            os.remove(file)