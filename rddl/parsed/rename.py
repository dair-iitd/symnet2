import os
import re

files = [x for x in os.listdir(".") if not x.endswith(".py")]
postfix = ["5000", "5001", "5002", "5003"]
correct = {"5000": "11", "5001": "12", "5002": "13", "5003": "14"}

for filename in files:
    found = False
    for p in postfix:
        if filename.endswith(p):
            found = True
            break
    if found:
        match = re.match('((\w)+)_inst_mdp__((\d)+)', filename)
        domain = match.group(1)
        inst = match.group(3)
        with open(filename, 'r') as file :
            filedata = file.read()

        orig = domain+"_inst_mdp__"+inst
        new = domain+"_inst_mdp__"+correct[inst]
        print(orig, new)
        filedata = filedata.replace(orig, new)

        # Write the file out again
        with open(filename, 'w') as file:
            file.write(filedata)
    
        os.rename(filename, filename.replace(inst, correct[inst]))
        # break
        
                