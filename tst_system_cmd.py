# coding: utf-8

import os
os.system('pwd')
os.system('cd ~')
os.system('ls -la')

import os
stream = os.popen('dir')
output = stream.readlines()
print( output )

import subprocess
x = subprocess.run(['dir'], shell=True)

print(x)
print(x.args)
print(x.returncode)
print(x.stdout)
print(x.stderr)

# SAVING THE COMMAND OUTPUT TO A TEXT FILE
import subprocess

with open('tmp_list.txt', 'w', encoding='UTF8') as f:
    subprocess.run(['dir' ], stdout=f)
pass