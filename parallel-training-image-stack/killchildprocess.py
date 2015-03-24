import pexpect
import os
import time
import sys

# ssh into other machines so we can get past unknown RSA fingerprint stuff
machinesFile = open("../parallel-training-image-stack/machines.txt", "r")
for line in machinesFile.readlines():
	child = pexpect.spawn("ssh -i /neural_networks/cluster-key " + line)
	print("sshed into machine " + line)
	child.expect('ubuntu.*')
        child.sendline("killall luajit")
	print("killed luajit processes")
	child.expect('ubuntu..*')
  	child.sendline("killall sshd")
	print("killed sshd")
