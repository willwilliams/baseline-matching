import pexpect
import os
import time

# ssh into other machines so we can get past unknown RSA fingerprint stuff
machinesFile = fopen("machines.txt", "r")
for line in machinesFile.readlines():
	child = pexpect.spawn("ssh -i /neural_networks/cluster-key " + line)
	child.expect(".*RSA.*")
	child.sendline("yes")
	time.sleep(10)
	os.system("exit")
	
