import boto.ec2
import sys

# args: args[1] is price, args[2] is ami, args[3] is number of instances
args = sys.argv

maxPrice = args[1]
imageID = args[2] 
numInstances = int(args[3])

ec2 = boto.ec2.connect_to_region('us-east-1')
request = ec2.request_spot_instances(
	price=maxPrice,
	image_id=imageID, 
	count=numInstances,
	key_name='cluster-key', 
	instance_type='g2.2xlarge',
	placement_group='parallel_cluster',
	ebs_optimized=True
)
