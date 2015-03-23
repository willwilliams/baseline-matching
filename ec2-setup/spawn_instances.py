import boto.ec2
import sys
import copy
import time

# args: args[1] is price, args[2] is ami, args[3] is number of instances
args = sys.argv

maxPrice = args[1]
imageID = args[2] 
numInstances = int(args[3])

# create spot requests
ec2 = boto.ec2.connect_to_region('us-east-1')
request = ec2.request_spot_instances(
	price=maxPrice,
	image_id=imageID, 
	count=numInstances,
	key_name='cluster-key', 
	instance_type='g2.2xlarge',
	placement='us-east-1e',
	placement_group='parallel_cluster',
	ebs_optimized=True
)

def wait_for_fulfillment(conn, request_ids, pending_request_ids):
    # Loop through all pending request ids waiting for them to be fulfilled.
    # If a request is fulfilled, remove it from pending_request_ids.
    # If there are still pending requests, sleep and check again in 10 seconds.
    # Only return when all spot requests have been fulfilled.
	instance_ids = []
	while len(pending_request_ids) > 0:
		results = conn.get_all_spot_instance_requests(request_ids=pending_request_ids)
		for result in results:
			if result.status.code == 'fulfilled':
				pending_request_ids.pop(pending_request_ids.index(result.id))
				print "spot request `{}` fulfilled!".format(result.id)
				instance_ids.append(result.instance_id)
			else:
				print "waiting on `{}`".format(result.id)
		time.sleep(60)
	print('all spot requests fulfilled!')
	return instance_ids

# get ids of spot requests
request_ids = [req.id for req in request]

# wait for spot requests to fulfill
instance_ids = wait_for_fulfillment(ec2, request_ids, copy.deepcopy(request_ids))

# get DNS names and print
instance_list = ec2.get_only_instances(instance_ids=instance_ids)
for instance in instance_list:
	print('ubuntu@' + instance.public_dns_name)

