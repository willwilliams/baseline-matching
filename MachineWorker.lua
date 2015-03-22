require 'parallel'
require 'lab'

function worker(workerFunc)
	require 'sys'
	require 'torch'

	parallel.print('Worker with id ' .. parallel.id .. 'starting up with IP' .. parallel.ip)

end

function machineSend(destTensor, sourceTensor)

function sendToWorker(data, childID) 
	parallel.children[childID]:send(data)
end

function sendToParent(data)
	parallel.parent:send(data)
end