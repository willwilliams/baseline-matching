require 'parallel'
require 'lab'

function worker(workerFunc)
	require 'sys'
	require 'torch'

	parallel.print('Worker with id ' .. parallel.id .. 'starting up with IP' .. parallel.ip)

end