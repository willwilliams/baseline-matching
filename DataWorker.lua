function dataWorker() 
	require 'torch'
	require 'cutorch'
	require 'nn'
	require 'cudnn'

	parallel.print('Process with ID ' .. parallel.id .. ' and ip ' .. parallel.ip .. ' is starting up.')
	
	function outputWorker() 
		while true do
			local info = parallel.parent:receive()
			if info do	
				local inputs = info.inputs
				local model = info.model
				return module:updateOutput(inputs)
			end
		end
	end

	function gradWorker() 
		while true do 
			local 
		end
	end

	while true do 
		local yieldMessage = parallel.yield()
		if yieldMessage == "computeOutput" do 
			local outs = outputWorker()
			parallel.parent:send(outs)
		else do
			gradworker()
		end
	end

end
