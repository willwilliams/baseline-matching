function dataWorker() 
	require 'torch'
	require 'cutorch'
	require 'nn'
	require 'cudnn'

	parallel.print('Process with ID ' .. parallel.id .. ' and ip ' .. parallel.ip .. ' is starting up.')
	
	local modelInputs = torch.CudaTensor()
	local currModel = nil

	function outputWorker() 
		local info = parallel.parent:receive()
		if info do	
			local inputsCPU = info.inputs
			modelInputs:resize(inputsCPU:size()):copy(inputsCPU)
			currModel = info.model
			return currModel:updateOutput(modelInputs)
		end
	end

	function gradWorker() 
		local grads = parallel.parent:receive()
		if grads do
			currModel:accGradParameters(modelInputs, grads)
			local _,grads = currModel:parameters()
			currModel = nil
			return grads
		end
	end

	while true do 
		local yieldMessage = parallel.yield()
		if yieldMessage == "computeOutput" do 
			local outs = outputWorker()
			parallel.parent:send(outs)
		else if yieldMessage == "gradParameter" do 
			assert(currModel != nil)
			gradWorker()
			parallel.parent:send(grads)
		end
	end

end
