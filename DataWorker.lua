function dataWorker() 
	require 'torch'
	require 'cutorch'
	require 'nn'
	require 'cudnn'

	parallel.print('Process with ID ' .. parallel.id .. ' and ip ' .. parallel.ip .. ' is starting up.')
	local criterion = parallel.parent.receive()
	criterion:cuda()

	local modelInputs = torch.CudaTensor()
	local currModel = nil

	function outputWorker() 
		local info = parallel.parent:receive()
		local inputsCPU = info.inputs
		modelInputs:resize(inputsCPU:size()):copy(inputsCPU)
		currModel = info.model
		currModel:cuda()
		local labelsCPU = info.labels
		local labels = torch.CudaTensor():resize(labelsCPU:size()):copy(labelsCPU)

		local currOutputs = currModel:updateOutput(modelInputs), 
		local err = criterion:forward(currOutputs, labels)
		local gradOutputs = criterion:backward(currOutputs, labels)

		return {
			output = currOutputs, 
			err = err, 
			grad = gradOutputs
		}	
	end

	function gradWorker() 
		local grads = parallel.parent:receive()
		currModel:accGradParameters(modelInputs, grads)
		local _,grads = currModel:parameters()
		currModel = nil
		return grads
	end

	while true do 
		local yieldMessage = parallel.yield()
		if yieldMessage == "computeOutput" do 
			local outs = outputWorker()
			parallel.parent:send(outs)
		else if yieldMessage == "gradParameter" do 
			assert(currModel != nil)
			local grads = gradWorker()
			parallel.parent:send(grads)
		end
	end
end
