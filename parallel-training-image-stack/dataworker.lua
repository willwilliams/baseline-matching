function dataWorker() 
	parallel.print('Process with ID ' .. parallel.id .. ' and ip ' .. parallel.ip .. ' is starting up.')
	require 'torch'
	require 'cutorch'
	require 'xlua'
	require 'nn'
	require 'cunn'
	require 'cudnn'

        local criterion = nn.ClassNLLCriterion()	
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

		local currOutputs = currModel:updateOutput(modelInputs)
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
		--local grads = torch.CudaTensor():resize(gradsSent:size()):copy(gradsSent)
                currModel:backward(modelInputs, grads)
		local _,gradParams = currModel:parameters()
		currModel = nil
		return gradParams
	end

	while true do 
		local yieldMessage = parallel.yield()
		if yieldMessage == "computeOutput" then
			parallel.print("Computing outputs.")
			local outs = outputWorker()
			parallel.parent:send(outs)
		else if yieldMessage == "gradParameter" then 
			assert(currModel ~= nil)
			parallel.print("Calculating parameter gradients.")
			local grads = gradWorker()
			parallel.parent:send(grads)
		     end
		end
	end
end
