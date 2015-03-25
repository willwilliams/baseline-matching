function dataWorker() 
	--parallel.print('Process with ID ' .. parallel.id .. ' and ip ' .. parallel.ip .. ' is starting up.')
	require 'torch'
	require 'cutorch'
	require 'xlua'
	require 'nn'
	require 'cunn'
	--require 'cudnn'

    	local criterion = nn.ClassNLLCriterion()	
    	criterion:cuda()

	local modelInputs = torch.CudaTensor()

	function outputGradsWorker() 
		local info = parallel.parent:receive()
		parallel.parent:send("data received")
		local inputsCPU = info.inputs
		modelInputs:resize(inputsCPU:size()):copy(inputsCPU)
		local currModel = info.model
		currModel:cuda()
		local labelsCPU = info.labels
		local labels = torch.CudaTensor():resize(labelsCPU:size()):copy(labelsCPU)

		local currOutputs = currModel:forward(modelInputs)
		local err = criterion:forward(currOutputs, labels)
		local gradOutputs = criterion:backward(currOutputs, labels)
		currModel:backward(modelInputs, gradOutputs)
		local _,gradParams = currModel:parameters()

		return {
			output = currOutputs, 
			err = err, 
			gradParam = gradParams
		}	
	end

	while true do 
		parallel.yield()
		local outputGrads = outputGradsWorker()
		parallel.parent:send(outputGrads)
		collectgarbage()
	end
end
