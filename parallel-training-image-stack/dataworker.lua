function dataWorker() 
	--parallel.print('Process with ID ' .. parallel.id .. ' and ip ' .. parallel.ip .. ' is starting up.')
	require 'torch'
	require 'cutorch'
	require 'xlua'
	require 'nn'
	require 'cunn'
	--require 'cudnn'

	local currBatch = 0
    local criterion = nn.ClassNLLCriterion()	
    criterion:cuda()

	local modelInputs = torch.CudaTensor()

	function outputGradsWorker() 
		local info = parallel.parent:receive()
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

		local acc = -1
		local top1 = 0
		if currBatch % 15 == 0 then
			local _,prediction_sorted = output:float():sort(2, true) -- descending
          	local gt = labelsCPU
          	local batchSize = labelsCPU:size()[1]
          	for i=1,batchSize do
            	local pi = prediction_sorted[i]
            	if pi[1] == gt[i] then top1 = top1 + 1 end
          	end
          	acc = top1*100/batchSize
        end

		return {
			acc = acc,
			err = err, 
			gradParam = gradParams
		}	
	end

	while true do 
		currBatch = currBatch + 1
		local outputGrads = outputGradsWorker()
		parallel.parent:send(outputGrads)
		collectgarbage()
	end
end
