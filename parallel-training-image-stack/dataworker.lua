function dataWorker() 
	--parallel.print('Process with ID ' .. parallel.id .. ' and ip ' .. parallel.ip .. ' is starting up.')
	require 'torch'
	require 'cutorch'
	require 'xlua'
	require 'nn'
	require 'cunn'
	--require 'cudnn'

	local trainLoader = parallel.parent.receive()
    local batchSize = parallel.parent.receive()
    
    local currBatch = 0

    local criterion = nn.ClassNLLCriterion()	
    criterion:cuda()

	local modelInputs = torch.CudaTensor()
	local labels = torch.CudaTensor()

	function outputGradsWorker(currModel) 
		local inputsCPU, labelsCPU = trainLoader:sample(batchSize)
		modelInputs:resize(inputsCPU:size()):copy(inputsCPU)
		labels:resize(labelsCPU:size()):copy(labelsCPU)

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
		local model = parallel.receive()
		model:cuda()
		local outputGrads = outputGradsWorker(model)
		parallel.parent:send(outputGrads)
		collectgarbage()
	end
end
