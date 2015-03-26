function dataWorker() 
	--parallel.print('Process with ID ' .. parallel.id .. ' and ip ' .. parallel.ip .. ' is starting up.')
	local fileLoadFunc = parallel.parent:receive()
 	fileLoadFunc()
	parallel.parent:send("message received")
	local utilFunc = parallel.parent:receive()
  	utilFunc()
	parallel.parent:send("message received")
	require 'cutorch'
	require 'xlua'
	require 'nn'
	require 'cunn'
	--require 'cudnn'
  local fileLoader = parallel.parent:receive()
  parallel.parent:send("message received")
  local Threads = require 'threads'
  do -- start K datathreads (donkeys)
    donkeys = Threads(
      3,
      function()
        require 'torch'
      end,
      function(idx)
        prepFunc = fileLoadFunc 
        prepFunc()
	doUtils = utilFunc
	doUtils()
      end,
      function(idx)
	local seed = 2 + idx
	trainLoader = fileLoader
      end
    );
  end
  local batchSize = parallel.parent:receive()
  parallel.parent:send("message received")
  local currBatch = 0

  local criterion = nn.ClassNLLCriterion()	
  criterion:cuda()

	local modelInputs = torch.CudaTensor()
	local labels = torch.CudaTensor()

	function outputGradsWorker(currModel, inputsCPU, labelsCPU) 
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
			local _,prediction_sorted = currOutputs:float():sort(2, true) -- descending
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
	
	local receivedInputsCPU = torch.FloatTensor()
	local receivedLabelsCPU = torch.LongTensor()
	
	function trainCallBack(inputsThread, labelsThread)
		currBatch = currBatch + 1
		local model = parallel.parent:receive()
		model:cuda()
		receiveTensor(inputsThread, receivedInputsCPU)
  		receiveTensor(labelsThread, receivedLabelsCPU)
		local outputGrads = outputGradsWorker(model, receivedInputsCPU, receivedLabelsCPU)
		parallel.parent:send(outputGrads)
		collectgarbage()
	end
	
	local loadcount = 1

	while true do
		loadcount = loadcount + 1
		donkeys:addjob(
      	 	-- the job callback (runs in data-worker thread)
  	  	function()
     			local inputs, labels = trainLoader:sample(batchSize)
      			return sendTensor(inputs), sendTensor(labels)
   	  	end,
   	  	-- the end callback (runs in the main thread)
   	  	trainCallBack
		)
		if loadcount % 8 == 0 then
   			loadcount = 0
			donkeys:synchronize()
		end	
	end
end
