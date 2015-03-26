function dataWorker() 
	--parallel.print('Process with ID ' .. parallel.id .. ' and ip ' .. parallel.ip .. ' is starting up.')
	local fileLoadFunc = parallel.parent:receive()
  fileLoadFunc()

	require 'cutorch'
	require 'xlua'
	require 'nn'
	require 'cunn'
	--require 'cudnn'
  local fileLoader = parallel.parent:receive()

  local Threads = require 'threads'
  do -- start K datathreads (donkeys)
    donkeys = Threads(
      2,
      function()
        require 'torch'
      end,
      function(idx)
        prepFunc = fileLoadFunc 
        local seed = 2 + idx
        torch.manualSeed(seed)
        prepFunc()
        trainLoader = fileLoader
      end
    );
  end

  -- LOADING utils.lua 
  ffi.cdef[[
  void THFloatStorage_free(THFloatStorage *self);
  void THLongStorage_free(THLongStorage *self);
  ]]

  function setFloatStorage(tensor, storage_p)
     assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
     local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
     if cstorage ~= nil then
        ffi.C['THFloatStorage_free'](cstorage)
     end
     local storage = ffi.cast('THFloatStorage*', storage_p)
     tensor:cdata().storage = storage
  end

  function setLongStorage(tensor, storage_p)
     assert(storage_p and storage_p ~= 0, "LongStorage is NULL pointer");
     local cstorage = ffi.cast('THLongStorage*', torch.pointer(tensor:storage()))
     if cstorage ~= nil then
        ffi.C['THLongStorage_free'](cstorage)
     end
     local storage = ffi.cast('THLongStorage*', storage_p)
     tensor:cdata().storage = storage
  end

  function sendTensor(inputs)
     local size = inputs:size()
     local ttype = inputs:type()
     local i_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
     inputs:cdata().storage = nil
     return {i_stg, size, ttype}
  end

  function receiveTensor(obj, buffer)
     local pointer = obj[1]
     local size = obj[2]
     local ttype = obj[3]
     if buffer then
        buffer:resize(size)
        assert(buffer:type() == ttype, 'Buffer is wrong type')
     else
        buffer = torch[ttype].new():resize(size)      
     end
     if ttype == 'torch.FloatTensor' then
        setFloatStorage(buffer, pointer)
     elseif ttype == 'torch.LongTensor' then
        setLongStorage(buffer, pointer)
     else
        error('Unknown type')
     end
     return buffer
  end
  --------------------------------------------------------------------------------------

  local batchSize = parallel.parent:receive()
    
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

	while true do
		donkeys:addjob(
      	 -- the job callback (runs in data-worker thread)
  	  function()
     		local inputs, labels = trainLoader:sample(opt.batchSize)
      		return sendTensor(inputs), sendTensor(labels)
   	  end,
   	  -- the end callback (runs in the main thread)
   	  trainCallBack
		)
		if i % 5 == 0 then
   		donkeys:synchronize()
		end	
	end
end
