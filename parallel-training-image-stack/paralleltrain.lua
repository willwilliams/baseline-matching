require 'optim'
require 'fbnn'

paths.dofile('ParallelOptim.lua')
--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

local optimator = ParallelOptim(opt.machines, model, optimState, criterion)

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, top5_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   optimator:setParameters(params)
   if newRegime then
       -- Zero the momentum vector by throwing away previous state.
       optimator = nn.ParallelOptim(model, optimState)
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()
   model:cuda() -- get it back on the right GPUs.

   top1_epoch = 0
   top5_epoch = 0
   loss_epoch = 0

   for i=1,opt.epochSize*optimator:getNumMachines() do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
            return sendTensor(inputs), sendTensor(labels)
         end,
         -- the end callback (runs in the main thread)
         trainSingleProcess
      )
      if i % 5 == 0 then
         donkeys:synchronize()
      end
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   local function sanitize(net)
      net:for_each(function (val)
            for name,field in pairs(val) do
               if torch.type(field) == 'cdata' then val[name] = nil end
               if name == 'homeGradBuffers' then val[name] = nil end
               if name == 'input_gpu' then val['input_gpu'] = {} end
               if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
               if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
               if (name == 'output' or name == 'gradInput')
               and torch.type(field) == 'torch.CudaTensor' then
                  cutorch.withDevice(field:getDevice(), function() val[name] = field.new() end)
               end
            end
      end)
   end
   sanitize(model)
   torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- create tensor buffers in main thread and deallocate their storages.
-- the thread loaders will push their storages to these buffers when done loading
local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainSingleProcess(inputsThread, labelsThread)
  cutorch.synchronize()
   -- set the data and labels to the main thread tensor buffers (free any existing storage)
  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(labelsThread, labelsCPU)

  optimator:singleProcessTrain(inputsCPU, labelsCPU)
  if(optimator:getNumProcessForBatch() % optimator:getNumMachines() == 0) then
    local err, outputs = optimator:optimize(optim.sgd)
    cutorch.synchronize()
    -- Calculate top-1 and top-5 errors, and print information
    print(('Epoch: [%d][%d/%d]\tErr %.4f LR %.0e'):format(
          epoch, batchNumber, opt.epochSize, err,
          optimState.learningRate))
    batchNumber = batchNumber + 1
    loss_epoch = loss_epoch + err
    if (batchNumber % 15) == 0 then
       -- top-1 and top-5 error
      local top1 = 0
      do
        local labelsCache = optimator:getCachedLabels()
        for processID, output in pairs(outputs) do 
          local _,prediction_sorted = output:float():sort(2, true) -- descending
          local gt = labelsCache[processID]
          for i=1,opt.batchSize do
            local pi = prediction_sorted[i]
            if pi[1] == gt[i] then top1 = top1 + 1 end
          end
        end
        top1_epoch = top1_epoch + top1
        top1 = top1 * 100 / (opt.batchSize * optimator:getNumMachines)
      end

       -- print info
      print(string.format('Accuracy ' ..
                              'top1-%%: %.2f \t' ..
                              'Loss: %.4f \t' ..
                              'LR: %.0e',
                           top1, err,
                           optimState.learningRate))
    end
  end
end