testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local top1_tot
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_tot = 0   
   loss = 0
   for i=1,opt.nTest/opt.testBatchSize do
      local indexStart = (i-1) * opt.testBatchSize + 1
      local indexEnd = (indexStart + opt.testBatchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd)
            return sendTensor(inputs), sendTensor(labels)
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
      if i % 5 == 0 then
         donkeys:synchronize()
      end
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_tot = top1_tot*100/(opt.nTest)  
   loss = loss / (opt.nTest/opt.testBatchSize) -- because loss is calculated per batch
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_tot,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 accuracy %.2f top-1 %d',
                       1, timer:time().real, loss, top1_tot, top1_tot*opt.nTest/100))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsThread, labelsThread)
   batchNumber = batchNumber + opt.testBatchSize
   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(labelsThread, labelsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)


   local outputs = model:forward(inputs)
   print(outputs:size())
   print(inputs:size())
   print(labels:size())
   local err = criterion:forward(outputs, labels)
   cutorch.synchronize()
   local pred = outputs:float()
   loss = loss + err

  local function topstats(p, g)
     local top1 = 0;     
      _,p = p:float():sort(2, true)
     if p[1][1] == g then
         top1 = top1 + 1
     end
     return top1
  end

  for i=1,pred:size(1) do
     local p = pred[{{i},{}}]
     local g = labelsCPU[i]
     -- center
     local top1 = topstats(p, g)
     top1_tot = top1_tot + top1
  end
  if batchNumber % 10 == 0 then
     print(('Epoch: Testing [%d][%d/%d]'):format(
              1, batchNumber, opt.nTest))
  end
end
