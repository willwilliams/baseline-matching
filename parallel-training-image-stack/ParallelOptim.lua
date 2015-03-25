local pl = require('pl.import_into')()

require 'optim'
require 'paths'

paths.dofile('DataMulti.lua')

local ParallelOptim, parent = torch.class('ParallelOptim', 'nn.Optim')

function ParallelOptim:__init(machinesList, model, optState, checkpoint_data)
	parent:__init(model, optState, checkpoint_data)
	self.parallelTrainer = DataMulti(machinesList, model)
	self.numMachines = self.parallelTrainer:getNumMachines()
end

function ParallelOptim:singleProcessTrain(inputsCPU, labelsCPU)
	self.parallelTrainer:updateOutputAccGradParams(inputsCPU, labelsCPU)
end

function ParallelOptim:getNumMachines()
	return self.numMachines
end

function ParallelOptim:getNumProcessForBatch()
	return self.parallelTrainer.startNum
end

function ParallelOptim:getCachedLabels()
	return self.parallelTrainer:getLabelsCache()
end

local function get_device_for_module(mod)
   local dev_id = nil
   for name, val in pairs(mod) do
       if torch.typename(val) == 'torch.CudaTensor' then
           local this_dev = val:getDevice()
           if this_dev ~= 0 then
               -- _make sure the tensors are allocated consistently
               assert(dev_id == nil or dev_id == this_dev)
               dev_id = this_dev
           end
       end
   end
   return dev_id -- _may still be zero if none are allocated.
end

local function on_device_for_module(mod, f)
    local this_dev = get_device_for_module(mod)
    if this_dev ~= nil then
        return cutorch.withDevice(this_dev, f)
    end
    return f()
end

--[[
	assumes that all outputs for a particular batch have finished computing for all processes
	calculates the gradients and performs update according to the optimization method
]]
function ParallelOptim:startOptimizationLoop()
  function optimizationLoop()
    local totalFinished = 0
    while true do
      local _,gradients = self.model:parameters()
      outputVals = parallel.children:receive("noblock")
      local numFinished = 0
      local totalErr = 0
      local totalAcc = 0
      for id, result in pairs(outputVals)
        if result ~= nil
          numFinished = numFinished + 1
          local childGrads = result.gradParam
          local childErr = result.err
          totalErr = totalErr + childErr
          local childAcc = result.acc
          if childAcc > 0 then totalAcc = totalAcc + childAcc end
          for j = 1, #childGrads do
            gradients[j]:add(childGrads[j])
          end
        end
      end
      if numFinished > 0 then
        for i = 1, #gradients do
          gradients[i]:div(numFinished)
        end
        totalErr = totalErr/numFinished
        self:optimize(optim.sgd, totalErr)
        totalFinished = totalFinished + numFinished
      end
    end
  end

end

function ParallelOptim:optimize(optimMethod, errTotal)
	assert(optimMethod)
	assert(#self.parallelTrainer:getOutputs() == self.numMachines)
	assert(self.modulesToOptState)

	local curGrad
	local curParam
	local function fEvalMod(x)
		return errTotal, curGrad
	end

	for curMod, opt in pairs(self.modulesToOptState) do
        on_device_for_module(curMod, function()
            local curModParams = self.weight_bias_parameters(curMod)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(curModParams) == 0 or
                   pl.tablex.size(curModParams) == 2)
            if curModParams then
                for i, tensor in ipairs(curModParams) do
                    if curModParams[i] then
                        -- expect param, gradParam pair
                        curParam, curGrad = table.unpack(curModParams[i])
                        assert(curParam and curGrad)
                        optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end
        end)
    end
    self.parallelTrainer:zeroGradParameters()
    return errTotal, self.parallelTrainer:getOutputs()
end
