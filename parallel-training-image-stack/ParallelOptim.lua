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

function ParallelOptim:singleProcessTrain()
	self.parallelTrainer:updateOutputAccGradParams()
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

function ParallelOptim:sendDataInfo(batchSize, trainLoader)
  self.parallelTrainer:sendDataInfo(batchSize, trainLoader)
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
function ParallelOptim:optimize(optimMethod)
	assert(optimMethod)
	assert(#self.parallelTrainer:getErrs() == self.numMachines)
	assert(self.modulesToOptState)

	local errTotal = 0
  	local accTotal = 0
	for _,errValue in pairs(self.parallelTrainer:getErrs()) do
		errTotal = errTotal + errValue
	end

  	for _,accValue in pairs(self.parallelTrainer:getAccs()) do 
    		accTotal = accTotal + accValue
  	end

	errTotal = errTotal/self.numMachines
  	accTotal = accTotal/self.numMachines

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
    return errTotal, accTotal
end
