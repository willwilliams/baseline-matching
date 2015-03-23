require 'optim'
require 'paths'

paths.dofile('DataMulti.lua')

local ParallelOptim, parent = torch.class('ParallelOptim', 'nn.Optim')

function ParallelOptim:__init(machinesList, model, optState, criterion, checkpoint_data)
	parent:__init(model, optState, checkpoint_data)
	self.parallelTrainer = DataMulti(machinesList, model, criterion)
	self.numMachines = self.parallelTrainer:getNumMachines()
end

function ParallelOptim:singleProcessTrain(inputsCPU, labelsCPU)
	self.parallelTrainer:updateOutput(inputsCPU, labelsCPU)
end

function ParallelOptim:getNumMachines()
	return self.numMachines
end

function getNumProcessForBatch()
	return self.parallelTrainer.startNum
end

function getCachedLabels()
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
function ParallelOptim:optimize(optimMethod)
	assert(optimMethod)
	assert(#self.parallelTrainer:getOutputs() == self.numMachines)
	assert(self.modulesToOptState)

	local errTotal = 0
	for _,errValue in self.parallelTrainer:getErrs() do
		errTotal = errTotal + errValue
	end

	self.parallelTrainer:zeroGradParameters()
	self.parallelTrainer:accGradParameters()

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
    return errTotal, outputs
end