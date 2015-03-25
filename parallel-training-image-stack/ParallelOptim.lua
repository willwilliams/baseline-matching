local pl = require('pl.import_into')()

require 'optim'
require 'paths'
require 'parallel'

paths.dofile('dataworker.lua')

local ParallelOptim, parent = torch.class('ParallelOptim', 'nn.Optim')

function ParallelOptim:__init(machinesList, model, optState, checkpoint_data)
	parent:__init(model, optState, checkpoint_data)
  local machineFile = io.open(machinesList, "r")
  io.input(machineFile)
  self.machine_list = {} -- key is machine id, value is the ip of the machine
  for line in io.lines() do 
    table.insert(self.machine_list, line)
  end

  self.process_list = {} -- list of all processes in order created
  self.process_to_machine = {} -- key is the child process id, value is the machine id

  self.numMachines = #self.machine_list
  self:restartChildren()
  assert(#self.process_list == self.numMachines)
end

function ParallelOptim:restartChildren() 
  parallel.reset()
  for i = 1, #self.machine_list do
    local child = parallel.fork(self.machine_list[i], 'ssh -o StrictHostKeyChecking=no -Y -i /neural_networks/cluster-key', paths.findprogram('th'))
    self.process_to_machine[child.id] = i
    table.insert(self.process_list, child.id)
  end
  parallel.children:exec(dataWorker)
end

function ParallelOptim:singleProcessTrain(inputsCPU, labelsCPU)
	self:updateOutputAccGradParams(inputsCPU, labelsCPU)
end

function ParallelOptim:getNumMachines()
	return self.numMachines
end

function ParallelOptim:getNumProcessForBatch()
	return self.startNum
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

function ParallelOptim:accGradParams()
  local _,gradients = self.model:parameters()
  local totalErr = 0
  local totalAcc = 0

  for i = 1, self.numMachines do
    local id = self.process_list[i]
    outputVal = parallel.children[id]:receive()
    print("Received outputs from child " .. id)
    local childGrads = outputVal.gradParam
    for j = 1, #childGrads do
      gradients[j]:add(childGrads[j])
    end
    totalErr = totalErr + outputVal.err
    totalAcc = totalAcc + outputVal.acc
  end
  for i = 1, #gradients do
    gradients[i]:div(self.numMachines)
  end

  totalErr = totalErr/self.numMachines
  totalAcc = totalAcc/self.numMachines
  return totalErr, totalAcc
end

--[[
	assumes that all outputs for a particular batch have finished computing for all processes
	calculates the gradients and performs update according to the optimization method
]]
function ParallelOptim:optimize(optimMethod)
	assert(optimMethod)
	assert(#self:getOutputs() == self.numMachines)
	assert(self.modulesToOptState)

  self.model:zeroGradParameters()
  local errTotal, accTotal = self:accGradParams()

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
  self.model:zeroGradParameters()
  return errTotal, accTotal
end

function ParallelOptim:getChildProcessByIndex(index)
  return parallel.children[self.process_list[index]]  
end

function ParallelOptim:extractModel()
  return self.model
end
