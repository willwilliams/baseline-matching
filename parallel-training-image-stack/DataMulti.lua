require 'parallel'
require 'paths'

paths.dofile('dataworker.lua')

local DataMulti, parent = torch.class('DataMulti')

--[[
	machinesList : list of machines we are parallelizing across
		machinesList will be a text file with separate server IP on each line
	model : which model we are training in parallel
]]
function DataMulti:__init(machinesList, model)

    if not model then 
        error "must specify a model!"
    end
    if not machinesList then 
        error "must specify list of machines!"
    end

    self.model = model

    local machineFile = io.open(machinesList, "r")
    io.input(machineFile)
    self.machine_list = {} -- key is machine id, value is the ip of the machine
    for line in io.lines() do 
    	table.insert(self.machine_list, line)
    end

    self.process_list = {} -- list of all processes in order created
    self.process_to_machine = {} -- key is the child process id, value is the machine id
    self.startNum = 0
    self.outputs = {}
    self.gradOutputs = {}
    self.errs = {}
    self.labels_cache = {}

    self.numMachines = #self.machine_list
    self:restartChildren()
    assert(#self.process_list == self.numMachines)
end

function DataMulti:_resetBatchBegin()
	self.outputs = {}
	self.gradOutputs = {}
	self.errs = {}
	self.labels_cache = {}
	self.startNum = 0
end

--[[function for reseting children, in the case that model is 
	loaded from a file
]]
function DataMulti:restartChildren() 
    parallel.reset()
    for i = 1, #self.machine_list do
    	local child = parallel.fork(self.machine_list[i], 'ssh -o StrictHostKeyChecking=no -Y -i /neural_networks/cluster-key', paths.findprogram('th'))
    	self.process_to_machine[child.id] = i
    	table.insert(self.process_list, child.id)
    end
    parallel.children:exec(dataWorker)
end

--[[sends small input chunk to process; repeats this until 
	all processes have done an input chunk
	calculates output of model, error of model (based on criterion),
	and gradients of loss with respect to the output
]]
function DataMulti:updateOutput(inputCPU, labelsCPU)
	if self.startNum % self.numMachines == 0 then 
		self:_resetBatchBegin()
	end
	
	self.startNum = self.startNum + 1

	local processID = self.process_list[self.startNum]
	
	-- cache labels for prediction purposes
	self.labels_cache[processID] = labelsCPU

	-- update output for each module
	local child = parallel.children[processID]
	child:join("computeOutput")

	-- object to pass to other process
	computeObject = {model = self.model, inputs = inputCPU, labels = labelsCPU}
	child:send(computeObject)
	if self.startNum % self.numMachines == 0 then
		outputTable = parallel.children:receive()
		print("Received outputs.")
		self:_convertOutputTable(outputTable)
		return self.outputs
	end
end

function DataMulti:getLabelsCache()
	return self.labels_cache
end

function DataMulti:_convertOutputTable(outputTable)
	local numOutputs = #outputTable
	assert(numOutputs == self.numMachines)
	for i, outputVal in pairs(outputTable) do  
		self.outputs[i] = outputVal.output
		self.errs[i] = outputVal.err
		self.gradOutputs[i] = outputVal.grad
	end
end

function DataMulti:accGradParameters(scale)
	local scale = scale or 1
	local gradOutput = self.gradOutputs
	for processID, grad in pairs(gradOutput) do
		local child = parallel.children[processID]
		child:join("gradParameter") 
		child:send(grad)
	end
	local _,gradients = self.model:parameters()
	--print(gradients)
	for processID,_ in pairs(gradOutput) do 
		local childGrads = parallel.children[processID]:receive()
	--	print(childGrads)
                for j = 1, #childGrads do
			gradients[j]:add(childGrads[j])
		end
	end
	for j = 1, #gradients do
	  gradients[j]:div(self.numMachines)
	end
end

function DataMulti:zeroGradParameters()
	self.model:zeroGradParameters()
end

function DataMulti:updateParameters(learningRate)
	self.model:updateParameters(learningRate)
end

function DataMulti:getErrs()
	return self.errs
end

function DataMulti:getOutputs()
	return self.outputs
end

function DataMulti:getGradOutputs()
	return self.gradOutputs
end

function DataMulti:extractModel()
	return self.model
end

function DataMulti:reset(stdv)
	self.model:reset(stdv)
	self:restartChildren()
end

function DataMulti:getNumMachines() 
	return self.numMachines
end
