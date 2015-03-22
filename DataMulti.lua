require 'parallel'
require 'paths'

paths.dofile("DataWorker.lua")

local DatMulti, parent = torch.class('DataMulti', 'nn.Container')

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

    self:restartChildren()
end

--[[function for reseting children, in the case that model is 
	loaded from a file
]]
function DataMulti:restartChildren() 
	parallel.reset()
 	for i = 1, #machine_list do
    	local child = parallel.fork(self.machine_list[i], 'ssh')
    	self.process_to_machine[child.id] = i
    	table.insert(self.process_list, child.id)
    end
    parallel.children.exec(dataWorker)
end

function DataMulti:_freeCaches() 
	self.input_machine = {}
	self.gradOutput_machine = {}
	self.gradInput_machine = {}
end

function DataMulti:machineSend(dest, source, destID, sourceID)
	assert(torch.typename(dest) == 'torch.CudaTensor')
    assert(torch.typename(source) == 'torch.CudaTensor')

end

--[[sends small input chunk to process; repeats this until 
	all processes have done an input chunk
]]
function DataMulti:updateOutput(input)
	self.startNum = self.startNum + 1

	-- update output for each module
	local processID = self.process_list[self.startNum]
	local child = parallel.children[processID]
	child.join("computeOutput")

	-- object to pass to other process
	computeObject = {model = self.model, inputs = input}
	child.send(computeObject)
	if self.startNum % #self.process_list == 0 do
		self.startNum = 0
		outputTable = parallel.children:receive()
		assert(#outputTable == #self.process_list)
		self.outputs = outputTable
		return self.outputs
	end
end

function DataMulti:accGradParameters(gradOutput, scale)
	local scale = scale or 1
	for indx, grad in pairs(gradOutput) do
		local processID = self.process_list[indx]
		local child = parallel.children[processID]
		child.join("gradParameter") 
		child.send(grad)
	end
	local _,gradients = self.model:parameters()
	for i = 1, #gradOutput do 
		local processID = self.process_list[i]
		local childGrads = parallel.children[processID].receive()
		for j = 1, #childGrads
			gradients[j]:add(childGrads[j])
		end
	end
end

function DataMulti:zeroGradParameters()
	self.model:zeroGradParameters()
end

function DataMulti:updateParameters(learningRate)
	self.model:updateParameters(learningRate)
end

function DataMulti:extractModel()
	return self.model
end

function DataMulti:reset(stdv)
	self.model:reset(stdv)
	self:restartChildren()
end


