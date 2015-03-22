require 'parallel'
require 'paths'

paths.dofile("DataWorker.lua")

local AbstractMulti, parent = torch.class('nn.AbstractMulti', 'nn.Container')

--[[
	dimension : dimension of data to parallelize
	machinesList : list of machines we are parallelizing across
		machinesList will be a text file with separate server IP on each line
]]

function AbstractMulti:__init(dimension, machinesList)

    if not dimension then 
        error "must specify a dimension!"
    end
    if not machinesList then 
        error "must specify list of machines!"
    end
    parent.__init(self)
    self.modules = {} -- list of modules - modules[i] will be handled by process process_list[i]
    self.size = torch.LongStorage()
    self.dimension = dimension

    local machineFile = io.open(machinesList, "r")
    io.input(machineFile)
    self.machine_list = {} -- key is machine id, value is the ip of the machine
    for line in io.lines() do 
    	table.insert(self.machine_list, line)
    end

    self.process_list = {} -- list of all processes in order created
    self.process_to_machine = {} -- key is the child process id, value is the machine id
    -- self.input_machine = {}
    -- self.gradOutput_machine = {}
    -- self.gradInput_machine = {}
    self.startNum = 0
    self:restartChildren()
end

--[[function for reseting children, in the case that model is 
	loaded from a file
]]
function AbstractMulti:restartChildren() 
	parallel.reset()
 	for i = 1, #machine_list do
    	local childID = parallel.fork(self.machine_list[i], 'ssh')
    	self.process_to_machine[childID] = i
    	table.insert(self.process_list, childID)
    end
end

function AbstractMulti:_freeCaches() 
	self.input_machine = {}
	self.gradOutput_machine = {}
	self.gradInput_machine = {}
end

-- function AbstractMulti:nextProcess() 
-- 	local numModules = #self.modules
-- 	assert(numModules < #self.machine_list)
-- 	return #self.process_assignments + 1
-- end

-- function AbstractMulti._getBuffer() 
-- 	local device = 
-- end

function AbstractMulti:add(module)
	assert(#self.modules < #self.machine_list)
	table.insert(self.modules, module)
	return self
end

function AbstractMulti:get(index)
	return self.modules[index]
end

function AbstractMulti:machineSend(dest, source, destID, sourceID)
	assert(torch.typename(dest) == 'torch.CudaTensor')
    assert(torch.typename(source) == 'torch.CudaTensor')

end

function AbstractMulti:updateOutput(input)
	self.startNum = self.startNum + 1

	-- update output for each module
	for i, module in ipairs(self.modules) do
		local processID = self.process_list[i]
		parallel.children[processID].join("computeOutput")
		parallel.children[processID].send(input)
end

function AbstractMulti:_distributeGradOutput()
end

function AbstractMulti:updateGradInput()
end

function AbstractMulti:_mixGrads()
end

function AbstractMulti:accGradParameters()
end

function AbstractMulti:zeroGradParameters()
end

function AbstractMulti:updateParameters()
end

function AbstractMulti:share()
end

function AbstractMulti:clone()
end

function AbstractMulti:reset()
end


