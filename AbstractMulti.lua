require 'parallel'

local AbstractMulti, parent = torch.class('nn.AbstractMulti', 'nn.Container')

--[[
	dimension : dimension of data to parallelize
	machinesList : list of machines we are parallelizing across
		machinesList will be a text file with separate server location on each line
]]

function AbstractMulti:__init(dimension, machinesList)
    if not dimension then 
        error "must specify a dimension!"
    end
    if not machinesList then 
        error "must specify list of machines!"
    end
    parent.__init(self)

    local machineFile = io.open(machinesList, "r")
    io.input(machineFile)
    self.machine_list = {}
    for line in io.lines() do 
    	table.insert(self.machine_list, line)
    end

    self.modules = {}
    self.machine_assignments = {}
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.input_machine = {}
    self.gradOutput_machine = {}
    self.gradInput_machine = {}
end

function AbstractMulti:_freeCaches() 
	self.input_machine = {}
	self.gradOutput_machine = {}
	self.gradInput_machine = {}
end

function AbstractMulti:nextMachine() 
	local machineID = #self.machine_assignments % #self.machine_list + 1
	return machineID
end

-- function AbstractMulti._getBuffer() 
-- 	local device = 
-- end

function AbstractMulti:add(module, machineID)
	table.insert(self.modules, module)
	local machineID = machineID or self:nextMachine()
	table.insert(self.machine_assignments, machineID)
	return self
end

function AbstractMulti:get(index)
	return self.modules[index]
end

function AbstractMulti:machineSend()
end

function AbstractMulti:updateOutput()
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


