require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local dataset = torch.class('pairfileLoader')

local initcheck = argcheck{
  pack = true, 
  
  {check=function(loadfilepath)
     local out = true;
     if type(loadfilepath) ~= 'string' then
       print('error')
       out = false
     end
     return out
   end,
   name="loadfilepath", 
   type="string",
   help="path of file containing list of image pairs"},

  {name="directory", 
   type="string",
   help="path of prefix directory for loadfilepath", 
   default=""},

  {name="sampleHookTrain", 
   type="function", 
   help="takes image path as input and applied to sample during training",
   opt=true},

  {name="sampleSize", 
   type="table", 
   help="a consistent sample size to resize images to",
   opt=true},
  
  {name="loadSize", 
   type="table",
   help="a size to load the images to, initially",
   opt=true},
  
  {name="numClasses", 
   type="number", 
   help="how many classes are there to classify into", 
   opt=true},
}

function dataset:__init(...)
  local args = initcheck(...)
  print("args:")
  print(args)
  for k,v in pairs(args) do print(k); self[k] = v end
  if not self.loadSize then self.loadSize = self.sampleSize; end
  if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
  if not self.numClasses then self.numClasses = 2 end
  print(self.loadSize)
  print(self.sampleSize)
  self.imageListTop = torch.CharTensor()
  self.imageListBottom = torch.CharTensor()
  self.imageClass = torch.LongTensor()
  local tokensList = {}
  local maxTopLength = 0
  local maxBotLength = 0
  for line in io.lines(self.loadfilepath) do
    local tokens = {}
    for token in string.gmatch(line, "%S+") do table.insert(tokens, token) end
    if string.len(tokens[1]) > maxTopLength then maxTopLength = string.len(tokens[1]) end
    if string.len(tokens[2]) > maxBotLength then maxBotLength = string.len(tokens[2]) end
    tokens[3] = tonumber(tokens[3])
    table.insert(tokensList, tokens)
  end
  maxTopLength = maxTopLength + 1 + string.len(self.directory)
  maxBotLength = maxBotLength + 1 + string.len(self.directory)
  local length = table.getn(tokensList)
  self.imageListTop:resize(length, maxTopLength):fill(0)
  self.imageListBottom:resize(length, maxBotLength):fill(0)
  self.imageClass:resize(length)
  self.numImages = length
  local top_data = self.imageListTop:data()
  local bot_data = self.imageListBottom:data()
  for i = 1, length do
    ffi.copy(top_data, self.directory .. tokensList[i][1])
    top_data = top_data + maxTopLength
    ffi.copy(bot_data, self.directory .. tokensList[i][2])
    bot_data = bot_data + maxBotLength
    self.imageClass[i] = tokensList[i][3]
  end 
end

function dataset:defaultSampleHook(imgpathtop, imgpathbot)
  local outtop = image.load(imgpathtop, self.loadSize[1])
  outtop = image.scale(outtop, self.sampleSize[3], self.sampleSize[2])
  local outbot = image.load(imgpathbot, self.loadSize[1])
  outbot = image.scale(outbot, self.sampleSize[3], self.sampleSize[2])
  local outcomb = torch.cat(outtop, outbot, 1)
  return outcomb
end

local function tableToOutput(self, dataTable, scalarTable)
  local data, scalarLabels, labels
  local quantity = #scalarTable
  local samplesPerDraw = 1
  if quantity == 1 and samplesPerDraw == 1 then 
    data = dataTable[1]
    scalarLabels = scalarTable[1]
    labels = torch.LongTensor(self.numClasses):fill(-1)
    labels[scalarLabels] = 1
  else 
    data = torch.Tensor(quantity*samplesPerDraw, 2*self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
    scalarLabels = torch.LongTensor(quantity * samplesPerDraw)
    labels = torch.LongTensor(quantity*samplesPerDraw, self.numClasses):fill(-1)
    for i=1,#dataTable do
      --print("Adding input to data table.")
      --print("Data input size:")
      --print(dataTable[i]:size())
      --print("Data chunk size:")
      --print(data:size())
      local idx = (i - 1)*samplesPerDraw
      data[{{idx+1, idx+samplesPerDraw}}]:copy(dataTable[i])
      scalarLabels[{{idx+1, idx+samplesPerDraw}}]:fill(scalarTable[i])
      labels[{{idx + 1, idx+samplesPerDraw}, {scalarTable[i]}}]:fill(1)
    end
  end
  return data, scalarLabels, labels
end

function dataset:sample(quantity)
  quantity = quantity or 1
  local dataTable = {}
  local scalarTable = {}
  for i = 1, quantity do
    local indx = math.ceil(torch.uniform()*self.numImages)
    local imgpathtop = ffi.string(torch.data(self.imageListTop[indx]))
    local imgpathbot = ffi.string(torch.data(self.imageListBottom[indx]))
    local label = self.imageClass[indx]
    local out = self:sampleHookTrain(imgpathtop, imgpathbot)
    table.insert(dataTable, out)
    table.insert(scalarTable, label + 1)
  end
  local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
  return data, scalarLabels, labels
end

function dataset:get(i1, i2)
  local indices, quantity
  if type(i1) == 'number' then
    if type(i2) == 'number' then
      indices = torch.range(i1, i2);
      quantity = i2 - i1 + 1;
    else 
      indices = {i1}; quantity = 1
    end
  elseif type(i1) == 'table' then 
    indices = i1; quantity = #i1;
  elseif (type(i1) == 'userdata' and i1:nDimension() == 1) then
    indices = i1; quantity = (#i1)[1];    -- tensor
  else
    error('Unsupported input types: ' .. type(i1) .. ' ' .. type(i2))
  end 
  assert(quantity > 0)
  local dataTable = {}
  local scalarTable = {}
  for i=1, quantity do 
    local imgpathtop = ffi.string(torch.data(self.imageListTop[indices[i]]))
    local imgpathbot = ffi.string(torch.data(self.imageListBottom[indices[i]]))
    local label = self.imageClass[indices[i]]
    local out = self:sampleHookTrain(imgpathtop, imgpathbot)
    table.insert(dataTable, out)
    table.insert(scalarTable, label + 1)
  end
  local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
  return data, scalarLabels, labels
end
