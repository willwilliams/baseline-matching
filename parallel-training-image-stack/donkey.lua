paths.dofile('pairfileloader.lua')
paths.dofile('util.lua')
ffi=require 'ffi'
require 'image'

local mean, std
local trainCache = paths.concat(opt.cache, 'trainCache.t7')
local testCache = paths.concat(opt.cache, 'testCache.t7')
local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')

local loadSize = {3, 128, 128}
local sampleSize = {3, 64, 64}

local mean, std

local trainHook = function(self, toppath, botpath)
  local topinput = image.load(toppath, self.loadSize[1])
  local botinput = image.load(botpath, self.loadSize[1])
  topinput = image.scale(topinput, sampleSize[3], sampleSize[2])
  botinput = image.scale(botinput, sampleSize[3], sampleSize[2]) 
  combo = torch.cat(topinput, botinput, 1)
  for i=1,6 do
    if mean then combo[{{i},{},{}}]:add(-mean[i]) end
    --print(std)
    --if std then combo[{{i},{},{}}]:div(std[i]) end
  end
  if torch.uniform() > 0.5 then combo = image.hflip(combo); end
  return combo
end

local testHook = function(self, toppath, botpath)
  local topinput = image.load(toppath, self.loadSize[1])
  local botinput = image.load(botpath, self.loadSize[1])
  topinput = image.scale(topinput, sampleSize[3], sampleSize[2])
  botinput = image.scale(botinput, sampleSize[3], sampleSize[2])
  combo = torch.cat(topinput, botinput, 1)
  for i=1,6 do
    if mean then combo[{{i},{},{}}]:add(-mean[i]) end
    --if std then combo[{{i},{},{}}]:div(std[i]) end
  end
  return combo
end

if paths.filep(trainCache) then
  print('Loading train metadata from cache')
  trainLoader = torch.load(trainCache)
  trainLoader.sampleHookTrain = trainHook
else
  trainLoader = pairfileLoader{
    loadfilepath = "/neural_networks/Washington12pairs.txt",
    loadSize = {3, 128, 128}, 
    sampleSize = {3, 64, 64}, 
    numClasses = 2,
    directory = opt.trainData,
    sampleHookTrain = trainHook
  }
  torch.save(trainCache, trainLoader)
end
if paths.filep(testCache) then
  print('Loading train metadata from cache')
  testLoader = torch.load(testCache)
  testLoader.sampleHookTrain = testHook
else
  testLoader = pairfileLoader{
    loadfilepath = "/neural_networks/Washington13pairs.txt", 
    loadSize = {3, 128, 128}, 
    sampleSize = {3, 64, 64}, 
    numClasses = 2,
    directory = opt.testData,
    sampleHookTrain = testHook
  }
  torch.save(testCache, testLoader)
end
collectgarbage()

if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
  local tm = torch.Timer()
  local nSamples = 10000
  print('Estimating the mean(per-channel, shared for all pixels)')
  local meanEstimate = {0, 0, 0, 0, 0, 0}
  for i = 1, nSamples do
    local img = trainLoader:sample(1)
    for j=1,6 do
      meanEstimate[j] = meanEstimate[j] + img[j]:mean()
    end
  end
  for j=1,6 do 
    meanEstimate[j] = meanEstimate[j]/nSamples
  end
  mean = meanEstimate

  print('Estimating std over images')
  local stdEstimate = {0, 0, 0, 0, 0, 0}
  for i = 1, nSamples do
    local img = trainLoader:sample(1)
    for j = 1,6 do
      stdEstimate[j] = stdEstimate[j] + img[j]:std()
    end
  end
  for j = 1, 6 do
    stdEstimate[j] = stdEstimate[j]/nSamples
  end
  std = stdEstimate
  do
    local testmean = 0
    local teststd = 0
    for i =1, 100 do
      local img = trainLoader:sample(1)
      testmean = testmean + img:mean()
      teststd = teststd + img:std()
    end
    print('Stats of 100 randomly sampled images after normalizing. Mean: '
      .. testmean/100 .. ' Std: ' .. teststd/100)
  end
  local cache = {}
  cache.mean = mean
  cache.std = std
  torch.save(meanstdCache, cache)
end
