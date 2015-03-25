function doMain() 
  require 'torch'
  require 'cutorch'
  require 'paths'
  require 'xlua'
  require 'optim'
  require 'nn'
  require 'cudnn'
  require 'fbnn'
  require 'cunn'
  
  local opts = paths.dofile('parallelopts.lua')
  
  opt = opts.parse(arg)
  print(opt)
  
  torch.setdefaulttensortype('torch.FloatTensor')
  
  torch.seed()
  print('Saving everything to: ' .. opt.save)
  os.execute('mkdir -p ' .. opt.save)
  
  paths.dofile('data.lua') 
  paths.dofile('donkey.lua') -- to bring trainLoader in namespace so we can serialize + send to other stuff
  paths.dofile('model.lua')
  paths.dofile('util.lua')
  paths.dofile('paralleltrain.lua')
  paths.dofile('test.lua')
  
  epoch = opt.epochNumber
  
  for i=1,opt.nEpochs do
     train()
     test()
     epoch = epoch + 1
  end
end
