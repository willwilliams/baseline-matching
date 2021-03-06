require 'nn'
require 'cunn'
require 'optim'
require 'cudnn'

function createModel()
  local networkModel = nn.Sequential() 
  networkModel:add(nn.SpatialConvolutionMM(6,40,5,5,2,2,2,2))       
  networkModel:add(nn.ReLU(true))
  networkModel:add(nn.SpatialMaxPooling(2,2,2,2))                   
  networkModel:add(nn.SpatialConvolutionMM(40,60,5,5,1,1,2,2))      
  networkModel:add(nn.ReLU(true))
  networkModel:add(nn.SpatialMaxPooling(2,2,2,2))                
  networkModel:add(nn.SpatialConvolutionMM(60,100,3,3,1,1,1,1))     
  networkModel:add(nn.ReLU(true))
  networkModel:add(nn.SpatialMaxPooling(2,2,2,2))              
  networkModel:add(nn.SpatialConvolutionMM(100,160,3,3,1,1,1,1))    
  networkModel:add(nn.ReLU(true))
  networkModel:add(nn.SpatialMaxPooling(2,2,2,2))  
  networkModel:add(nn.SpatialConvolutionMM(160,200,3,3,1,1,1,1)) 
  networkModel:add(nn.ReLU(true))
  networkModel:add(nn.SpatialMaxPooling(2,2,2,2)) 

  local classifier = nn.Sequential()
  classifier:add(nn.View(200*1*1))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(200*1*1, 2))
  classifier:add(nn.LogSoftMax())

  local model = nn.Sequential():add(networkModel):add(classifier)

  return model
end

model = createModel()

print('=> Model')
print(model)


print('==> Converting model to CUDA')
model = model:cuda()

collectgarbage()
