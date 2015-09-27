require 'nn'

-- Network-in-Network
-- achieves 92% with BN and 88% without

local backend_name = 'cudnn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end

model = nn.Sequential()
function Block(...)
  local arg = {...}
  model:add(backend.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(backend.ReLU(true))
  return model
end

Block(3,128,10,10,3,3,2,2)
Block(128,64,1,1)
Block(64,64,1,1)
model:add(backend.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout())
Block(64,128,5,5,2,2,2,2)
Block(128,64,1,1)
Block(64,64,1,1)
model:add(backend.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout())
Block(64,128,5,5,2,2,2,2)
Block(128,64,1,1)
Block(64,64,1,1)
model:add(backend.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout())
Block(64,128,5,5,1,1,2,2)
Block(128,64,1,1)
Block(64,64,1,1)
model:add(backend.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.View(512))

for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
  v.weight:normal(0,0.05)
  v.bias:zero()
end

-- print(#model:cuda():forward(torch.CudaTensor(4,3,768,512)))
-- batch size of 4 should work...

return model
