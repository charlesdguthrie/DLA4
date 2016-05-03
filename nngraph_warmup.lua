require('nngraph')

--inputs
ix = nn.Identity()()
iy = nn.Identity()()
iz = nn.Identity()()
--Wx + b
h1 = nn.Linear(4,2)({ix})
h2 = nn.Linear(5,2)({iy})
-- tanh, sigmoid
tanh = nn.Tanh()({h1})
sigmoid = nn.Sigmoid()({h2})
-- square
tsq = nn.Square()({tanh})
ssq = nn.Square()({sigmoid})
-- cmul
cmul = nn.CMulTable()({tsq,ssq})
a = nn.CAddTable()({cmul,iz})
-- final graph
output = nn.gModule({ix,iy,iz},{a})

--graph.dot(output.fg, 'output','outputBaseName')

--initialize weights
h1.data.module.weight = torch.ones(2,4)
h1.data.module.bias = torch.ones(2)
h2.data.module.weight = torch.ones(2,5)
h2.data.module.bias = torch.ones(2)

x = torch.ones(4)
y = torch.ones(5)
z = torch.ones(2)
gradOutput = torch.ones(2)

print("===> x:")
print(x)

print("===> y:")
print(y)

print("===> z:")
print(z)

print("===> forward propagation output:")
print(output:forward({x,y,z}))

residual = output:backward({x,y,z},gradOutput)
print("===> backward propagation output:")
print("===> w.r.t x:")
print(residual[1])
print("===> w.r.t y:")
print(residual[2])
print("===> w.r.t z:")
print(residual[3])
print("===> w.r.t Wx")
print(h1.data.module.gradWeight)
print("===> w.r.t. b1")
print(h1.data.module.gradBias)
print("===> w.r.t Wy")
print(h2.data.module.gradWeight)
print("===> w.r.t. b2")
print(h2.data.module.gradBias)
