x = torch.ones(4)
y = torch.ones(5)
z = torch.ones(2)

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

h1.data.module.weight = torch.ones(2,4)
h1.data.module.bias = torch.ones(2)
h2.data.module.weight = torch.ones(2,5)
h2.data.module.bias = torch.ones(2)

print(output:forward({x,y,z}))