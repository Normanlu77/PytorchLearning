import torch
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
#函数 y=a*x^2*b
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size()) #y增加了噪点

#x,y=torch.autograd.Variable(x),Variable(y)

#plt.scatter(x.data.numpy(),y.data.numpy())
#plt.show()

class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=nn.Linear(n_feature,n_hidden)
        self.predict=nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(n_feature=1,n_hidden=10,n_output=1)
print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.2)
loss_func=nn.MSELoss() #MSELoss mean squared loss

plt.ion()

for t in range(200):
    prediction=net(x)

    loss=loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%loss.data.numpy(),fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()