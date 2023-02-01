import torch
from torch.autograd import Variable
import torch.nn as NN
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x,requires_grad=False),Variable(y,requires_grad=False)

class Net(NN.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=NN.Linear(n_feature,n_hidden)
        self.predict=NN.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x


def save():
    net1=Net(n_feature=1,n_hidden=10,n_output=1)
    optimizer=torch.optim.SGD(net1.parameters(),lr=0.5)
    loss_func=NN.MSELoss()

    for t in range(100):
        prediction=net1(x)
        loss=loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #plt.cla()
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)

    torch.save(net1,'net.pkl')
    torch.save(net1.state_dict(),'net_params.pkl')

def restore_net():
    net2=torch.load('net.pkl')
    prediction=net2(x)

    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
    #plt.show()

def restore_params():
    net3=Net(n_feature=1,n_hidden=10,n_output=1)
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction=net3(x)

    plt.subplot(133)
    plt.title('net3')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
    #plt.show()

save()

restore_net()

restore_params()

plt.show()


