import torch
import numpy as np
np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)
tensor2array=torch_data.numpy()
print('\nnumpy array:\n',np_data,
        '\ntorch tensor:\n',torch_data,
        '\ntensor array:\n',tensor2array)

#abs
data=[-1,-2,1,2]
tensor=torch.FloatTensor(data)
print('\nabs',
        '\nnumpy: ',np.abs(data),
        '\ntorch: ',torch.abs(tensor))       
#sin
print('\nsin',
    '\nnumpy: ',np.sin(data),
    '\ntorch: ',torch.sin(tensor))
#mean
print('\nmean',
    '\nnumpy: ',np.mean(data),
    '\ntorch: ',torch.mean(tensor))
#点乘
data=[[1,2],[3,4]]
tensor=torch.FloatTensor(data)
print('\nmatrix multiplication(matmul)',
    '\nnumpy: ',np.matmul(data,data),
    '\ntorch: ',torch.mm(tensor,tensor))
data=np.array(data)
print('\nmatrix multiplication(dot)',
    '\nnumpy: ',data.dot(data)
    )
#torch 已不支持二维数组.dot，tensor.dot(tensor),会报错无法运行。