# import numpy as np
# import torch
#
# x=torch.ones((3,4),dtype=torch.float32)
# #print(x)
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x=x.to(device)
# x.requires_grad=True
#
# # for  i in range(2):
# #     y=x**2
# #     z=torch.sum(y)
# #     z.backward()
# #     dx=x.grad
# #     print('x的梯度值为：{}'.format(dx))
# #     # print('x的梯度函数为：{}'.format(x.grad_fn))
# #     x.data=x.data-0.1*dx.data
# #     x.grad.data=torch.tensor(np.zeros((3,4)),dtype=torch.float32).to(device)
# #     print('x的值为：{}'.format(x))
# #     print('-' * 100)
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1=nn.Conv2d(1,6,3)
#         self.conv2=nn.Conv2d(6,16,3)
#         self.linear1=nn.Linear(576,32)
#         self.linear2=nn.Linear(32,2)
#     def forward(self,x):
#         x=F.max_pool2d(F.relu(self.conv1(x)),kernel_size=(2,2),stride=(2,2))
#         x=F.max_pool2d(F.relu(self.conv2(x)),2)
#         print(self.num_flat_features(x))
#         x=x.view(-1,self.num_flat_features(x))
#         x=F.relu(self.linear1(x))
#         x=F.relu(self.linear2(x))
#         return x
#     def num_flat_features(self,x):
#         size=x.shape[1:]
#         num_features=1
#         for i in size:
#             num_features*=i
#         return num_features
# net=Net()
# learning_rate=0.01

# print(net)
# params=list(net.parameters())
# print(params[0].shape) #六个滤波器是在第一维重叠在一起的，torch.Size([6, 1, 3, 3])
# input=torch.tensor(np.random.randn(1,1,32,32),dtype=torch.float32,requires_grad=True)
# target=torch.randn(1,2,dtype=torch.float32)
# import torch.optim as optim
# optimizer=optim.SGD(net.parameters(),lr=learning_rate)
# criterion=nn.MSELoss()
# losses=[]
# for i in range(10):
#     out = net(input)
#     loss = criterion(out, target)
#     optimizer.zero_grad()
#     loss.backward()
#     for f in net.parameters():
#         f.data.sub_(learning_rate*f.grad.data)
#     print(loss)
#     losses.append(loss.item())
#
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#训练一个分类器，使用数据集CIFAR10
transfrom=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),  #第一个元组是各个通道的均值mean，第二个元组是各个通道的标准差std
    ])

#定义模型架构
#输入的图片size是(m,)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5) #(32-5)+1=28  (6,28,28)
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) #14  (6,14,14)
        self.conv2=nn.Conv2d(6,16,5) #(14-5)/1+1=10  (16,10,10)

        self.fc1=nn.Linear(16*5*5,120,bias=True)
        self.fc2=nn.Linear(120,84,bias=True)
        self.fc3=nn.Linear(84,10,bias=True)
    def forward(self,input):
        x=self.pool(F.relu(self.conv1(input)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        #out=F.softmax(x,dim=1)
        return x
if __name__=='__main__':

    trainset=torchvision.datasets.CIFAR10(root='./CIFAR10_Data',train=True,transform=transfrom,download=True)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
    testset=torchvision.datasets.CIFAR10(root='./CIFAR10_Data',train=False,transform=transfrom,download=True)
    testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
    classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    # dataiter=iter(trainloader)
    # images,labels=dataiter.next()
    # print(images[0].size())
    # # print('-'*100)
    # # print(labels)
    # net=Net()
    # print(len(list(net.parameters())))
    # #定义损失函数和优化函数
    # criterion=nn.CrossEntropyLoss()
    # optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
    # #训练网络
    # epochs=2
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net.to(device)
    # for epoch in range(epochs):
    #     running_loss=0.0
    #     for i,data in enumerate(trainloader,0):
    #         inputs,labels=data
    #         inputs=inputs.to(device)
    #         labels=labels.to(device)
    #         #清零各参数的梯度
    #         optimizer.zero_grad()
    #         output=net(inputs)
    #         loss=criterion(output,labels)
    #         loss.backward()
    #         optimizer.step()
    #         #打印统计信息
    #         running_loss+=loss.item()
    #         if i%2000==1999:
    #             print('Epoch:{},step:{},loss:{: ^20.6f}'.format(epoch,i,running_loss/2000))
    #             running_loss=0.0
    # print('Finished Training')
    # PATH='./CIFAR10_SavedModel'
    # torch.save(net.state_dict(),PATH)
    net=Net()
    net.load_state_dict(torch.load('./CIFAR10_SavedModel'))
    testiter=iter(testloader)
    images,labels=next(testiter)
    out=net(images)
    _,out=torch.max(out,dim=1)
    print(out)
    print(labels)
    correct=0
    total=0
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            output=net(images)
            _,predicted=torch.max(output,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('准确率为{}'.format(correct/total))










