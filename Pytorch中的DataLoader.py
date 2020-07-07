import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision
#定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d((2,2),(2,2))
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,input):
        x=self.pool(F.relu(self.conv1(input)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
if __name__ =='__main__':
    net=Net()
    print(net)
    #加载数据
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    trainset=torchvision.datasets.CIFAR10('./CIFAR10_Data',train=True,transform=transform,download=True)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=16,shuffle=True,num_workers=2)
    testset=torchvision.datasets.CIFAR10('./CIFAR10_Data',train=False,transform=transform,download=True)
    testloader=torch.utils.data.DataLoader(testset,batch_size=16,shuffle=False,num_workers=2)
    classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    device=torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    #模型的训练
    lr=0.001
    epochs=3
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(),lr)
    for epoch in range(epochs):
        running_loss=0.0
        for i,data in enumerate(trainloader,0):
            images,labels=data
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            output=net(images)
            loss=criterion(output,labels)
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            if(i%500==499):
                print('Epoch:{:<3}  |  step:{:<5}  |  loss:{:<15.6f}\n'.format(epoch+1,i+1,running_loss/500))

                running_loss=0.0
    PATH=r'./CIFAR10_SaveModel.pth'
    torch.save(net.state_dict(),PATH)
    print('Finished Training')
    net=Net()
    net.load_state_dict(torch.load(r'./CIFAR10_SaveModel.pth'))
    net.to(device)
    #模型的测试
    correct=0
    total=0
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            images = images.to(device)
            labels = labels.to(device)
            output=net(images)
            _,predicted=torch.max(output,dim=1)
            total+=labels.shape[0]
            correct+=(predicted==labels).sum().item()
        accuracy=correct/total
        print('Accuracy on testset:{}'.format(accuracy))

# Finished Training
# Accuracy on testset:0.5835 十分类准确率58%，已经说明模型在起作用了