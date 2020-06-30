import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transfroms
import matplotlib.pyplot as plt

if __name__ == '__main__':
    transform = transfroms.Compose([
        transfroms.ToTensor(),
        transfroms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # trainset=torchvision.datasets.CIFAR10(root='./CIFAR10_Data',train=True,transform=transform,download=False)
    # trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10_Data', train=False, transform=transform, download=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=1)
    testiter = iter(testloader)
    images, labels = next(testiter)


    def img_transpose(img):
        img = img / 2 + 0.5
        img = img.numpy().transpose(1, 2, 0)
        return img
    fig,axes=plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            image=img_transpose(images[4*i+j])
            label=labels[4*i+j]
            ax=axes[i][j]

            ax.set_xticks([])
            ax.set_yticks([])

            ax.imshow(image)
    plt.show()

