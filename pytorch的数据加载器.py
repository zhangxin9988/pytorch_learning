#模型数据的预处理
import numpy as np
import matplotlib.pyplot as plt
from scipy import io as spio
from scipy import linalg
from PIL import Image
# 奇异值分解进行图片的压缩
# A=plt.imread(r'C:\Users\zx\Pictures\GrayScale.jpg')
# # plt.imshow(img)
# # plt.show()
# U,s,Vh=np.linalg.svd(A)
# print(U.shape)
# print(s.shape)
# print(Vh.shape)
# k=50
# M,N=A.shape
# U=U[:,:k]
# Sig=linalg.diagsvd(s[:k],k,k)
# Vh=Vh[:k,:]
# img=U.dot(Sig.dot(Vh))
# print(img.shape)
# plt.imshow(img)
# plt.show()
# image=Image.open(r'C:\Users\zx\Pictures\压缩.jpg')
# image=image.convert('L')
# image.show()
# image.save(r'C:\Users\zx\Pictures\aaa.jpg')


#PCA主成分分析
image=plt.imread(r'C:\Users\zx\Pictures\GrayScale.jpg').T
print(image.shape)  #shape=(498维特征, 536个样例)
# plt.imshow(image)
# plt.show()
#第一步，零均值化

mean=np.mean(image,axis=1)
A=image-mean.reshape(-1,1)

M,N=A.shape  #shape=(498, 536)
#找出协方差矩阵
c=np.dot(A,np.transpose(A))/536

eigval,eigvec=np.linalg.eig(c)

# print(eigval.shape) #特征值(536,)
# print(eigvec.shape) #特征向量(536, 536)
#取前k个特征向量
k=50
vecs=eigvec[:,:k].T  #特征向量(k,536)
img_new=np.dot(vecs,image)
img=np.dot(eigvec[:,:k],img_new)

plt.imshow(img)
plt.show()

plt.imsave(r'C:\Users\zx\Pictures\asd.jpg',img)











# import numpy as np
# from numpy.linalg import svd
# A = np.array([[126,  52,  -3, -69],
#                [ 52, 292, -73, -80],
#                [ -3, -73, 141, -31],
#                [-69, -80, -31,  78],
#                [-69, -80, -31,  178]])
# def pca(X, n_components):
#     # 第一步：X减去均值
#     mean = np.mean(X, axis=1)
#     normX = X - mean.reshape(-1, 1)
#
#     # 第二步：对协方差矩阵XXT做特征值分解，得到特征值和对应特征向量
#     cov_mat = np.dot(normX, np.transpose(normX))
#
#     vals, vecs = np.linalg.eig(cov_mat)
#
#     # 第三步：按照特征值降序排序，取得对应的特征向量拼成投影矩阵WT
#
#     WT =vecs[:,n_components].T
#
#     # 第四步：对X做转换
#     data = np.dot(WT, normX)
#     return data
#
# n_components = 1
# X = plt.imread(r'C:\Users\zx\Pictures\GrayScale.jpg').T
# data = pca(X, n_components=n_components)
# print(data)
