import numpy as np
import torch as th
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False
x=np.arange(0,20,1)
y=np.linspace(0,10,20)
plt.subplot(121)
plt.plot(x,y,'r--*')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xticks(x)
plt.yticks(y)
plt.legend(['你好'])
grids=np.meshgrid(x,y)
plt.grid(grids)
plt.subplot(122)
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
ax=plt.gca()
ax.set_xlabel('你他妈的傻逼')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴

ax.spines['bottom'].set_position(('data', 20))  # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上

ax.spines['left'].set_position(('data', 20))

print(ax)
plt.show()