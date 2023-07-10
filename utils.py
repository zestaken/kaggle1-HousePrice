import random
from matplotlib import pyplot as plt


#构建一个数据小批量迭代器
def data_iter(data, label, batch_size):
    nums = len(data)
    #创建索引
    indices = list(range(nums))
    #打乱索引顺序，以随机读取样本
    random.shuffle(indices)
    #从0到nums取数，步幅为batch_size
    for i in range(0, nums, batch_size):
        batch_indices = indices[i:min(i+batch_size, nums)]
        #生成迭代器
        yield data[batch_indices], label[batch_indices]

# 绘制损失图像
def draw_losses(train_losses, valid_losses):
    # x轴取值是epochs的次数
    x = range(1,len(train_losses)+1)
    plt.title("training losses") #设置图像名
    plt.xlabel('epochs') #设置x轴的名字
    plt.ylabel('loss') #设置y轴名字
    plt.plot(x,train_losses, color='r') #绘制损失曲线一，设置为红色
    plt.plot(x,valid_losses, color='b', linestyle='dashed') #绘制损失曲线二，设置为蓝色，虚线
    plt.legend(['train loss', 'valid loss']) #设置两条曲线的名字
    plt.show() #绘制图像