import pandas as pd
import torch
from torch import nn
import torch.nn.functional as  F
import utils

#1. 读取数据###########################
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")
train_lable = train_data['Sold Price']
train_data = train_data.drop('Sold Price', axis=1)
#2.数据预处理###########################
#2.1删除所有数据的id特征,并将训练数据和试数据合并后统一处理(逗号前是行，逗号后是列)
all_features = pd.concat((train_data.iloc[:,1:], test_data.iloc[:,1:]))
print(all_features.shape)
#2.2特征值范围缩放（数据-均值）/方差，缺失值替换为均值，缩放后均为0
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index # 获取数字类型的值
#缩放数据
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x : (x - x.mean())/(x.std()))
#将缺失值填充为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
#summary列是大段文字叙述难以使用，state列都是ca没有用，删除这两个特征
all_features = all_features.drop(['Address', 'Summary','Heating', 'Cooling', 'Parking', 'Elementary School', 'Middle School',
                                  'High School', 'Flooring', 'Heating features', 'Cooling features',
                                  'Appliances included','Laundry features', 'Parking features',
                                  'City','State','Listed On', 'Last Sold On','Region'],axis=1)
#2.3用独热编码处理离散值
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
#2.4将数据转化成张量格式
n_train = train_data.shape[0]
device = torch.device("cuda:0")
train_data = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
test_data = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)
train_lable = torch.tensor(train_lable, dtype = torch.float32).reshape(-1,1).to(device)
#用5000个数据来验证
valid_data = train_data[:5000]
valid_lable = train_lable[:5000]
train_data = train_data[5000:]
train_lable = train_lable[5000:]

input_num = train_data.shape[1]
#3.构建模型###############
class Net(nn.Module):
    def __init__(self, input_num):
        super().__init__()
        self.hidden1 = nn.Linear(input_num, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.hidden4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
    
    def forward(self, input):
        x = F.relu(self.hidden1(input))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        y = F.relu(self.out(x))
        return y
#4. 训练模型


#超参数      
batch_size = 512
epochs = 1000
lr = 0.0001
#权重衰减参数
weight_decay = 0.0001
class Module():
    def __init__(self, input_num):
        #初始化模型的网络，损失函数，优化器
        self.net = Net(input_num).to(device)
        self.loss = nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                lr = lr, weight_decay=weight_decay)
    #模型训练函数
    def train(self, data_iter):
        for X, y in data_iter:
            self.optimizer.zero_grad()
            y_hat = self.net(X)
            #训练误差数值太大，将其转化为对数形式
            loss = self.loss(torch.log(y_hat), torch.log(y))
            loss.backward()
            self.optimizer.step()
        return loss.detach().cpu().numpy()
        
#构建模型
module = Module(input_num)
#记录训练过程的损失值
train_losses = []
valid_losses = []
#开始训练epochs次
for i in range(epochs):
    #将数据构建为迭代器
    train_iter = utils.data_iter(train_data, train_lable, batch_size)
    #传入数据迭代器，训练模型，得到训练误差
    train_loss = module.train(train_iter)
    #在验证数据集上运行一次模型，得到验证误差
    y_hat = module.net(valid_data)
    valid_loss = module.loss(torch.log(y_hat), torch.log(valid_lable)).detach().cpu().numpy()
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f"epoch: {i} ,train loss: {train_loss}, valid_loss: {valid_loss}")

#预测结果，并将结果转换为要求的格式
result = module.net(test_data).reshape(-1,1).cpu().detach().numpy()
submission = pd.DataFrame()
submission['Id'] = range(47439, len(test_data)+47439)
submission['Sold Price'] = result
submission.to_csv("submission.csv",index=None)
utils.draw_losses(train_losses[400:], valid_losses[400:])
