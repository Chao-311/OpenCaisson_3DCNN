import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import r2_score  #R square
from sklearn.metrics import mean_squared_error #均方误差


df_total=pd.read_csv(r'D:\TotalSamples.csv')
x_total=torch.tensor(df_total.iloc[:,6:131].values).double()
y_total=torch.tensor(df_total.iloc[:,1].values).double()

df_train=pd.read_csv(r'D:\TrainSamples.csv')
x_train=torch.tensor(df_train.iloc[:,6:131].values).double()
y_train=torch.tensor(df_train.iloc[:,1].values).double()

df_test=pd.read_csv(r'D:\TestSamples.csv')
x_test=torch.tensor(df_test.iloc[:,6:131].values).double()
y_test=torch.tensor(df_test.iloc[:,1].values).double()


def MSE(predicted,real):   #计算平均相对误差MRE  见陈祖煜论文
    MSE=mean_squared_error(real,predicted)
    return MSE

def R_2(predicted,real):  # 计算拟合优度  见语雀笔记有
    R_2=r2_score(real,predicted)
    return R_2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv3d(1,16,(3,3,3),stride=1,padding=1)
        self.conv2=nn.Conv3d(16,32,(3,3,3),stride=1,padding=1)
        self.fc1 = nn.Linear(32*5*5*5, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 32*5*5*5)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

model=Net()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=0.005)


x_train1=x_train.reshape(-1,5,5,5).unsqueeze(1)
x_total1=x_total.reshape(-1,5,5,5).unsqueeze(1)
x_test1=x_test.reshape(-1,5,5,5).unsqueeze(1)


num_epochs=1000
for epoch in range(num_epochs):
    out=model(x_train1)
    out=out.squeeze(-1)
    out_test=model(x_test1)
    out_test = out_test.squeeze(-1)
    loss=criterion(out,y_train)
    loss_test=criterion(out_test,y_test)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1 ==0:
        print('Epoch[{}/{}],loss_train:{:.6f},loss_test:{:.6f}'.format(epoch+1,num_epochs,loss.detach().numpy(),loss_test.detach().numpy()))


y_hat_total=model(x_total1)
MSE_Total=MSE(y_hat_total.squeeze(-1).data.numpy(),y_total.squeeze(-1).data.numpy())
R_Total=R_2(y_hat_total.squeeze(-1).data.numpy(),y_total.squeeze(-1).data.numpy())
print('Total,  MSE={:.6f},  $R^2$={:.6f}'.format(MSE_Total,R_Total))

y_hat_train=model(x_train1)
MSE_Train=MSE(y_hat_train.squeeze(-1).data.numpy(),y_train.squeeze(-1).data.numpy())
R_Train=R_2(y_hat_train.squeeze(-1).data.numpy(),y_train.squeeze(-1).data.numpy())
print('Train,  MSE={:.6f},  $R^2$={:.6f}'.format(MSE_Train,R_Train))

y_hat_test=model(x_test1)
MSE_Test = MSE(y_hat_test.squeeze(-1).data.numpy(), y_test.squeeze(-1).data.numpy())
R_Test = R_2(y_hat_test.squeeze(-1).data.numpy(), y_test.squeeze(-1).data.numpy())

print('Test,  MSE={:.6f},  $R^2$={:.6f}'.format(MSE_Test,R_Test))
print(y_hat_test.squeeze(-1).data.numpy())
print(y_test.squeeze(-1).data.numpy())