#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
#from torchvision.transforms import ToTensor
import torchvision
from torchvision.transforms import ToTensor
import uproot
import uproot3

import numpy as np
import matplotlib.pyplot as plt


# データセットの定義
class CustomRootDataset(Dataset):
    def __init__(self, root_file_path):
        self.root_file = uproot.open(root_file_path)
        self.tree = self.root_file["tree"]
        self.n_entries = len(self.tree[0].array())
        
    def __len__(self):
        return self.n_entries
    
    def __getitem__(self, idx):
        entry = self.tree
        
        reaction = entry["reaction"].array()[idx]
        p_Pi = entry["p_Pi"].array()[idx]
        pth_Pi = entry["pth_Pi"].array()[idx]
        pphi_Pi = entry["pphi_Pi"].array()[idx]
        p_DP = entry["p_DP"].array()[idx]
        pth_DP = entry["pth_DP"].array()[idx]
        pphi_DP = entry["pphi_DP"].array()[idx]
        p_DPi = entry["p_DPi"].array()[idx]
        pth_DPi = entry["pth_DPi"].array()[idx]
        pphi_DPi = entry["pphi_DPi"].array()[idx]
        p_SP = entry["p_SP"].array()[idx]
        pth_SP = entry["pth_SP"].array()[idx]
        pphi_SP = entry["pphi_SP"].array()[idx]
        mm_d = entry["mm_d"].array()[idx]
        theta = entry["theta"].array()[idx]
  
        
        # Convert to PyTorch tensors
        reaction = torch.tensor(reaction, dtype=torch.long)
        p_Pi = torch.tensor(p_Pi, dtype=torch.float32)
        pth_Pi = torch.tensor(pth_Pi, dtype=torch.float32)
        pphi_Pi = torch.tensor(pphi_Pi, dtype=torch.float32)
        p_DP = torch.tensor(p_DP, dtype=torch.float32)
        pth_DP = torch.tensor(pth_DP, dtype=torch.float32)
        pphi_DP = torch.tensor(pphi_DP, dtype=torch.float32)
        p_DPi = torch.tensor(p_DPi, dtype=torch.float32)
        pth_DPi = torch.tensor(pth_DPi, dtype=torch.float32)
        pphi_DPi = torch.tensor(pphi_DPi, dtype=torch.float32)
        p_SP = torch.tensor(p_SP, dtype=torch.float32)
        pth_SP = torch.tensor(pth_SP, dtype=torch.float32)
        pphi_SP = torch.tensor(pphi_SP, dtype=torch.float32)
        mm_d = torch.tensor(mm_d, dtype=torch.float32)
        theta = torch.tensor(theta, dtype=torch.float32)

        p_Pi = p_Pi.unsqueeze(0)
        pth_Pi = pth_Pi.unsqueeze(0)
        pphi_Pi = pphi_Pi.unsqueeze(0)
        p_DP = p_DP.unsqueeze(0)
        pth_DP = pth_DP.unsqueeze(0)
        pphi_DP = pphi_DP.unsqueeze(0)
        p_DPi = p_DPi.unsqueeze(0)
        pth_DPi = pth_DPi.unsqueeze(0)
        pphi_DPi = pphi_DPi.unsqueeze(0)
        p_SP = p_SP.unsqueeze(0)
        pth_SP = pth_SP.unsqueeze(0)
        pphi_SP = pphi_SP.unsqueeze(0)
        mm_d = mm_d.unsqueeze(0)
        theta = theta.unsqueeze(0)

        return {"input": torch.cat((p_Pi, pth_Pi, pphi_Pi,
                                    p_DP, pth_DP, pphi_DP,
                                    p_DPi, pth_DPi, pphi_DPi,
                                    p_SP, pth_SP, pphi_SP,
                                    mm_d), dim=0), "target": reaction}
#        return {"input": torch.cat((px_Pi, py_Pi, pz_Pi,
#                                    px_DP, py_DP, pz_DP,
#                                    px_DPi, py_DPi, pz_DPi,
#                                    px_SP, py_SP, pz_SP,
#                                    mm_d), dim=0), "target": reaction}

#        return {"input": torch.cat((p_Pi, 
#                                    p_SP,
#                                    mm_d), dim=0), "target": reaction}


            #pz_Pi,mm_d, theta
            #), dim=0), "target": reaction}


# 訓練用データセットとデータローダの設定
train_root_file_path = "create_rootfiles/train_reaction.root"
train_dataset = CustomRootDataset(train_root_file_path)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6)
#train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# テスト用データセットの設定
test_root_file_path = "create_rootfiles/test_reaction.root"
test_dataset = CustomRootDataset(test_root_file_path)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=6)
#test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)


# モデルの定義
class ExampleNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        return self.fc3(z2)
        #return F.softmax(self.fc3(z2))
        

# モデルの再定義と初期化
input_size = 13
#input_size = 3
hidden1_size = 1024
hidden2_size = 512
#hidden1_size = 64
#hidden2_size = 64
output_size = 3

device = 'cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ExampleNN(input_size, hidden1_size, hidden2_size, output_size).to(device)

print(model)
# 学習に使用するデバイスの設定

# 損失関数とオプティマイザの定義
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
#optimizer = optim.Adam(model.parameters(), lr=0.001)

''' training function '''
def train_model(model, train_loader, loss_function, optimizer, device='cpu'):
    train_loss = 0.0
    num_train = 0
    train_accuracy = 0.0
    model.train() # train mode
    for i, batch in enumerate(train_loader):
        num_train += len(batch["target"]) # count batch number
        inputs, labels = batch["input"].to(device), batch["target"].to(device)
        test1 = inputs[0] 
        test2 = inputs[1]
        reaction = batch["target"].cpu().numpy()
        optimizer.zero_grad() # initialize grad
        #1 forward
        outputs = model(inputs)
        #2 calculate loss
        loss = loss_function(outputs, labels)
        #3 calculate grad
        loss.backward()
        #4 update parameters
        optimizer.step()
        reaction_ML = torch.argmax(outputs, dim=1).cpu().numpy()
        train_loss += loss.item()
        #        print(f'num:{num_train}, loss:{loss.item()}')

        if i==0:
            print('train')
            print(f'reaction   : {reaction}')
            print(f'reaction_ML: {reaction_ML}')
            print(f'test1: {test1.cpu().numpy()}')
            #print(f'test2: {test2.cpu().numpy()}')
            
        for j in range(len(labels)):
            if reaction[j]==reaction_ML[j]:
                prediction=1
            else:
                prediction=0
            train_accuracy += prediction
    train_loss = train_loss / num_train
    train_accuracy = train_accuracy / num_train
    return train_loss, train_accuracy

''' test function '''
def test_model(model, test_loader, loss_function, optimizer, device='cpu'):
    test_loss = 0.0
    test_accuracy = 0.0
    num_test = 0
    model.eval() # eval mode
    with torch.no_grad(): # invalidate grad
        for i, batch in enumerate(test_loader):
            num_test += len(batch["target"])
            inputs, labels = batch["input"].to(device), batch["target"].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            reaction = batch["target"].cpu().numpy()
            reaction_ML = torch.argmax(outputs, dim=1).cpu().numpy()
            if i==0:
                print('test')
                print(f'reaction   : {reaction}')
                print(f'reaction_ML: {reaction_ML}')
            
            for j in range(len(labels)):
                if reaction[j]==reaction_ML[j]:
                    prediction=1
                else:
                    prediction=0
                test_accuracy += prediction
        test_loss = test_loss / num_test
        test_accuracy = test_accuracy / num_test
    return test_loss, test_accuracy

''' leaning function '''
def learning(model, train_loader, test_loader, loss_function,
             opimizer, n_epoch, device='cpu'):
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    num_epoch = []
    # epoch loop
    for epoch in range(1, n_epoch+1, 1):
        train_loss, train_accuracy = train_model(model, train_loader, loss_function, optimizer, device=device)
        test_loss, test_accuracy = test_model(model, test_loader, loss_function, optimizer, device=device)
        print(f'epoch : {epoch}, train_loss : {train_loss:.6f}, test_loss : {test_loss:.5f}')
        print(f'train_accuracy : {train_accuracy:.6f}, test_accuracy : {test_accuracy:.5f}')
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
    return train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list

''' learning '''
n_epoch = 500
#train_loss_list, test_loss_list = learning(model, train_loader, test_loader, loss_function, optimizer, n_epoch, device=device)
train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list = learning(model, train_loader, test_loader, loss_function, optimizer, n_epoch, device=device)

epoch_list = []
for i in range(n_epoch):
    epoch_list.append(i+1)

''' plot loss '''
#plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
#plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
#plt.xlabel("epoch")
#plt.ylabel("loss")
#plt.legend()
#plt.grid()
#plt.show()

#plt.plot(range(len(train_accuracy_list)), train_accuracy_list, c='b', label='train accuracy')
#plt.plot(range(len(test_accuracy_list)), test_accuracy_list, c='r', label='test accuracy')
#plt.xlabel("epoch")
#plt.ylabel("accuracy")
#plt.legend()
#plt.grid()
#plt.show()

torch.save(model.state_dict(), 'model_digits_rootfile_rthphi500.pth')

file = uproot3.recreate("create_rootfiles/output_digits_rthphi500.root")
file["tree"] = uproot3.newtree({"train_loss": np.float32,
                                "test_loss": np.float32,
                                "train_accuracy": np.float32,
                                "test_accuracy": np.float32,
                                "n_epoch": np.int32})

file["tree"].extend({"train_loss": train_loss_list,
                     "test_loss": test_loss_list,
                     "train_accuracy": train_accuracy_list,
                     "test_accuracy": test_accuracy_list,
                     "n_epoch": epoch_list})

