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
        #return {"input": torch.cat((px_Pi, py_Pi, pz_Pi,
        #                            px_DP, py_DP, pz_DP,
        #                            px_DPi, py_DPi, pz_DPi,
        #                            px_SP, py_SP, pz_SP,
        #                            mm_d), dim=0), "target": reaction}



# 訓練用データセットとデータローダの設定
train_root_file_path = "create_rootfiles/train_reaction.root"
train_dataset = CustomRootDataset(train_root_file_path)
#train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6)


# テスト用データセットの設定
#test_root_file_path = "create_rootfiles/test_reaction.root"
test_root_file_path = "create_rootfiles/testHigh_reaction.root"
test_dataset = CustomRootDataset(test_root_file_path)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=6)
#test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=6)


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
        
# モデルの再定義と初期化
input_size = 13
hidden1_size = 1024
hidden2_size = 512
#hidden1_size = 64
#hidden2_size = 64
output_size = 3

device = 'cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ExampleNN(input_size, hidden1_size, hidden2_size, output_size).to(device)

''' load model '''
#model.load_state_dict(torch.load('model_digits_rootfile.pth'))
model.load_state_dict(torch.load('model_digits_rootfile_rthphi500.pth'))
loss_function = nn.CrossEntropyLoss()

test_loss = 0.0
num_test = 0

write_p_Pi = []
write_pth_Pi = []
write_pphi_Pi = []
write_p_DP = []
write_pth_DP = []
write_pphi_DP = []
write_p_DPi = []
write_pth_DPi = []
write_pphi_DPi = []
write_p_SP = []
write_pth_SP = []
write_pphi_SP = []
write_mm_d = []
write_reaction = []
write_reaction_ML = []
write_prediction = []
test_accuracy = 0

for i, batch in enumerate(test_loader):
    num_test += len(batch["target"])
    inputs, labels = batch["input"].to(device), batch["target"].to(device)
    reaction_ = labels.cpu().numpy()
    outputs = model(inputs)
    reaction_ML = torch.argmax(outputs, dim=1).cpu().numpy()
    for j in range(len(labels)):
        p_Pi = inputs[j, 0]
        write_p_Pi.append(p_Pi.cpu().numpy())
        pth_Pi = inputs[j, 1]
        write_pth_Pi.append(pth_Pi.cpu().numpy())
        pphi_Pi = inputs[j, 2]
        write_pphi_Pi.append(pphi_Pi.cpu().numpy())
        p_DP = inputs[j, 3]
        write_p_DP.append(p_DP.cpu().numpy())
        pth_DP = inputs[j, 4]
        write_pth_DP.append(pth_DP.cpu().numpy())
        pphi_DP = inputs[j, 5]
        write_pphi_DP.append(pphi_DP.cpu().numpy())
        p_DPi = inputs[j, 6]
        write_p_DPi.append(p_DPi.cpu().numpy())
        pth_DPi = inputs[j, 7]
        write_pth_DPi.append(pth_DPi.cpu().numpy())
        pphi_DPi = inputs[j, 8]
        write_pphi_DPi.append(pphi_DPi.cpu().numpy())
        p_SP = inputs[j, 9]
        write_p_SP.append(p_SP.cpu().numpy())
        pth_SP = inputs[j, 10]
        write_pth_SP.append(pth_SP.cpu().numpy())
        pphi_SP = inputs[j, 11]
        write_pphi_SP.append(pphi_SP.cpu().numpy())
        mm_d = inputs[j, 12]
        write_mm_d.append(mm_d.cpu().numpy())

        reaction = reaction_[j]
        write_reaction.append(reaction)
        write_reaction_ML.append(reaction_ML[j])
        if reaction==reaction_ML[j]:
            prediction=1
        else:
            prediction=0
        test_accuracy += prediction
        
    loss = loss_function(outputs, labels)
    test_loss += loss.item()
test_loss = test_loss / num_test
test_accuracy = test_accuracy / num_test

file = uproot3.recreate("create_rootfiles/test_reaction_rthphi_500.root")

file["tree"] = uproot3.newtree({"p_Pi": np.float32,
                                "pth_Pi": np.float32,
                                "pphi_Pi": np.float32,
                                "p_DP": np.float32,
                                "pth_DP": np.float32,
                                "pphi_DP": np.float32,
                                "p_DPi": np.float32,
                                "pth_DPi": np.float32,
                                "pphi_DPi": np.float32,
                                "p_SP": np.float32,
                                "pth_SP": np.float32,
                                "pphi_SP": np.float32,
                                "mm_d": np.float32,
                                "reaction": np.int32,
                                "reaction_ML": np.int32})

file["tree"].extend({"p_Pi": write_p_Pi,
                     "pth_Pi": write_pth_Pi,
                     "pphi_Pi": write_pphi_Pi,
                     "p_DP": write_p_DP,
                     "pth_DP": write_pth_DP,
                     "pphi_DP": write_pphi_DP,
                     "p_DPi": write_p_DPi,
                     "pth_DPi": write_pth_DPi,
                     "pphi_DPi": write_pphi_DPi,
                     "p_SP": write_p_SP,
                     "pth_SP": write_pth_SP,
                     "pphi_SP": write_pphi_SP,
                     "mm_d": write_mm_d,
                     "reaction": write_reaction,
                     "reaction_ML": write_reaction_ML})
    
 
