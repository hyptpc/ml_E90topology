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
        px_Pi = entry["px_Pi"].array()[idx]
        py_Pi = entry["py_Pi"].array()[idx]
        pz_Pi = entry["pz_Pi"].array()[idx]
        px_DP = entry["px_DP"].array()[idx]
        py_DP = entry["py_DP"].array()[idx]
        pz_DP = entry["pz_DP"].array()[idx]
        px_DPi = entry["px_DPi"].array()[idx]
        py_DPi = entry["py_DPi"].array()[idx]
        pz_DPi = entry["pz_DPi"].array()[idx]
        px_SP = entry["px_SP"].array()[idx]
        py_SP = entry["py_SP"].array()[idx]
        pz_SP = entry["pz_SP"].array()[idx]
        mm_d = entry["mm_d"].array()[idx]
        theta = entry["theta"].array()[idx]
  
        
        # Convert to PyTorch tensors
        reaction = torch.tensor(reaction, dtype=torch.long)
        px_Pi = torch.tensor(px_Pi, dtype=torch.float32)
        py_Pi = torch.tensor(py_Pi, dtype=torch.float32)
        pz_Pi = torch.tensor(pz_Pi, dtype=torch.float32)
        px_DP = torch.tensor(px_DP, dtype=torch.float32)
        py_DP = torch.tensor(py_DP, dtype=torch.float32)
        pz_DP = torch.tensor(pz_DP, dtype=torch.float32)
        px_DPi = torch.tensor(px_DPi, dtype=torch.float32)
        py_DPi = torch.tensor(py_DPi, dtype=torch.float32)
        pz_DPi = torch.tensor(pz_DPi, dtype=torch.float32)
        px_SP = torch.tensor(px_SP, dtype=torch.float32)
        py_SP = torch.tensor(py_SP, dtype=torch.float32)
        pz_SP = torch.tensor(pz_SP, dtype=torch.float32)
        mm_d = torch.tensor(mm_d, dtype=torch.float32)
        theta = torch.tensor(theta, dtype=torch.float32)

        px_Pi = px_Pi.unsqueeze(0)
        py_Pi = py_Pi.unsqueeze(0)
        pz_Pi = pz_Pi.unsqueeze(0)
        px_DP = px_DP.unsqueeze(0)
        py_DP = py_DP.unsqueeze(0)
        pz_DP = pz_DP.unsqueeze(0)
        px_DPi = px_DPi.unsqueeze(0)
        py_DPi = py_DPi.unsqueeze(0)
        pz_DPi = pz_DPi.unsqueeze(0)
        px_SP = px_SP.unsqueeze(0)
        py_SP = py_SP.unsqueeze(0)
        pz_SP = pz_SP.unsqueeze(0)
        mm_d = mm_d.unsqueeze(0)
        theta = theta.unsqueeze(0)

        #        return {"input": torch.cat((p_Pi, pth_Pi, pphi_Pi,
        #                                    p_DP, pth_DP, pphi_DP,
        #                                    p_DPi, pth_DPi, pphi_DPi,
        #                                    p_SP, pth_SP, pphi_SP,
        #                                    mm_d), dim=0), "target": reaction}
        return {"input": torch.cat((px_Pi, py_Pi, pz_Pi,
                                    px_DP, py_DP, pz_DP,
                                    px_DPi, py_DPi, pz_DPi,
                                    px_SP, py_SP, pz_SP,
                                    mm_d), dim=0), "target": reaction}



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
#model.load_state_dict(torch.load('model_digits_rootfile300_64_64.pth'))
model.load_state_dict(torch.load('model_digits_rootfile500.pth'))
loss_function = nn.CrossEntropyLoss()

test_loss = 0.0
num_test = 0

write_px_Pi = []
write_py_Pi = []
write_pz_Pi = []
write_px_DP = []
write_py_DP = []
write_pz_DP = []
write_px_DPi = []
write_py_DPi = []
write_pz_DPi = []
write_px_SP = []
write_py_SP = []
write_pz_SP = []
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
        px_Pi = inputs[j, 0]
        write_px_Pi.append(px_Pi.cpu().numpy())
        py_Pi = inputs[j, 1]
        write_py_Pi.append(py_Pi.cpu().numpy())
        pz_Pi = inputs[j, 2]
        write_pz_Pi.append(pz_Pi.cpu().numpy())
        px_DP = inputs[j, 3]
        write_px_DP.append(px_DP.cpu().numpy())
        py_DP = inputs[j, 4]
        write_py_DP.append(py_DP.cpu().numpy())
        pz_DP = inputs[j, 5]
        write_pz_DP.append(pz_DP.cpu().numpy())
        px_DPi = inputs[j, 6]
        write_px_DPi.append(px_DPi.cpu().numpy())
        py_DPi = inputs[j, 7]
        write_py_DPi.append(py_DPi.cpu().numpy())
        pz_DPi = inputs[j, 8]
        write_pz_DPi.append(pz_DPi.cpu().numpy())
        px_SP = inputs[j, 9]
        write_px_SP.append(px_SP.cpu().numpy())
        py_SP = inputs[j, 10]
        write_py_SP.append(py_SP.cpu().numpy())
        pz_SP = inputs[j, 11]
        write_pz_SP.append(pz_SP.cpu().numpy())
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

file = uproot3.recreate("create_rootfiles/test_reaction_ML_500.root")

file["tree"] = uproot3.newtree({"px_Pi": np.float32,
                                "py_Pi": np.float32,
                                "pz_Pi": np.float32,
                                "px_DP": np.float32,
                                "py_DP": np.float32,
                                "pz_DP": np.float32,
                                "px_DPi": np.float32,
                                "py_DPi": np.float32,
                                "pz_DPi": np.float32,
                                "px_SP": np.float32,
                                "py_SP": np.float32,
                                "pz_SP": np.float32,
                                "mm_d": np.float32,
                                "reaction": np.int32,
                                "reaction_ML": np.int32})

file["tree"].extend({"px_Pi": write_px_Pi,
                     "py_Pi": write_py_Pi,
                     "pz_Pi": write_pz_Pi,
                     "px_DP": write_px_DP,
                     "py_DP": write_py_DP,
                     "pz_DP": write_pz_DP,
                     "px_DPi": write_px_DPi,
                     "py_DPi": write_py_DPi,
                     "pz_DPi": write_pz_DPi,
                     "px_SP": write_px_SP,
                     "py_SP": write_py_SP,
                     "pz_SP": write_pz_SP,
                     "mm_d": write_mm_d,
                     "reaction": write_reaction,
                     "reaction_ML": write_reaction_ML})
    
 
