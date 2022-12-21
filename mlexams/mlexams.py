# model libraries
from operator import mod
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# nn model libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda, Compose
# gpu libraries
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.svm import SVC as cuSVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Union, List


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
          out_data = self.transform(self.data)[0][idx]
          out_label = self.label[idx]
        else:
          out_data = self.data[idx]
          out_label =  self.label[idx]

        return out_data, out_label

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.batchnorm1 = nn.BatchNorm1d(hid_dim)
        self.dropout1 = nn.Dropout(0.4)

        self.linear2 = nn.Linear(hid_dim, hid_dim // 2)
        self.batchnorm2 = nn.BatchNorm1d(hid_dim // 2)
        self.dropout2 = nn.Dropout(0.4)
        self.linear3 = nn.Linear(hid_dim // 2, hid_dim // 4)
        self.batchnorm3 = nn.BatchNorm1d(hid_dim // 4)
        self.dropout3 = nn.Dropout(0.4)
        self.linear4 = nn.Linear(hid_dim // 4, hid_dim // 8)
        self.batchnorm4 = nn.BatchNorm1d(hid_dim // 8)
        self.dropout4 = nn.Dropout(0.4)
        self.linear5 = nn.Linear(hid_dim // 8, hid_dim // 16)
        self.batchnorm5 = nn.BatchNorm1d(hid_dim // 16)
        self.dropout5 = nn.Dropout(0.4)
        '''
        self.dropout1 = nn.Dropout(0.4)
        self.linear2 = nn.Linear(hid_dim, int(hid_dim / 2))
        self.dropout2 = nn.Dropout(0.4)'''
        # self.linear3 = nn.Linear(hid_dim, out_dim)
        # self.linear3 = nn.Linear(hid_dim // 2, out_dim)
        self.linear6 = nn.Linear(hid_dim // 16, out_dim)

    def forward(self, x):
        # x = F.relu(self.linear1(x))
        x = self.linear1(x)
        x = F.relu(self.batchnorm1(x))
        x = self.dropout1(x)

        x = self.linear2(x)
        x = F.relu(self.batchnorm2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        x = F.relu(self.batchnorm3(x))
        x = self.dropout3(x)
        x = self.linear4(x)
        x = F.relu(self.batchnorm4(x))
        x = self.dropout4(x)
        x = self.linear5(x)
        x = F.relu(self.batchnorm5(x))
        x = self.dropout5(x)
        '''
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)'''
        # x = F.softmax(self.linear3(x), dim=1)
        x = F.softmax(self.linear6(x), dim=1)
        return x


def get_accuracy(
  train_data: any,
  train_label: any,
  test_data: any,
  test_label: any,
  label_kind: int,
  model_kind: str,
  options: Dict[str, Union[int, str]] = {
    "C": 5, "kernel": 'rbf', "gamma": 'auto'
  },
  on_gpu: bool = False,
) -> float:


    if model_kind == "svc":
        # learning
        if not on_gpu:
            # clf = SVC(C=1, kernel='rbf', gamma='auto')
            clf = SVC(**options)
        else:
            train_data = cp.asarray(train_data)
            train_label = cp.asarray(train_label)
            test_data = cp.asarray(test_data)
            test_label = cp.asarray(test_label)

            clf = cuSVC(**options)

        clf.fit(train_data, train_label)

        p = clf.predict(test_data)
        accuracy = accuracy_score(
            test_label,
            p
        )

    elif model_kind == "krr":
        # one-hot coding
        label_kind = label_kind
        train_label = (
            (
                -1 * np.ones((label_kind, label_kind), dtype=int)
            ) + (
                2 * np.eye(label_kind)
            )
        )[train_label]
        test_label = (
            (
                -1 * np.ones((label_kind, label_kind), dtype=int)
            ) + (
                2 * np.eye(label_kind)
            )
        )[test_label]

        # learning
        clf = KernelRidge(**options)
        clf.fit(train_data, train_label)

        p = clf.predict(test_data)
        accuracy = accuracy_score(
            np.argmax(test_label, axis=1), 
            np.argmax(p, axis=1)
        )
    
    elif model_kind == "logistic":
        
        # learning
        clf = LogisticRegression(**options)
        clf.fit(train_data, train_label)

        p = clf.predict(test_data)
        accuracy = accuracy_score(
            test_label,
            p
        )
    
    elif model_kind == "rf":
        
        # learning
        if not on_gpu:
            clf = RandomForestClassifier(**options)
        else:
            train_data = cp.asarray(train_data)
            train_label = cp.asarray(train_label)
            test_data = cp.asarray(test_data)
            test_label = cp.asarray(test_label)

            clf = cuRFC(**options)
            
        clf.fit(train_data, train_label)

        p = clf.predict(test_data)
        accuracy = accuracy_score(
            test_label,
            p
        )
    
    elif model_kind == "mlp":
        
        # Dataset
        # train_data, valid_data, train_label, valid_label = train_test_split(
        #     train_data, train_label, test_size=0.125, random_state=0
        # )
        dataset_train = MyDataset(train_data, train_label, transform=Compose([ToTensor()]))
        # dataset_valid = MyDataset(valid_data, valid_label, transform=Compose([ToTensor()]))
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=int(len(train_label) / 10), shuffle=False)
        # valid_dataloader = torch.utils.data.DataLoader(dataset_valid, batch_size=int(len(valid_label) / 10), shuffle=False)
        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(**options).to(device)  # 500, 20
        
        # learning
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        n_epochs = 400
        best_train_loss = -1
        # best_valid_loss = -1
        limit_patient = 40
        patient = 0
        for epoch in range(n_epochs):

            model.train()  # 訓練時には勾配を計算するtrainモードにする
            train_loss_list = []
            for x, t in train_dataloader:

                # 勾配の初期化
                model.zero_grad()

                # テンソルをGPUに移動
                x = x.float().to(device)
                t = t.to(device)

                # 順伝播
                y = model.forward(x)
                # 誤差の計算(クロスエントロピー誤差関数)
                loss = criterion(y, t)
                train_loss_list.append(loss.item())

                # 誤差の逆伝播
                optimizer.zero_grad()
                loss.backward()

                # パラメータの更新
                optimizer.step()

                ''' # モデルの出力を予測値のスカラーに変換
                pred = y.argmax(1)

                losses_train.append(loss.tolist())

                acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
                train_num += acc.size()[0]
                train_true_num += acc.sum().item()'''

            '''
            model.eval()  # 評価時には勾配を計算しないevalモードにする
            valid_loss_list = []
            for x, t in valid_dataloader:

                # テンソルをGPUに移動
                x = x.float().to(device)
                t = t.to(device)

                # 順伝播
                y = model.forward(x)

                # 誤差の計算(クロスエントロピー誤差関数)
                loss = criterion(y, t)
                valid_loss_list.append(loss.item())'''
            
            mean_loss = np.mean(train_loss_list)
            # mean_loss = np.mean(valid_loss_list)
            # if best_valid_loss == -1 or best_valid_loss > mean_loss:
            #     best_valid_loss = mean_loss
            #     patient = 0
            if best_train_loss == -1 or best_train_loss > mean_loss:
                best_train_loss = mean_loss
                patient = 0
            else:
                patient += 1
            if patient >= limit_patient:
                break
        model.eval()  # 評価時には勾配を計算しないevalモードにする
        # p = clf.predict(test_data)
        outputs = model(torch.tensor(test_data, dtype=torch.float).to(device))
        _, p = torch.max(outputs.data, dim=1)
        p = p.to('cpu')
        accuracy = accuracy_score(
            test_label,
            p
        )
    
    else:
        raise KeyError("input model kind has not been matched.")

    return accuracy

# def grid_search(
#   train_data: any,
#   train_label: any,
#   label_kind: int,
#   model_kind: str,
#   options: Dict[str, Union[int, str]] = {
#     "C": 5, "kernel": 'rbf', "gamma": 'auto'
#   }
# ) -> float:


#     if model_kind == "rf":
#         # learning
#         # clf = SVC(C=1, kernel='rbf', gamma='auto')
#         clf = SVC(**options)
#         clf.fit(train_data, train_label)

#         p = clf.predict(test_data)
#         accuracy = accuracy_score(
#             test_label,
#             p
#         )
    
#     else:
#         raise KeyError("input model kind has not been matched.")

#     return accuracy