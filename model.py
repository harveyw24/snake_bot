import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='linear-model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='linear-model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)

        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()
            print ('Loading existing state dict.')
            return True
        
        print ('No existing state dict found. Starting from scratch.')
        return False

class CNN_QNet(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        # Input -> (w=640, h=480, n=3)
        # Pool  -> (w=32,  h=24,  n=3)
        self.setup = nn.MaxPool2d(kernel_size=20, stride=20)

        # Input -> (w=32, h=24, n=3)
        # Conv  -> (w=32, h=24, n=32)
        # Pool  -> (w=16, h=12, n=32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Input -> (w=16, h=12, n=32)
        # Conv  -> (w=16, h=12, n=64)
        # Pool  -> (w=8,  h=6,  n=64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Input   -> 8*6*64
        # Linear1 -> hidden_size
        # Linear2 -> output_size
        self.layer3 = nn.Sequential(
            nn.Linear(8*6*64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def save(self, file_name='cnn-model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='cnn-model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)

        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()
            print ('Loading existing state dict.')
            return True
        
        print ('No existing state dict found. Starting from scratch.')
        return False


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = Reward + gamma * max(next_predicted Q value)
        # pred.clone()
        # preds[argmax(action)] = Q_new
        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()
