import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)

        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)

        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, output_dim)

    def forward(self, state):
        y1 = self.relu(self.fc1(state))
        y2 = self.relu(self.fc2(y1))
        value = self.relu(self.fc_value(y2))
        adv = self.relu(self.fc_adv(y2))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage
        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()