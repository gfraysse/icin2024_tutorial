import os
from network_architecture.QNetwork import QNetwork
from experience_replay.replay_buffer import PrioritizedReplayBuffer
from datetime import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

class D3QNAgent:
    def __init__(self, num_metrics, num_actions, device, optimizer="Adam", initial_lr=1e-5, per_alpha=0.6, buffer_size=50000, buffer_path="backups/replay_buffer", buffer_name='d3qn_buffer.pickle', load_buffer=False, seed=None):
        self.num_metrics = num_metrics
        self.num_actions = num_actions
        self.device = device
        self.memory_replay = PrioritizedReplayBuffer(buffer_size=buffer_size, buffer_path=buffer_path, buffer_name=buffer_name, load_buffer=load_buffer, alpha=per_alpha, seed=seed)

        self.onlineQNetwork = QNetwork(input_dim=num_metrics, output_dim=num_actions).apply(D3QNAgent.init_weights).to(self.device)
        self.targetQNetwork = QNetwork(input_dim=num_metrics, output_dim=num_actions).apply(D3QNAgent.init_weights).to(self.device)
        
        self.targetQNetwork.load_state_dict(copy.deepcopy(self.onlineQNetwork.state_dict()))
        
        self.optimizer = D3QNAgent.get_optim(model_params=self.onlineQNetwork.parameters(),
                                    optimizer=optimizer,
                                    initial_lr=initial_lr)

        random.seed(seed)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_optim(model_params, optimizer, initial_lr, momentum=0, weight_decay=0, betas=(0.9, 0.999)):
        assert isinstance(optimizer, str), "Optimizer keyword (name) must be STR type"
        optimizer = optimizer.strip().lower()
        if optimizer == "sgd":
            return optim.SGD(model_params, lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == "adam":
            return optim.Adam(model_params, lr=initial_lr, weight_decay=weight_decay, betas=betas)
        else:
            raise NotImplementedError(
                "Optimizer name given does not match implemented optimizers, kindly check with the authors.")

    def resume_training(self, model_info_dict):
        self.onlineQNetwork.load_state_dict(model_info_dict['online_model'])
        self.targetQNetwork.load_state_dict(model_info_dict['target_model'])

        self.optimizer.load_state_dict(model_info_dict['optimizer_state_dict'])


    def get_action(self, state, epsilon, seed=None):
    
        p = random.random()   #0.0 0.1
        
        if p < epsilon:
            
            action = random.randint(0, self.num_actions - 1)
        else:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.onlineQNetwork.select_action(tensor_state)
        
        return action

    def forward(self, beta, batch, gamma):
        batch, self.batch_indices,  weights = self.memory_replay.sample(batch, beta)
                    

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)
        
        self.batch_state = torch.FloatTensor(batch_state).to(self.device)
        self.batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
        self.batch_weights = torch.FloatTensor(weights).to(self.device)       

        batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)
        
                    
        self.onlineQ_next = self.onlineQNetwork(self.batch_state).gather(1, batch_action.long()).squeeze(-1) 

        with torch.no_grad():
            targetQ_next = self.targetQNetwork(self.batch_next_state)
            onlineQ_next_ = self.onlineQNetwork(self.batch_next_state)
            online_max_action = torch.argmax(onlineQ_next_, dim=1, keepdim=True)
            targetQ_next = targetQ_next.detach()
            self.y = batch_reward + (1 - batch_done) * gamma * targetQ_next.gather(1, online_max_action.long())

    def calc_loss(self):
        raw_loss = (self.onlineQ_next - self.y.squeeze(-1)) ** 2 
        raw_loss_v =  self.batch_weights * raw_loss
                    
        sample_priorities = (raw_loss_v + 1e-5).data.cpu().numpy()
                    
        self.loss = raw_loss_v.mean() 

        self.memory_replay.update_priorities(self.batch_indices, sample_priorities)

        return self.loss
    
    def optim_step(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_current_buffer_size(self):
        return len(self.memory_replay)
    
    def update_target_network(self):
        self.targetQNetwork.load_state_dict(copy.deepcopy(self.onlineQNetwork.state_dict()))
    
    def save_memory_replay(self):
        self.memory_replay.save()

    def save_model(self, epoch, model_path):
        current_time = dt.now().strftime("%Y_%m_%d_%H_%M")
            
        save_dict = {'online_model': self.onlineQNetwork.state_dict(),
                        'target_model': self.targetQNetwork.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'loss_fn': nn.MSELoss()}

        if not os.path.isdir('saved_models'):
                os.mkdir('saved_models')
            
        torch.save(save_dict, os.path.join(model_path, f'save_model_{current_time}.pth'))
