# This version is used for Agent with unity Ray perception sensor 3D and upCamera above Area3 (3,200,200)
# addition data have 1.Agent&Target distance(float) 2.Agent&Target angle(float) 3.Agent moving velocity(Vector3) 4. Agent rotation value(Quaternion)
# Agent Ray Sensor has 205 float var
# Agent addition data have 9 var
# Agent's Total observation num is 214

# Author : Timothy

from os import stat
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import argparse

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import wandb


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--wandb', dest='wandb', action='store_true', default=False)
parser.add_argument('--start_eps', type=int, default=0)

args = parser.parse_args()
# Super Parameter
#######################################
GPU = True
device_idx = 0
hierarchical_freq = 20  # freqency of first layer command
replay_buffer_size = 10000
img_shape = 200
agent_action_dim = 2
hierarchical_action_dim = 4
learning_rate = 0.01
EPSILON = 0.9
max_episodes  = 15000
frame_idx   = 0
batch_size  = 1
explore_steps = 200  # for random action sampling in the beginning of training
update_itr = 64
AUTO_ENTROPY=True
DETERMINISTIC=False
# Second layer model
GoStraight_model_path = './model/GoStraight/sac_Ray_GoStraight_1103Reward'
Escape_model_path = './model/Escape/sac_Ray_Escape_1101Reward'
PassThrought_model_path = './model/PedestrianPassThrough/sac_Ray_PassThrough_1026Reward'
# First layer model
model_path = './model/Hierarchical/Img_DRQN_random'
project_name = "Hierarchical_IMG_DRQN"
# Unity Env Setting
unity_mode = "Editor"   #Use 'Editor' or 'BuildGame'
buildGame_Path = "/home/timothy/Unity/BuildedGames/Hierarchical_img_Field/hierarchical_img.x86_64"
unity_workerID = 0
unity_turbo_speed = 4.0
#######################################
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)
#######################################
# initial W&B
if args.train and args.wandb:
    wandb.init(project=project_name, entity="timothy584")
    # Log for super parameter
    wandb.config.update({
        "epochs": max_episodes, 
        "batch_size": batch_size,
        "Random_steps" : explore_steps,
        "Hierarchical_Freqency" : hierarchical_freq,
        "replay_buffer_size" : replay_buffer_size,
        "img_shape" : img_shape
    })
#######################################
# Unity control function
#######################################
# obs[0] RGB img, obs[1] Ray Sensor, obs[2] Agent state
class Unity_wrapper:
    def __init__(self, env, behavior_name, agent_num, action_num):
        self.env = env
        self.behavior = behavior_name
        self.agentNum = agent_num
        self.actionNum = action_num
    def unity_step(self, action):
        # numpy action input
        # execute action
        action_tuple = ActionTuple()
        action_tuple.add_continuous(action.reshape(self.agentNum, self.actionNum))
        self.env.set_actions(self.behavior, action_tuple)
        self.env.step()
        
        # Get next_state, reward, done
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        
        for agents in decision_steps:
            next_whole_img = decision_steps[agents].obs[0]   #img(200,200,3)
            next_whole_img = np.transpose(next_whole_img, (2,0,1))  # change shape to (3,200,200)
            next_obs_ray = decision_steps[agents].obs[1]  #shape(205,)
            next_obs_state = decision_steps[agents].obs[2]  #shape(9,)
            
            next_state = np.concatenate((next_obs_ray, next_obs_state), axis=0) # output (214,)
            next_state = np.around(next_state, decimals=4)
            
            reward = decision_steps[agents].reward
            
            done = False
            
        for agents in terminal_steps:
            next_whole_img = terminal_steps[agents].obs[0]   #img(200,200,3)
            next_whole_img = np.transpose(next_whole_img, (2,0,1))  # change shape to (3,200,200)
            next_obs_ray = terminal_steps[agents].obs[1]
            next_obs_state = terminal_steps[agents].obs[2]

            next_state = np.concatenate((next_obs_ray, next_obs_state), axis=0) # output (214,)
            next_state = np.around(next_state, decimals=4)

            reward = terminal_steps[agents].reward
            
            done = True
        
        return next_whole_img, next_state, reward, done

    def unity_reset(self):
        self.env.reset()
        # Get first state
        decision_steps, _ = self.env.get_steps(self.behavior)
        for agents in decision_steps:
            whole_img = decision_steps[agents].obs[0]   #img(200,200,3)
            whole_img = np.transpose(whole_img, (2,0,1))  # change shape to (3,200,200)
            obs_ray = decision_steps[agents].obs[1]
            obs_state = decision_steps[agents].obs[2]
            
            state = np.concatenate((obs_ray, obs_state), axis=0)
            state = np.around(state, decimals=4)
            
        return whole_img, state
#######################################
# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        # Notice : this only use for batch_size = 1
        s_lst, a_lst, r_lst, ns_lst, d_lst=[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            state, action, reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        return s_lst, a_lst, r_lst, ns_lst, d_lst
    
    def __len__(self):
        return len(self.buffer)
#######################################
# Convolution DQN Network
class DRQN(nn.Module):
    def __init__(self, in_channels=3, num_actions=4):
        super(DRQN, self).__init__()
        self.num_action = num_actions

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc4 = nn.Linear(3 * 3 * 64, 512)
        self.gru = nn.GRU(512, num_actions, batch_first=True) # input shape (batch, seq, feature)

    def forward(self, x, hidden = None, max_seq = 1, batch_size=1):
        # DQN input B*C*feature (b 3 200 200)
        # DRQN input B*C*feature (b*seq_len 3 200 200)
        x = self.pool(F.relu(self.conv1(x)))    # (b, 32, 24, 24)
        x = self.pool(F.relu(self.conv2(x)))    # (b, 64, 5, 5)
        x = F.relu(self.conv3(x))               # (b, 64, 3, 3)
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        hidden = self.init_hidden(batch_size) if hidden is None else hidden
        # before go to RNN, reshape the input to (barch, seq, feature)
        x = x.reshape(batch_size, max_seq, 512)
        return self.gru(x, hidden)
    
    def init_hidden(self, batch_size):
        # initialize hidden state to 0
        return torch.zeros(1, batch_size, self.num_action, device=device, dtype=torch.float)

#######################################
# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(device)
        action = self.action_range* torch.tanh(mean + std*z)
        
        action = self.action_range* torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action
#######################################
# DQN Trainer
class DQN_Trainer(object):
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer
        self.eval_net = DRQN(num_actions=hierarchical_action_dim).to(device)
        self.target_net = DRQN(num_actions=hierarchical_action_dim).to(device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def select_action(self, state, hidden=None):
        x = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values, hidden = self.eval_net.forward(x, hidden)
        q_values = torch.max(q_values, 1)[1].detach().cpu().numpy()
        if np.random.uniform() < EPSILON:
            aciton = q_values[0]
        else:
            action = np.random.randint(0, hierarchical_action_dim)
        return action, hidden
    
    def learn(self, batch_size, GAMMA=0.99):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        # print('img_state shape :', img_state.shape)
        # print('target_state shape :', target_state.shape)
        state      = torch.FloatTensor(state).squeeze(0).to(device)
        action     = torch.FloatTensor(action).to(device).type(torch.int64)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        next_state = torch.FloatTensor(next_state).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        q_eval, hidden = self.eval_net(state,max_seq=int(state.shape[0]/batch_size), batch_size=batch_size)
        
        q_eval = q_eval.gather(1, action.unsqueeze(1))
        q_next, hidden_target = self.target_net(next_statemax_seq=int(state.shape[0]/batch_size), batch_size=batch_size).detach()
        q_target = reward + GAMMA * q_next.max(1)[0].view(batch_size, 1)
        
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if args.wandb:
            wandb.log({"Q Network Loss" : loss})
    
    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path+'_DQN')

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path+'_DQN'))
        self.eval_net.eval()
    
    def initial_with_model(self, path):
        self.eval_net.load_state_dict(torch.load(path+'_DQN'))

#######################################
# SAC Policy Loader
class SacPolicyLoader():
    def __init__(self, escape_path, goStraight_path, passThrough_path, num_inputs, num_actions, hidden_size):
        self.escape_policy = PolicyNetwork(num_inputs, num_actions, hidden_size).to(device)
        self.goStraight_policy = PolicyNetwork(num_inputs, num_actions, hidden_size).to(device)
        self.passThrough_policy = PolicyNetwork(num_inputs, num_actions, hidden_size).to(device)

        self.escape_policy.load_state_dict(torch.load(escape_path+'_policy'))
        self.escape_policy.eval()

        self.goStraight_policy.load_state_dict(torch.load(goStraight_path+'_policy'))
        self.goStraight_policy.eval()

        self.passThrough_policy.load_state_dict(torch.load(passThrough_path+'_policy'))
        self.passThrough_policy.eval()

    def escape_mode(self, state):
        action = self.escape_policy.get_action(state)
        return action
    def goStraight_mode(self, state):
        action = self.goStraight_policy.get_action(state)
        return action
    def passThrough_mode(self, state):
        action = self.passThrough_policy.get_action(state)
        return action
#######################################
# wrap unity env
###################################
channel = EngineConfigurationChannel()
# Used for builded Game
if unity_mode == "BuildGame":
    env = UnityEnvironment(file_name=buildGame_Path, worker_id=unity_workerID, seed=1, side_channels=[channel])
    channel.set_configuration_parameters(width=500, height=500, time_scale=unity_turbo_speed)
# Used for Unity Editer
elif unity_mode == "Editor":
    env = UnityEnvironment(file_name=None, seed=1, side_channels=[])

env.reset()
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]
decision_steps, terminal_steps = env.get_steps(behavior_name)
agent_num = len(decision_steps.agent_id)

print("Success initial Unity Env!")
####################################
# Initial Replay Buffer
replay_buffer = ReplayBuffer(replay_buffer_size)
# Initial DQN Trainer
dqn_trainer=DQN_Trainer(replay_buffer)
# Initial SAC Policy loader
policy_loader=SacPolicyLoader(Escape_model_path, GoStraight_model_path, PassThrought_model_path, 214, 2, 512)
# Initial Unity Wrapper
unity = Unity_wrapper(env, behavior_name, agent_num, agent_action_dim)
####################################
# Main Code
if __name__ == '__main__':
    if args.train:
        eps = 0
        mode_name = ['GoStraight', 'PassThrough', 'Escape', 'Stop']
        # Load trained model
        if args.start_eps > 0:
            eps = args.start_eps
            dqn_trainer.initial_with_model(model_path)
            print("Trained model loaded!")
            print('Path {}'.format(model_path))

        # Episode Loop
        while eps < max_episodes:
            # Get first state
            h_img, state = unity.unity_reset()

            episode_reward = 0
            episode_steps = 0
            action_mode = 0 # Mode_0:GoStright(Default), Mode_1:PassThrough, Mode_2:Escape, Mode_3:Stop
            hidden = None

            while True: # Steps Loop
                # Get Hierarchical first layer action
                if frame_idx > explore_steps:
                    action_mode, hidden = dqn_trainer.select_action(h_img, hidden)
                else:
                    action_mode = random.randint(0,3)
                state_lst = []
                action_lst = []
                reward_lst = []
                next_state_lst = []
                done_lst = []

                h_reward = 0
                # Second Layer action
                for i in range(hierarchical_freq):
                    # GoStraight Mode
                    if action_mode == 0:
                        agent_action = policy_loader.goStraight_mode(state)
                    
                    # PassThrough Mode
                    elif action_mode == 1:
                        agent_action = policy_loader.passThrough_mode(state)
                    
                    # Escape Mode
                    elif action_mode == 2:
                        agent_action = policy_loader.escape_mode(state)

                    # Stop Mode
                    else:
                        agent_action = np.array([0.,0.])

                    next_h_img, next_state, reward, done = unity.unity_step(agent_action)

                    # replay_buffer.push(h_img, action_mode, reward, next_h_img, done)
                    state_lst.append(h_img)
                    action_lst.append(action_mode)
                    reward_lst.append(reward)
                    next_state_lst.append(next_h_img)
                    done_lst.append(done)

                    state = next_state
                    h_img = next_h_img
                    h_reward += reward
                    episode_steps += 1
                    frame_idx += 1

                    if done:
                        break
                
                print('Episode: ', eps, '| Action Mode', mode_name[action_mode], '| Hierarchical_Reward', h_reward)
                replay_buffer.push(state_lst, action_lst, reward_lst, next_state_lst, done_lst)

                episode_reward += h_reward

                if done:
                    break
            
            if len(replay_buffer) > 2*batch_size:
                for i in range(update_itr):
                    dqn_trainer.learn(batch_size)
                print('Model updated.')
            
            if eps % 20 == 0 and len(replay_buffer) > 2*batch_size:
                dqn_trainer.save_model(model_path)
                print('Model Saved.')

            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Steps: ', episode_steps)
            if args.wandb:
                # Add episode reward to W&B
                wandb.log({'Episode_Reward': episode_reward, 'epoch': eps})
                wandb.log({'Episode_Steps': episode_steps, 'epoch': eps})

            eps += 1

        print('Training Finish!')
        dqn_trainer.save_model(model_path)
        print('Model Saved.')

    if args.test:
        dqn_trainer.loss_func(model_path)
        # Episode Loop
        for eps in range(10):
            # reset environment and get state
            h_img, state = unity.unity_reset()
            # Set initial episode starting value
            episode_reward = 0
            episode_steps = 0
            hidden = None

            while True:
                # Firsr Layer Action
                action_mode, hidden = dqn_trainer.select_action(h_img, hidden)
                # Set default parameter
                h_reward = 0
                done = False
                # Second Layer Action
                for i in range(hierarchical_freq):
                    # GoStraight Mode
                    if action_mode == 0:
                        agent_action = policy_loader.goStraight_mode(state)
                    
                    # PassThrough Mode
                    elif action_mode == 1:
                        agent_action = policy_loader.passThrough_mode(state)
                    
                    # Escape Mode
                    elif action_mode == 2:
                        agent_action = policy_loader.escape_mode(state)

                    # Stop Mode
                    else:
                        agent_action = np.array([0.,0.])
                
                    next_h_img, next_state, reward, done = unity.unity_step(agent_action)

                    state = next_state
                    h_reward += reward
                    episode_steps += 1

                    if done:
                        break
                print('Episode: ', eps, '| Action Mode', action_mode, '| Reward', h_reward)
                
                h_img = next_h_img
                episode_reward += h_reward

                if done:
                    # End this episode
                    break
            
            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Steps: ', episode_steps)
    
    env.close()
    print("Environment Closed.")