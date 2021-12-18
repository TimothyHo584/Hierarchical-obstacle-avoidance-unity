# This version is used for Agent with unity Ray perception sensor 3D and agent&goal state
# addition data have 1.Agent&Target distance(float) 2.Agent&Target angle(float) 3.Agent moving velocity(Vector3) 4. Agent rotation value(Quaternion)
# Agent Ray Sensor has 205 float var
# Agent addition data have 9 var
# Agent's Total observation num is 214

# Design for DQN with hierarchical first layer state only agent position and goal position
# include dueling DDQN and DDQN.

# Author : Timothy

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Normal

import argparse
from collections import deque

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import wandb

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--wandb', dest='wandb', action='store_true', default=False)
parser.add_argument('--dueling', dest='dueling', action='store_true', default=False)
parser.add_argument('--h_freq', type=int, default=10, help="Hierarchical first layer command frequency.")
parser.add_argument('--run_id', type=int, default=0, help="Process on different channel connect with unity.")
parser.add_argument('--turbo', type=int, default=1, help="Adjust unity env speed.")

args = parser.parse_args()
# Super Parameter
#######################################
GPU = True
device_idx = 0

# Env Setting
agent_action_dim = 2
hierarchical_state_dim = 4
hierarchical_action_dim = 4

# epsilon greedy:
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.1
EXPLORE = 5000

# Network Setting
max_episodes  = 10000
learning_rate = 1e-4
hierarchical_freq = args.h_freq  # freqency of first layer command
# efficient_threshold = 6 # Evaluate efficient of first layer. < this value can get more reward.
batch_size  = 64
replay_buffer_size = 100000
update_itr = 16
DETERMINISTIC = False

# Training model save & wandb log
project_name = "Hierarchical_IMG_DDQN"
if args.dueling:
    model_path = './model/Hierarchical/Dueling_DDQN_simpleState_hFreq_' + str(hierarchical_freq)
    run_name = 'Dueling_DDQN_simpleState_hFreq_' + str(hierarchical_freq)
else:
    model_path = './model/Hierarchical/DDQN_simpleState_hFreq_' + str(hierarchical_freq)
    run_name = 'DDQN_simpleState_hFreq_' + str(hierarchical_freq)

# Second Layer Model
GoStraight_model_path = './model/GoStraight/sac_Ray_GoStraight_1103Reward'
Escape_model_path = './model/Escape/sac_Ray_Escape_1101Reward'
PassThrought_model_path = './model/PedestrianPassThrough/sac_Ray_PassThrough_1026Reward'

# Unity Env Setting
unity_mode = "BuildGame"   #Use 'Editor' or 'BuildGame'
buildGame_Path = "/home/timothy/Unity/BuildedGames/Hierarchical_simple/hierarchical_simple.x86_64"
unity_workerID = args.run_id
unity_turbo_speed = args.turbo
#######################################
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)
#######################################
# initial W&B
if args.train and args.wandb:
    wandb.init(project=project_name, entity="timothy584", name=run_name)
    # Log for super parameter
    wandb.config.update({
        "epochs": max_episodes,
        "batch_size": batch_size,
        "update_itr": update_itr,
        "Random_steps" : EXPLORE,
        "Learning Rate": learning_rate,
        "Hierarchical_Freqency" : hierarchical_freq,
        "replay_buffer_size" : replay_buffer_size
    })
#######################################
# Unity control function
#######################################
# obs[0] gray img, obs[1] Ray Sensor, obs[2] Agent state
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
            next_obs_ray = decision_steps[agents].obs[0]  # shape(205,)
            next_obs_state = decision_steps[agents].obs[1]  # shape(13,)
            next_secLayer_state = next_obs_state[:9]
            next_fstLayer_state = next_obs_state[9:]

            next_state = np.concatenate((next_obs_ray, next_secLayer_state), axis=0)  # output (255,)
            next_state = np.around(next_state, decimals=4)

            reward = decision_steps[agents].reward

            done = False

        for agents in terminal_steps:
            next_obs_ray = terminal_steps[agents].obs[0]    # Ray sensor shape(205,)
            next_obs_state = terminal_steps[agents].obs[1]  # Ray sensor shape(13,)
            next_secLayer_state = next_obs_state[:9]
            next_fstLayer_state = next_obs_state[9:]

            next_state = np.concatenate((next_obs_ray, next_secLayer_state), axis=0)  # output (255,)
            next_state = np.around(next_state, decimals=4)

            reward = terminal_steps[agents].reward

            done = True

        return next_fstLayer_state, next_state, reward, done

    def unity_reset(self):
        self.env.reset()
        # Get first state
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        for agents in decision_steps:
            obs_ray = decision_steps[agents].obs[0]
            obs_state = decision_steps[agents].obs[1]
            secLayer_state = obs_state[:9]
            fstLayer_state = obs_state[9:]
            # print("Obs_ray shape :", obs_ray.shape)
            # print("Obs_state shape :", obs_state.shape)

            state = np.concatenate((obs_ray, secLayer_state), axis=0)
            state = np.around(state, decimals=4)

        return fstLayer_state, state
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
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
#######################################
# Convolution DQN Network
class Conv_DQN(nn.Module):
    def __init__(self, action_dim):  # Default img shape is (3,200,200)
        super(Conv_DQN, self).__init__()
        self.fc1 = nn.Linear(hierarchical_state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        y = F.relu(self.fc1(state))
        y = F.relu(self.fc2(y))
        action = self.fc3(y)
        return action


#######################################
# Dueling DQN Network
class Dueling_DQN(nn.Module):
    def __init__(self, action_dim):
        super(Dueling_DQN, self).__init__()
        # State Value Layer
        self.fc1_value = nn.Linear(hierarchical_state_dim, 128)
        self.fc2_value = nn.Linear(128, 128)
        self.fc3_value = nn.Linear(128, 1)
        # Action Layer
        self.fc1_action = nn.Linear(hierarchical_state_dim, 128)
        self.fc2_action = nn.Linear(128, 128)
        self.fc3_action = nn.Linear(128, action_dim)

    def forward(self, state):
        Vy = F.relu(self.fc1_value(state))
        Vy = F.relu(self.fc2_value(Vy))
        Vy = self.fc3_value(Vy)

        Ay = F.relu(self.fc1_action(state))
        Ay = F.relu(self.fc2_action(Ay))
        Ay = self.fc3_action(Ay)

        Q = Vy + (Ay - Ay.mean())

        return Q
#######################################
# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20,
                 log_std_max=2):
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

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
        action.detach().cpu().numpy()[0]
        return action
#######################################
# DQN Trainer
class DQN_Trainer(object):
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer
        if args.dueling:
            self.eval_net = Dueling_DQN(hierarchical_action_dim).to(device)
            self.target_net = Dueling_DQN(hierarchical_action_dim).to(device)
            print("Use Dueling DDQN Network!")
        else:
            self.eval_net = Conv_DQN(hierarchical_action_dim).to(device)
            self.target_net = Conv_DQN(hierarchical_action_dim).to(device)
            print("Use DDQN Network!")

        self.target_net.load_state_dict(self.eval_net.state_dict())  # Let target parameter = eval parameter
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.epsilon = INITIAL_EPSILON

    def select_action(self, state):
        self.eval_net.eval()

        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x = torch.FloatTensor(state).unsqueeze(0).to(device)
        estimate = self.eval_net.forward(x)

        if random.random() < self.epsilon:
            action = random.randint(0, hierarchical_action_dim - 1)
            print("Random Action index : {} | Epsilon : {}".format(action, round(self.epsilon, 4)))
        else:
            action = torch.argmax(estimate).detach().cpu().numpy()
            print("Policy Action index : {} | Epsilon : {}".format(action, round(self.epsilon, 4)))
        return action

    def learn(self, batch_size, GAMMA=0.99):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        # use unsqueeze() to add one dim to be [reward] at the sample dim;
        # print('img_state shape :', img_state.shape)
        # print('target_state shape :', target_state.shape)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).unsqueeze(1).to(device).type(torch.int64)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.BoolTensor(done).to(device)

        self.target_net.eval()
        self.eval_net.eval()

        # double dqn loss
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        action_new = self.eval_net.forward(next_state).max(dim=1)[1].cpu().data.view(-1, 1)
        action_new_onehot = torch.zeros(batch_size, hierarchical_action_dim)
        action_new_onehot = Variable(action_new_onehot.scatter_(1, action_new, 1.0)).to(device)

        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        y = (reward + torch.mul(((self.target_net.forward(next_state) * action_new_onehot).sum(dim=1) * done),
                                    GAMMA))
        # regression Q(s, a) -> y
        self.eval_net.train()
        Q = (self.eval_net.forward(state) * action).sum(dim=1)
        loss = F.mse_loss(input=Q, target=y.detach())

        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if args.wandb:
            wandb.log({"Q Network Loss": loss})

    def sync_target_net(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path))
        self.eval_net.eval()

    def initial_with_model(self, path):
        self.eval_net.load_state_dict(torch.load(path))

#######################################
# SAC Policy Loader
class SacPolicyLoader():
    def __init__(self, escape_path, goStraight_path, passThrough_path, num_inputs, num_actions, hidden_size):
        self.escape_policy = PolicyNetwork(num_inputs, num_actions, hidden_size).to(device)
        self.goStraight_policy = PolicyNetwork(num_inputs, num_actions, hidden_size).to(device)
        self.passThrough_policy = PolicyNetwork(num_inputs, num_actions, hidden_size).to(device)

        self.escape_policy.load_state_dict(torch.load(escape_path + '_policy'))
        self.escape_policy.eval()

        self.goStraight_policy.load_state_dict(torch.load(goStraight_path + '_policy'))
        self.goStraight_policy.eval()

        self.passThrough_policy.load_state_dict(torch.load(passThrough_path + '_policy'))
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
        frame_idx = 0
        eps = 0
        mode_name = ['GoStraight', 'PassThrough', 'Escape', 'Stop']
        # Episode Loop
        for eps in range(max_episodes):
            # Get first state
            h_state, state = unity.unity_reset()

            episode_reward = 0
            episode_steps = 0
            action_mode = 0  # Mode_0:GoStright(Default), Mode_1:PassThrough, Mode_2:Escape, Mode_3:Stop

            while True:  # Steps Loop
                # Get Hierarchical first layer action
                action_mode = dqn_trainer.select_action(h_state)

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
                        agent_action = np.array([0., 0.])

                    next_h_state, next_state, reward, done = unity.unity_step(agent_action)

                    state = next_state
                    h_reward += reward
                    episode_steps += 1
                    frame_idx += 1

                    if done:
                        break

                print("Episode : {} | Action Mode : {} | H_Reward : {} | Replay Buffer capacity(%) : {} %".format(eps,
                    mode_name[action_mode], h_reward, (len(replay_buffer)/replay_buffer_size)*100))

                # stack with each steps
                replay_buffer.push(h_state, action_mode, h_reward, next_h_state, done)

                episode_reward += h_reward

                if len(replay_buffer) > 2*batch_size:
                    for i in range(update_itr):
                        dqn_trainer.learn(batch_size)
                    print('Model updated.')

                if done:
                    break

            if eps % 20 == 0 and len(replay_buffer) > 2*batch_size:
                dqn_trainer.save_model(model_path)
                print('Model Saved.')
                dqn_trainer.sync_target_net()
                print("Target_net sync with Eval_net")

            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Steps: ', episode_steps)
            if args.wandb:
                # Add episode reward to W&B
                wandb.log({'Episode_Reward': episode_reward, 'epoch': eps})
                wandb.log({'Episode_Steps': episode_steps, 'epoch': eps})

        print('Training Finish!')
        dqn_trainer.save_model(model_path)
        print('Model Saved.')

    if args.test:
        frame_idx = 0
        dqn_trainer.loss_func(model_path)
        # Episode Loop
        for eps in range(10):
            # reset environment and get state
            h_state, state = unity.unity_reset()
            # Set initial episode starting value
            episode_reward = 0
            episode_steps = 0
            action_mode = 0

            while True:
                # Firsr Layer Action
                action_mode = dqn_trainer.select_action(h_state)

                # Set default parameter
                h_reward = 0

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
                        agent_action = np.array([0., 0.])

                    next_h_state, next_state, reward, done = unity.unity_step(agent_action)

                    state = next_state
                    h_reward += reward
                    episode_steps += 1

                    if done:
                        break
                print('Episode: ', eps, '| Action Mode', action_mode, '| Reward', h_reward)

                episode_reward += h_reward

                if done:
                    # End this episode
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Steps: ', episode_steps)

    env.close()
    print("Environment Closed.")