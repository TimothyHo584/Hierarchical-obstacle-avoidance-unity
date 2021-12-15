# This version is used for Agent with unity Ray perception sensor 3D and upCamera above Area3 (3,200,200)
# addition data have 1.Agent&Target distance(float) 2.Agent&Target angle(float) 3.Agent moving velocity(Vector3) 4. Agent rotation value(Quaternion)
# Agent Ray Sensor has 205 float var
# Agent addition data have 9 var
# Agent's Total observation num is 214
# learn k step experience from human.

# Author : Timothy

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Super Parameter
#######################################
GPU = True
device_idx = 0

# Env Setting
img_shape = 200
agent_action_dim = 2
hierarchical_action_dim = 4

# Network Setting
hierarchical_freq = 25
DETERMINISTIC = False

# Second Layer Model
GoStraight_model_path = './model/GoStraight/sac_Ray_GoStraight_1103Reward'
Escape_model_path = './model/Escape/sac_Ray_Escape_1101Reward'
PassThrought_model_path = './model/PedestrianPassThrough/sac_Ray_PassThrough_1026Reward'

# Unity Env Setting
unity_mode = "BuildGame"   #Use 'Editor' or 'BuildGame'
buildGame_Path = "/home/timothy/Unity/BuildedGames/Hierarchical_Gray_FullView/hierarchical_gray_fullview.x86_64"
unity_workerID = 2
unity_turbo_speed = 1.0
#######################################
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)
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
            next_whole_img = decision_steps[agents].obs[0]  # img(200,200,3)
            next_whole_img = np.transpose(next_whole_img, (2, 0, 1))  # change shape to (3,200,200)
            next_obs_ray = decision_steps[agents].obs[1]  # shape(205,)
            next_obs_state = decision_steps[agents].obs[2]  # shape(9,)

            next_state = np.concatenate((next_obs_ray, next_obs_state), axis=0)  # output (255,)
            next_state = np.around(next_state, decimals=4)

            reward = decision_steps[agents].reward

            done = False

        for agents in terminal_steps:
            next_whole_img = terminal_steps[agents].obs[0]  # img(200,200,3)
            next_whole_img = np.transpose(next_whole_img, (2, 0, 1))  # change shape to (3,200,200)
            next_obs_ray = terminal_steps[agents].obs[1]
            next_obs_state = terminal_steps[agents].obs[2]

            next_state = np.concatenate((next_obs_ray, next_obs_state), axis=0)  # output (255,)
            next_state = np.around(next_state, decimals=4)

            reward = terminal_steps[agents].reward

            done = True

        return next_whole_img, next_state, reward, done

    def unity_reset(self):
        self.env.reset()
        # Get first state
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        for agents in decision_steps:
            whole_img = decision_steps[agents].obs[0]  # img(200,200,3)
            whole_img = np.transpose(whole_img, (2, 0, 1))  # change shape to (3,200,200)
            obs_ray = decision_steps[agents].obs[1]
            obs_state = decision_steps[agents].obs[2]
            # print("Obs_ray shape :", obs_ray.shape)
            # print("Obs_state shape :", obs_state.shape)

            state = np.concatenate((obs_ray, obs_state), axis=0)
            state = np.around(state, decimals=4)

        return whole_img, state
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
# Initial SAC Policy loader
policy_loader=SacPolicyLoader(Escape_model_path, GoStraight_model_path, PassThrought_model_path, 214, 2, 512)
# Initial Unity Wrapper
unity = Unity_wrapper(env, behavior_name, agent_num, agent_action_dim)
####################################
# Main Code
if __name__ == '__main__':
    try:
        frame_idx = 0
        eps = 0
        mode_name = ['GoStraight', 'PassThrough', 'Escape', 'Stop']

        # Episode Loop
        while True:
            # Get first state
            _, state = unity.unity_reset()

            episode_reward = 0
            episode_steps = 0
            action_mode = 0  # Mode_0:GoStright(Default), Mode_1:PassThrough, Mode_2:Escape, Mode_3:Stop

            while True:  # Steps Loop
                # Get Hierarchical first layer action
                action_mode = int(input("Key in action mode :"))

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

                    next_h_img, next_state, reward, done = unity.unity_step(agent_action)

                    state = next_state
                    h_reward += reward
                    episode_steps += 1
                    frame_idx += 1

                    if done:
                        break

                print('Episode: ', eps, '| Action Mode', mode_name[action_mode], '| Hierarchical_Reward', h_reward)

                episode_reward += h_reward

                if done:
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Steps: ', episode_steps)

            eps += 1
    except KeyboardInterrupt:
        env.close()
        print("Environment Closed.")