# This version is used for unity Ray perception sensor 3D.
# addition data have 1.Agent&Target distance(float) 2.Agent&Target angle(float) 3.Agent moving velocity(Vector3) 4. Agent rotation value(Quaternion)
# Ray Sensor has 205 float var
# addition data have 25 var (stack 5 frame)
# Total observation num is 1230 (stack 5 frame)

import argparse
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import torch.onnx

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--behavior', dest='behavior', type=str, default='straight')
parser.add_argument('--wandb', dest='wandb', action='store_true', default=False)
parser.add_argument('--load_model', dest='load_model', action='store_true', default=False)
parser.add_argument('--load_buffer', dest='load_buffer', action='store_true', default=False)
parser.add_argument('--env', type=str, default='Editor', help="Adjust unity training Env.(Editor or BuildGame)")
parser.add_argument('--run_id', type=int, default=0, help="Process on different channel connect with unity.")
parser.add_argument('--turbo', type=int, default=1, help="Adjust unity env speed.")
args = parser.parse_args()

# (Option) show training result on W&B
if args.wandb:
    import wandb
# Super Parameter
#######################################
GPU = True
device_idx = 0

replay_buffer_size = 100000
state_dim = (1230+25)
action_range = 1.
max_episodes  = 8000
batch_size  = 128
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 32
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 512

frame_idx   = 0
log_buffer_dir = './logs/GoStraight_mode/gostraight_mode_end.pickle'
pretrain_model = ''
if args.behavior == 'pedestrian':
    model_path = './model/PedestrianPassThrough/pedestrian_mode'
elif args.behavior == 'escape':
    model_path = './model/Escape/escape_mode'
else:
    model_path = './model/GoStraight/goStraight_mode'
print(f"Behavior Mode : {args.behavior}")

project_name = f"sac_Ray_{args.behavior}_finalEnv"
run_name = f'{args.behavior}_mode_v1'

# Unity Env Setting
unity_mode = args.env   #Use 'Editor' or 'BuildGame'
if args.behavior == 'pedestrian':
    buildGame_Path = "./BuildGame/Pedestrian_mode/pedestrian.exe"
elif args.behavior == 'escape':
    buildGame_Path = "./BuildGame/Escape_mode/escape.exe"
else:
    buildGame_Path = "./BuildGame/GoStraight_mode/gostraight.exe"
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
        "Random_steps" : explore_steps,
        "hidden_dim" : hidden_dim,
        "replay_buffer_size" : replay_buffer_size,
        "state_dim" : state_dim
    })
#######################################
# Unity control function
#######################################
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
            next_obs_ray = decision_steps[agents].obs[0]  #shape(1230,)
            next_obs_state = decision_steps[agents].obs[1]  #shape(25,)
            
            next_state = np.concatenate((next_obs_ray, next_obs_state), axis=0) # output (1255,)
            next_state = np.around(next_state, decimals=4)
            
            reward = decision_steps[agents].reward
            
            done = False
            
        for agents in terminal_steps:
            next_obs_ray = terminal_steps[agents].obs[0]    #shape(1230,)
            next_obs_state = terminal_steps[agents].obs[1]  #shape(25,)

            next_state = np.concatenate((next_obs_ray, next_obs_state), axis=0) # output (1255,)
            next_state = np.around(next_state, decimals=4)

            reward = terminal_steps[agents].reward
            
            done = True
        
        return next_state, reward, done

    def unity_reset(self):
        self.env.reset()
        # Get first state
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        for agents in decision_steps:
            obs_ray = decision_steps[agents].obs[0]
            obs_state = decision_steps[agents].obs[1]
            
            state = np.concatenate((obs_ray, obs_state), axis=0)
            state = np.around(state, decimals=4)
            
        return state
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
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    def store_buffer(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
    def load_buffer(self, path):
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
    def __len__(self):
        return len(self.buffer)
#######################################
# Q Network
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
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
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(device)
        action = self.action_range*torch.tanh(mean + std*z)
        
        action = self.action_range*torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action

    def random_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a.numpy()
#######################################
# SAC Trainer
class SAC_Trainer():
    def __init__(self, replay_buffer, hidden_dim, action_range):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)
        # Log model on wandb
        if args.train and args.wandb:
            wandb.watch(self.soft_q_net1, log="soft_q_net1")
            wandb.watch(self.soft_q_net2, log="soft_q_net2")
            wandb.watch(self.target_soft_q_net1, log="target_soft_q_net1")
            wandb.watch(self.target_soft_q_net2, log="target_soft_q_net2")
            wandb.watch(self.policy_net, log="policy_net")
        
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0
        
        if args.wandb:  # Log alpha on wandb
            wandb.log({"Alpha" : self.log_alpha})

    # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        if args.wandb:  # Log q loss on wandb
            wandb.log({"q_value_loss1" : q_value_loss1})
            wandb.log({"q_value_loss2" : q_value_loss2})
            wandb.log({"policy loss" : policy_loss})


    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        # self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        # self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))

        # self.soft_q_net1.eval()
        # self.soft_q_net2.eval()
        self.policy_net.eval()
    
    def initial_with_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
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

action_dim = len(spec.action_spec)
agent_num = len(decision_steps.agent_id)

action_range = 1.

print('Num Agents : {}, Num Obs : {}, Num Action : {}'.format(agent_num, state_dim, action_dim))
####################################
# Initial Replay Buffer
replay_buffer = ReplayBuffer(replay_buffer_size)
# Initial SAC Trainer
sac_trainer=SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range)
# Initial Unity Wrapper
unity = Unity_wrapper(env, behavior_name, agent_num, action_dim)
####################################
# Main Code
if __name__ == '__main__':
    if args.train:
        eps = 0
        # Load trained model
        '''
        if args.start_eps > 0:
            eps = args.start_eps
            sac_trainer.initial_with_model(model_path)
            print("Trained model loaded!")
            print('Path {}'.format(model_path))
        if args.train_from_model:
            sac_trainer.initial_with_model(model_path)
            print("Trained model loaded!")
            print('Path {}'.format(model_path))
        '''
        # Load replay buffer
        if args.load_buffer:
            replay_buffer.load_buffer(log_buffer_dir)
            explore_steps = 0
            print('Buffer load.')

        # Load trained model
        if args.load_model:
            sac_trainer.initial_with_model(pretrain_model)
            print('Model load.')

        # training loop
        while eps < max_episodes:
            # Get first state
            state = unity.unity_reset()

            episode_reward = 0
            episode_steps = 0
            
            while True:
                if frame_idx > explore_steps:
                    action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
                    # print('Origin action', action)
                else:   # random explore_steps for beginning
                    action = sac_trainer.policy_net.random_action()
                    # print('Origin action', action)
                
                # execute action & Get next_state, reward, done
                next_state, reward, done = unity.unity_step(action)
                # print('next_state shape', next_state.shape)
                replay_buffer.push(state, action, np.around(reward, decimals=5), next_state, done)
                
                state = next_state
                episode_reward += np.around(reward, decimals=5)
                episode_steps += 1
                frame_idx += 1

                if done:
                    break

            if len(replay_buffer) > 2*batch_size:
                for i in range(update_itr):
                    _=sac_trainer.update(batch_size, reward_scale=1., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)
                print('Model updated.')
                
            if eps % 30 == 0 and len(replay_buffer) > 2*batch_size:
                sac_trainer.save_model(model_path)
                print('Model Saved.')

            # print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Steps: ', episode_steps)
            print("Episode : {} | Eps_Reward : {} | Eps_steps : {} | Replay Buffer capacity(%) : {} %".format(eps,
                    episode_reward, episode_steps, (replay_buffer.__len__() / replay_buffer_size) * 100))

            if args.wandb:
                # Add episode reward to W&B
                wandb.log({'Episode_Reward': episode_reward, 'epoch': eps})
                wandb.log({'Episode_Steps': episode_steps, 'epoch': eps})
            
            eps += 1

        print('Training Finish!')
        sac_trainer.save_model(model_path)
        print('Model Saved.')
        replay_buffer.store_buffer(log_buffer_dir)
        print('Latest Buffer Stored.')

    if args.test:
        sac_trainer.load_model(model_path)
        for eps in range(30):
            # reset environment and get state
            state = unity.unity_reset()
            
            episode_reward = 0
            episode_steps = 0

            while True:
                action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
                
                next_state, reward, done = unity.unity_step(action)

                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                if done:
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Steps: ', episode_steps)

    env.close()
    print("Environment Closed.")