# This file can help you to check the communication between unity and python.

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import random
import numpy as np

env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()
behavior_name = list(env.behavior_specs)[0]
print(f"Behavior name {behavior_name} .")
spec = env.behavior_specs[behavior_name]
Obs_types = len(spec.observation_specs)
print(f"There are {Obs_types} types of observation.")

# Is the Action continuous or multi-discrete ?
if spec.action_spec.continuous_size > 0:
    print(f"There are {spec.action_spec.continuous_size} continuous actions")
if spec.action_spec.is_discrete():
    print(f"There are {spec.action_spec.discrete_size} discrete actions")

try:
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        for agents in decision_steps:
            next_obs1 = decision_steps[agents].obs[0]
            next_obs2 = decision_steps[agents].obs[1]
            reward = decision_steps[agents].reward

        for agents in terminal_steps:
            next_obs1 = terminal_steps[agents].obs[0]
            next_obs2 = terminal_steps[agents].obs[1]
            reward = terminal_steps[agents].reward

        print(f"Obs1 shape : {next_obs1.shape}")
        print(f"Obs2 shape : {next_obs2.shape}")
        print(f"Step reward : {reward}")

        # execute action
        action_tuple = ActionTuple()
        action = 2*np.random.rand(len(decision_steps), spec.action_spec.continuous_size)-1
        print(f"Action value : {action}")
        action_tuple.add_continuous(action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

except BaseException:
    env.close()
    print("Environment close.")
