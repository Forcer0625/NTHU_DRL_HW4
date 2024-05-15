import gym
import numpy as np
import torch
from torch.distributions.normal import Normal
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import deque
from osim.env import L2M2019Env

DEVICE = torch.device("cpu")

def preprocess_leg(in_leg):
    leg = []
    for val in in_leg.values():
        if type(val) == list:
            for num in val:
                leg.append(num)
        else:
            for num in val.values():
                leg.append(num)
    return np.array(leg)

def preprocess_v_tgt_field(v_tgt_field):
    return v_tgt_field.reshape(-1)

def preprocess_pelvis(in_pelvis):
    pelvis = []
    for value in in_pelvis.values():
        if type(value) == list:
            for num in value:
                pelvis.append(num)
        else:
            pelvis.append(value)
    return np.array(pelvis)

def preprocess(state):
    v_tgt_field = preprocess_v_tgt_field(state['v_tgt_field'])
    pelvis = preprocess_pelvis(state['pelvis'])
    r_leg = preprocess_leg(state['r_leg'])
    l_leg = preprocess_leg(state['l_leg'])
    return np.concatenate([v_tgt_field, pelvis, r_leg, l_leg], axis=0)

class FrameStack():
    def __init__(self, n_stacks:int=4):
        self.n_stacks = n_stacks
        self.frame_buffer = deque(maxlen=n_stacks)

    def get(self):
        stacked_frames = np.stack(self.frame_buffer, axis=0).reshape(-1)
        return stacked_frames # [n_stacks,H,W]

    def push(self, image:np.ndarray):
        self.frame_buffer.append(image)
        while len(self.frame_buffer) < self.n_stacks:
            self.frame_buffer.append(image)

    def render(self):
        pass

    def clear(self):
        self.frame_buffer.clear()
    
    def next_frame(self, image:np.ndarray):
        '''Return stacked frames the next frame'''
        temp = deepcopy(self.frame_buffer)
        temp.append(image)
        return np.stack(temp, axis=0).reshape(-1)

class Actor(nn.Module):
    def __init__(self, in_dims, n_actions, lr=1e-4):
        super(Actor, self).__init__()
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(in_dims, 512)
        self.fc2 = nn.Linear(512, 512)
        self.mean = nn.Linear(512, n_actions)
        self.sigma = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(DEVICE)

    def forward(self, state):
        action_prob = self.fc1(state)
        action_prob = F.relu(action_prob)
        action_prob = self.fc2(action_prob)
        action_prob = F.relu(action_prob)

        mean = self.mean(action_prob)
        sigma = torch.clamp(self.sigma(action_prob), min=self.reparam_noise, max=1)

        return mean, sigma
    
    def sample(self, state, reparameterize=True):
        mean, sigma = self.forward(state)
        probs = Normal(mean, sigma)

        if reparameterize:
            actions = probs.rsample()
        else:
            actions = probs.sample()
        
        action = torch.tanh(actions)
        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        #self.action_space = gym.spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)
        env = L2M2019Env(visualize=False, difficulty=2)
        obs = env.reset(project=True, obs_as_dict=True)
        obs = preprocess(obs)
        self.n_frame_stack = 4
        self.frame_buffer = FrameStack(self.n_frame_stack)
        self.frame_buffer.push(obs)
        obs = self.frame_buffer.get()
        
        self.actor = Actor(obs.shape[0], 22, 1e-4)
        self.actor.load_state_dict(torch.load('112062546_hw4_data', map_location=torch.device('cpu')))

        self.frame_buffer.clear()
    
    def select_action(self, obs:np.ndarray):
        frame = preprocess(obs)
        self.frame_buffer.push(frame)
        
        stack_frame = self.frame_buffer.get()
        stack_frame = torch.tensor([stack_frame], dtype=torch.float).to(DEVICE)
        actions, _ = self.actor.sample(stack_frame, False)
        return actions.cpu().detach().numpy()[0]

    def act(self, observation):
        return self.select_action(observation)
