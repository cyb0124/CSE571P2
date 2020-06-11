import torch, car
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

LR = 0.001
N_BATCH = 32
N_LAYER_1 = 400
N_LAYER_2 = 300
TARGET_MOMENTUM = 0.999
NOISE_E = 1.0
NOISE_S = 0.2
CLIP_S = 0.5
DISCOUNT = 0.998
REWARD_COLLISION = -10
REWARD_GOAL = 10
REWARD_IDLE = -1
REWARD_PROGRESS = 100
SATURATION_PENALTY = 0.1

def assemble_features(depth, relative_goal):
  return np.concatenate((depth / car.LASER_MAX_RANGE, relative_goal))

def crelu(x):
  return torch.cat((F.relu(x), F.relu(-x)), 1)

dummy_features = np.zeros(car.N_RAY + 2)

class Actor(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(len(dummy_features), N_LAYER_1)
    self.layer_2 = nn.Linear(N_LAYER_1 * 2, N_LAYER_2)
    self.layer_3 = nn.Linear(N_LAYER_2 * 2, 2)
  
  def forward(self, x):
    x = crelu(self.layer_1(x))
    x = crelu(self.layer_2(x))
    return self.layer_3(x)

class Critic(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(len(dummy_features) + 2, N_LAYER_1)
    self.layer_2 = nn.Linear(N_LAYER_1 * 2, N_LAYER_2)
    self.layer_3 = nn.Linear(N_LAYER_2 * 2, 1)
  
  def forward(self, x):
    x = crelu(self.layer_1(x))
    x = crelu(self.layer_2(x))
    return self.layer_3(x)
