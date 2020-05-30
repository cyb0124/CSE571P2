# The non-holonomic robot from hw3
import matplotlib.pyplot as plt
from map_utils import *
import numpy as np

# Dynamics
MAX_LINEAR_VEL = 0.2
MAX_STEER_ANGLE = 1
L = 0.075
DT = 0.1

# Laser
LASER_MAX_RANGE = 4
FOV = np.deg2rad(240)
N_RAY = 100

# Collision
COLLISION_RADIUS = 0.15
COLLISION_DIVS = 100

# s: [x, y, heading]
# a: [linear_vel, steer_angle] normalized
def simulate(s, a):
  linear_vel = a[0] * MAX_LINEAR_VEL
  steer_angle = a[1] * MAX_STEER_ANGLE
  s_next = np.empty_like(s)
  s_next[0] = s[0] + linear_vel * np.cos(s[2]) * DT
  s_next[1] = s[1] + linear_vel * np.sin(s[2]) * DT
  s_next[2] = s[2] + (linear_vel/L) * np.tan(steer_angle) * DT
  s_next[2] = s_next[2] % (2*np.pi)
  return s_next

def collision_violation(s, m):
  thetas = np.linspace(-np.pi, np.pi, COLLISION_DIVS, endpoint=False)
  circle = np.stack([np.cos(thetas), np.sin(thetas)], 1) * COLLISION_RADIUS
  coords = m.grid_coord_batch(s[None, :2] + circle)
  return np.any(coords[:, 0] < 0) \
    or np.any(coords[:, 0] >= m.occupancy_grid.shape[1]) \
    or np.any(coords[:, 1] < 0) \
    or np.any(coords[:, 1] >= m.occupancy_grid.shape[0]) \
    or np.any(m.occupancy_grid[coords[:, 1], coords[:, 0]])

def draw(s, ax, color):
  ax.add_artist(plt.Circle((s[0], s[1]), COLLISION_RADIUS, color=color, fill=False))
  if len(s) > 2:
    ax.arrow(s[0], s[1], np.cos(s[2]) * COLLISION_RADIUS, np.sin(s[2]) * COLLISION_RADIUS, color=color)

def compute_relative_goal(s, goal):
  return rotate_2d(goal - s[:2], -s[2])
