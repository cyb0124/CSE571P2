import os
from map_utils import *
import numpy as np
import car


def sampleMap(map_dir_list):
  """ Get a random map configuration file """
  map = np.random.choice(map_dir_list)
  map_config = 'maps/' + map + '/floorplan.yaml' 
  return map_config

def visualizeGoal(start, goal, ax, vis):
  ax.clear()
  vis.draw_map()
  car.draw(start, ax, 'green')
  car.draw(goal, ax, 'blue')
  plt.show()

def sampleStart(map):
  x_min, x_max, y_min, y_max = map._compute_map_bbox()
  while(True):
    x_rand = np.random.random_sample() * (x_max - x_min) + x_min
    y_rand = np.random.random_sample() * (y_max - y_min) + y_min
    theta_rand = np.random.random_sample() * 2 * np.pi - np.pi
    s = np.array([x_rand, y_rand, theta_rand])
    if(car.collision_violation(s, map)):
      continue
    else:
      return s

def sampleGoal(map, start, num_steps=100, step_size=0.1, min_dis=1.0):
  """ 
  Note: We need to make sure that the goal is reachable, and the path is not too long
  We apply some random walk from the start point.
  """
  current_pos = start[:2]
  pre_theta = np.random.random_sample() * np.pi * 2
  for i in range(num_steps):
    theta = pre_theta + (np.random.random_sample() - 0.5) * np.pi/2
    next_x = current_pos[0] + step_size*np.cos(theta)
    next_y = current_pos[1] + step_size*np.sin(theta)
    next_pos = np.array([next_x, next_y])
    if car.collision_violation(next_pos, map):
      continue
    else:
      #print(np.linalg.norm(current_pos - start[:2]))
      current_pos = next_pos
      pre_theta = theta
  
  return current_pos

if __name__ == '__main__':
  dir_list = os.listdir('maps')
  while(True):
    m = Map(sampleMap(dir_list))
    start = sampleStart(m)
    goal = sampleGoal(m, start, num_steps=200, step_size=0.05) 
    fig, ax = plt.subplots()
    vis = Visualizer(m, ax)
    visualizeGoal(start, goal, ax, vis)



