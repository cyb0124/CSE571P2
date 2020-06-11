import os
from map_utils import *
import numpy as np
import car
from AStarPlanner import AStarPlanner


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

def sampleGoal(map, start, radius=1.0, search_steps = 1000):
  """ 
  Note: We need to make sure that the goal is reachable, and the path is not too long
  We use A* planner to check if the goal can be reached in bounded search iterations
  """
  current_pos = start[:2]
  planner = AStarPlanner(map, 10)
  goal = None
  r = radius
  i = 0
  N = 10
  # we first try to find a goal r away.
  # if can't find a valid goal after N tries, we reduce r to its half
  while(True):
    theta = np.random.random_sample() * np.pi * 2
    next_x = current_pos[0] + r*np.cos(theta)
    next_y = current_pos[1] + r*np.sin(theta)
    next_pos = np.array([next_x, next_y])
    cost = planner.Plan(current_pos, next_pos, search_steps)
    if cost > 0:
      return next_pos, cost
    else:
      i += 1
    if (i > 10):
      r = r/2
      i = 0

if __name__ == '__main__':
  dir_list = os.listdir('maps')
  while(True):
    map_dir = sampleMap(dir_list)
    print(map_dir)
    m = Map(map_dir)
    start = sampleStart(m)
    print(start)
    goal = sampleGoal(m, start, radius=1.5, search_steps=1000) 
    fig, ax = plt.subplots()
    vis = Visualizer(m, ax)
    visualizeGoal(start, goal, ax, vis)