import os, car
import numpy as np
from map_utils import *
from AStarPlanner import AStarPlanner

MAP_LIST = os.listdir('maps')

def sampleMap():
  """ Get a random map configuration file """
  map = np.random.choice(MAP_LIST)
  return 'maps/' + map + '/floorplan.yaml'

def sampleStart(map):
  x_min, x_max, y_min, y_max = map._compute_map_bbox()
  while True:
    x_rand = np.random.uniform(x_min, x_max)
    y_rand = np.random.uniform(y_min, y_max)
    theta_rand = np.random.uniform(-np.pi, np.pi)
    s = np.array([x_rand, y_rand, theta_rand])
    if not car.collision_violation(s, map):
      return s

def sampleGoal(map, start, radius=1.5, search_steps=1000, radius_patience=10):
  """
  Note: We need to make sure that the goal is reachable, and the path is not too long
  We use A* planner to check if the goal can be reached in bounded search iterations
  """
  current_pos = start[:2]
  goal = None
  i = 0
  # We first try to find a goal r away.
  # If can't find a valid goal after N tries, we reduce r to its half
  while True:
    theta = np.random.uniform(0, np.pi * 2)
    next_x = current_pos[0] + radius * np.cos(theta)
    next_y = current_pos[1] + radius * np.sin(theta)
    next_pos = np.array([next_x, next_y])
    cost = AStarPlanner(map, next_pos).plan(current_pos, search_steps)
    if cost > 0:
      return next_pos, cost
    i += 1
    if i > radius_patience:
      radius /= 2
      i = 0

if __name__ == '__main__':
  while True:
    map_dir = sampleMap()
    print('map: ' + str(map_dir))
    m = Map(map_dir)
    start = sampleStart(m)
    print('start: ' + str(start))
    goal, cost = sampleGoal(m, start)
    print('goal: ' + str(goal))
    print('cost: ' + str(cost))
    fig, ax = plt.subplots()
    vis = Visualizer(m, ax)
    vis.draw_map()
    car.draw(start, ax, 'green')
    car.draw(goal, ax, 'blue')
    plt.show()
