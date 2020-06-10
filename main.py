from prioritized_experience_database import Pool
import matplotlib.pyplot as plt
import torch, model, car
from map_utils import *
import numpy as np
from sampler import *

DISPLAY = True
fig = None
ax = None
vis = None
records = []

def visualize(s, depth):
  ax.clear()
  vis.draw_map()
  obstacles = depth_to_xy(depth, s[:2], s[2], car.FOV)
  vis.draw_obstacles(obstacles, markeredgewidth=1.5, color='red')
  car.draw(s, ax, 'green')
  car.draw(goal, ax, 'blue')
  plt.pause(0.01)

FN_MODEL = 'model.pt'
pool = Pool('pool.db')
dev = torch.device('cuda')
actor = model.Actor().to(dev)
actor_t = model.Actor().to(dev)
critic_1 = model.Critic().to(dev)
critic_2 = model.Critic().to(dev)
critic_1_t = model.Critic().to(dev)
critic_2_t = model.Critic().to(dev)
actor_opt = torch.optim.Adam(actor.parameters())
critic_opt = torch.optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()))
params_c = list(actor.parameters()) + list(critic_1.parameters()) + list(critic_2.parameters())
params_t = list(actor_t.parameters()) + list(critic_1_t.parameters()) + list(critic_2_t.parameters())

if os.path.isfile(FN_MODEL):
  save = torch.load(FN_MODEL)
  actor.load_state_dict(save['actor'])
  actor_t.load_state_dict(save['actor_t'])
  critic_1.load_state_dict(save['critic_1'])
  critic_2.load_state_dict(save['critic_2'])
  critic_1_t.load_state_dict(save['critic_1_t'])
  critic_2_t.load_state_dict(save['critic_2_t'])
  actor_opt.load_state_dict(save['actor_opt'])
  critic_opt.load_state_dict(save['critic_opt'])
  do_target_update = save['do_target_update']
else:
  actor_t.load_state_dict(actor.state_dict())
  critic_1_t.load_state_dict(critic_1.state_dict())
  critic_2_t.load_state_dict(critic_2.state_dict())
  do_target_update = False

def save_model():
  torch.save({
    'actor' : actor.state_dict(),
    'actor_t' : actor_t.state_dict(),
    'critic_1' : critic_1.state_dict(),
    'critic_2' : critic_2.state_dict(),
    'critic_1_t' : critic_1_t.state_dict(),
    'critic_2_t' : critic_2_t.state_dict(),
    'actor_opt' : actor_opt.state_dict(),
    'critic_opt' : critic_opt.state_dict(),
    'do_target_update' : do_target_update
  }, FN_MODEL)

def train():
  batch = []
  for _ in range(model.N_BATCH):
    batch.append(pool.sample())
  features_c = torch.tensor([i.o for i in batch], dtype=torch.float, device=dev)
  features_t = torch.tensor([model.dummy_features if i.oNext is None else i.oNext for i in batch], dtype=torch.float, device=dev)
  is_terminal = torch.tensor([i.oNext is None for i in batch], dtype=torch.bool, device=dev)
  action_c = torch.tensor([i.a for i in batch], dtype=torch.float, device=dev)
  reward = torch.tensor([i.r for i in batch], dtype=torch.float, device=dev)
  noise_s = torch.normal(mean=torch.zeros((model.N_BATCH, 2), dtype=torch.float, device=dev), std=model.NOISE_S)
  noise_s = torch.clamp(noise_s, -model.CLIP_S, model.CLIP_S)
  with torch.no_grad():
    action_t = torch.clamp(actor_t.forward(features_t), -1, 1)
    action_t = torch.clamp(action_t + noise_s, -1, 1)
    query_t = torch.cat((features_t, action_t), 1)
    q_t_1 = critic_1_t.forward(query_t).squeeze(1)
    q_t_2 = critic_2_t.forward(query_t).squeeze(1)
    q_t = torch.where(is_terminal, torch.zeros(model.N_BATCH, dtype=torch.float, device=dev), torch.min(q_t_1, q_t_2))
    q_t = q_t * model.DISCOUNT + reward
  query_c = torch.cat((features_c, action_c), 1)
  q_1 = critic_1.forward(query_c).squeeze(1)
  q_2 = critic_2.forward(query_c).squeeze(1)
  error_1 = q_1 - q_t
  error_2 = q_2 - q_t
  for i, e in zip(batch, torch.abs(error_1) + torch.abs(error_2)):
    i.update(e.item())
  pool.commit()
  critic_opt.zero_grad()
  loss = torch.mean(error_1 * error_1 + error_2 * error_2)
  loss.backward()
  critic_opt.step()
  
  global do_target_update
  if do_target_update:
    do_target_update = False
    action_c = actor.forward(features_c)
    penalty = action_c ** 2
    penalty = model.SATURATION_PENALTY * torch.sum(penalty, 1)
    action_c = torch.clamp(action_c, -1, 1)
    query_c = torch.cat((features_c, action_c), 1)
    q_1 = critic_1.forward(query_c).squeeze(1)
    actor_opt.zero_grad()
    loss = torch.mean(penalty - q_1)
    loss.backward()
    actor_opt.step()
    with torch.no_grad():
      for t, c in zip(params_t, params_c):
        t.copy_(model.TARGET_MOMENTUM * t + (1 - model.TARGET_MOMENTUM) * c)
  else:
    do_target_update = True

def episode(start, goal, m):
  s = start
  distance = np.linalg.norm(s[:2] - goal)
  depth = m.get_1d_depth(s[:2], s[2], car.FOV, car.N_RAY)
  features = model.assemble_features(depth, car.compute_relative_goal(s, goal))
  while features is not None:
    if DISPLAY:
      visualize(s, depth)
    with torch.no_grad():
      action = actor.forward(torch.tensor([features], dtype=torch.float, device=dev))[0].cpu().numpy()
    action = np.clip(action, -1, 1)
    action = np.clip(action + np.random.normal(scale=model.NOISE_E, size=2), -1, 1)
    s_next = car.simulate(s, action)
    if car.collision_violation(s_next, m):
      distance_next = None
      depth_next = None
      features_next = None
      reward = model.REWARD_COLLISION
      print('collision')
      records.append(0)
    else:
      distance_next = np.linalg.norm(s_next[:2] - goal)
      #print('distance=' + str(distance_next))
      if distance_next <= car.COLLISION_RADIUS:
        depth_next = None
        features_next = None
        reward = model.REWARD_GOAL
        print('goal')
        records.append(1)
      else:
        depth_next = m.get_1d_depth(s_next[:2], s_next[2], car.FOV, car.N_RAY)
        features_next = model.assemble_features(depth_next, car.compute_relative_goal(s_next, goal))
        reward = model.REWARD_IDLE + model.REWARD_PROGRESS * (distance - distance_next)
    pool.addSample(features, action, reward, features_next)
    train()
    s = s_next
    distance = distance_next
    depth = depth_next
    features = features_next

dir_list = os.listdir('maps')

if DISPLAY:
  fig, ax = plt.subplots()

count = 0
while True:
  m = Map(sampleMap(dir_list))
  if DISPLAY:
    vis = Visualizer(m, ax)
  start = sampleStart(m)
  goal = sampleGoal(m, start, num_steps=200, step_size=0.05)  
  episode(start, goal, m)
  save_model()
  count += 1
  # save records every 10 episodes
  if count>1 and count%10 == 0:
    data = np.array(records)
    np.savetxt('records.csv', data, delimiter=',')
    if len(data) > 100:
      data_clip = data[-100:]
      print("success rate in last 100 episode: {}".format(np.sum(data_clip)))
