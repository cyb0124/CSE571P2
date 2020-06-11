from prioritized_experience_database import Pool
import matplotlib.pyplot as plt
import torch, model, car
from map_utils import *
from sampler import *
import numpy as np

RECORD_SAVE_PERIOD = 10
N_RECENT_RECORD = 100
DISPLAY_PERIOD = 10
MIN_COST = 0.5
ALPHA = 0.6
BETA = 0.4

records = []
if DISPLAY_PERIOD > 0:
  fig, ax = plt.subplots()
def visualize(m, s, depth, start, goal):
  ax.clear()
  vis = Visualizer(m, ax)
  vis.draw_map()
  obstacles = depth_to_xy(depth, s[:2], s[2], car.FOV)
  vis.draw_obstacles(obstacles, markeredgewidth=1.5, color='red')
  car.draw(s, ax, 'green')
  car.draw(goal, ax, 'blue')
  ax.set_xlim([min(start[0], goal[0]) - 1, max(start[0], goal[0]) + 1])
  ax.set_ylim([min(start[1], goal[1]) - 1, max(start[1], goal[1]) + 1])
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

for p in actor_opt.param_groups:
  p['lr'] = model.LR
for p in critic_opt.param_groups:
  p['lr'] = model.LR

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
  importance = 1 / torch.tensor([i.getPriority() for i in batch], dtype=torch.float, device=dev) ** BETA
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
  for i, e in zip(batch, (torch.abs(error_1) + torch.abs(error_2)) ** ALPHA):
    i.update(e.item())
  pool.commit()
  critic_opt.zero_grad()
  loss = torch.sum((error_1 * error_1 + error_2 * error_2) * importance)
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
    loss = torch.sum((penalty - q_1) * importance)
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
  step = 0
  while features is not None:
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
      records.append(False)
    else:
      distance_next = np.linalg.norm(s_next[:2] - goal)
      if DISPLAY_PERIOD > 0 and step % DISPLAY_PERIOD == 0:
        print('distance: ' + str(distance_next))
        visualize(m, s, depth, start, goal)
      if distance_next <= car.COLLISION_RADIUS:
        depth_next = None
        features_next = None
        reward = model.REWARD_GOAL
        print('goal')
        records.append(True)
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
    step += 1

count = 0
while True:
  map_name = sampleMap()
  m = Map(map_name)
  start = sampleStart(m)
  goal, cost = sampleGoal(m, start)
  print("current map {}, start={}, goal={}, cost={}".format(map_name, start, goal, cost))
  if cost > MIN_COST: # Skip goals that are too close.
    while True:
      # Train for (map, start, goal) configuration until no collision.
      episode(start, goal, m)
      save_model()
      count += 1
      # Save records every RECORD_SAVE_PERIOD episodes.
      if count % RECORD_SAVE_PERIOD == 0:
        data = np.array(records)
        np.savetxt('records.csv', data, delimiter=',')
        if len(data) > N_RECENT_RECORD:
          data_clip = data[-N_RECENT_RECORD:]
          print("success rate in last {} episode: {}".format(N_RECENT_RECORD, np.sum(data_clip)))
      if records[-1]:
        break
