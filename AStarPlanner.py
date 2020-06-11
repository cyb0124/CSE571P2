import numpy as np
from heapq import *
import itertools
import car

class AStarPlanner(object):    
    def __init__(self, env_map, epsilon):
        self.map = env_map 
        self.nodes = {}
        self.epsilon = epsilon
        self.visited = np.zeros(self.map.occupancy_grid.shape)
        self.back_track = BackTracker()

    def Plan(self, start_config, goal_config, max_itr=1000):
        state_count = 0
        OPEN = My_Priority_Queue()
        OPEN.insert(start_config, self.epsilon * self.h(start_config, goal_config))
        itr = 0
        reached_goal = None
        while(True):
            itr += 1
            candidate = OPEN.delete_min()
            if len(candidate) > 0:
                (c_config, c_cost) = candidate 
                #print(c_config)
            else:
                # all state has been expanded but goal still not found
                #print("ERROR: no plan found!!!")
                return 0.0 
            if(itr > max_itr):
                #print("timed out!")
                return 0.0 
            #if(np.array_equal(c_config, goal_config)):
            if(self.is_equal(c_config, goal_config)):
                cost = c_cost
                reached_goal = c_config
                #print("goal reached!")
                break
            # get all successors and push them to the OPEN list
            successor_list = self.Successors(c_config)
            state_count += 1
            for cost, successor in successor_list:
                current_cost = c_cost - self.epsilon * self.h(c_config, goal_config)
                new_cost = current_cost + cost + self.epsilon * self.h(successor, goal_config)
                if successor in OPEN:
                    if OPEN[successor] > new_cost:
                        OPEN.insert(successor, new_cost)     # update priority
                        self.back_track.record(c_config, successor)
                else:
                    OPEN.insert(successor, new_cost)
                    self.back_track.record(c_config, successor)
            # mark current as visited
            self.MarkVisited(c_config)
        
        # now back-tracking
        plan = []
        current = reached_goal 
        while(not self.is_equal(current, start_config)):
            plan.append(current)
            current = self.back_track.getParent(current)
        plan.append(start_config) 
        plan.reverse()
        #print("States Expanded: %d" % state_count)
        #sprint("Cost: %f" % cost)
        return cost

    def is_equal(self, pos1, pos2):
        ''' check if two position (in world system) is equal '''
        if np.sum(np.abs(pos1 - pos2)) < self.map.resolution:
            return True
        else:
            return False

    def h(self, pos1, pos2):
        pos1 = pos1[:2]
        pos2 = pos2[:2]
        return np.sqrt(np.sum((pos1 - pos2)**2))

    def Successors(self, config):
        """ return all successors that havn't visited and the corresponding cost from config to each successor """
        r = self.map.resolution
        successors_list = []
        for dx in [-r, 0, r]:
            for dy in [-r, 0, r]:
                if dx != 0 or dy !=0:
                    y1 = config[1] + dy
                    x1 = config[0] + dx
                    successor_candidate = np.array((x1, y1))
                    if not car.collision_violation(successor_candidate, self.map) and not self.isVisited(successor_candidate):
                    #if self.env.state_validity_checker(successor_candidate) and not self.isVisited(successor_candidate):
                        successors_list.append((np.sqrt(dx**2 + dy**2), successor_candidate))
        return successors_list

    def MarkVisited(self, config):
        ''' config in world frame'''
        x , y = self.map.grid_coord(config[0], config[1])
        self.visited[y, x] = 1

    def isVisited(self, config):
        ''' conifg in world frame '''
        x , y = self.map.grid_coord(config[0], config[1])
        return (self.visited[y, x] == 1)

class BackTracker:
    def __init__(self):
        self.mydict = {}

    def record(self, parent, child):
        p = (float(parent[0]), float(parent[1]))
        c = (float(child[0]), float(child[1]))
        self.mydict[c] = p 
        
    def getParent(self, child):
        c = (float(child[0]), float(child[1]))
        p = self.mydict[c]
        return np.array((p[0], p[1]))

class My_Priority_Queue:
    ''' This priority queue is implemented with heap queue algorithm
        see https://docs.python.org/2/library/heapq.html#priority-queue-implementation-notes'''
        
    def __init__(self):
        self.pq = [] 
        self.entry_finder = {}
        self.counter = itertools.count()

    def Convert(self, config):
        ''' convert the state into turple which is hashable'''
        return (config[0], config[1])

    def deConvert(self, t):
        return np.array((t[0], t[1]))

    def __contains__(self, config):
        c = self.Convert(config) 
        if c in self.entry_finder:
            if self.entry_finder[c] != None:
                return True
        return False

    def __getitem__(self, config):
        '''This method enables Pythons right-bracket syntax.
        Here, something like  priority_val = my_queue[state]
        becomes possible. Note that the syntax is actually used
        in the insert method above:  self[state] != -1  '''
        c = self.Convert(config)
        if c not in self:
            return -1
        entry = self.entry_finder[c]
        return self.entry_finder[c][0]

    def __delitem__(self, config):
        '''This method enables Python's del operator to delete
        items from the queue.'''
        c = self.Convert(config)
        entry = self.entry_finder.pop(c)
        entry[-1] = None

    def __str__(self):
        txt = "My_Priority_Queue: ["
        for (p,_,s) in self.pq: txt += '('+str(s)+', '+str(p)+')'
        txt += ']'
        return txt

    def delete_min(self):
        ''' Standard priority-queue dequeuing method.'''
        if self.pq==[]: return [] # Simpler than raising an exception.
        while self.pq:
            priority, count, c = heappop(self.pq)
            if c != None:
                del self.entry_finder[c]
                return (self.deConvert(c), priority)

    def insert(self, config, priority):
        ''' add a new task or update the priority of an existing task '''
        c = self.Convert(config)
        if c in self.entry_finder:
            del self[config]
        count = next(self.counter)
        entry = [priority, count, c]
        self.entry_finder[c] = entry
        heappush(self.pq, entry)
        