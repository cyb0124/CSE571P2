import numpy as np
import car
import numpy as np

class AStarNode:
    def __init__(self, coord):
        self.coord = coord
        # open, cost, heuristic, priority, heap_idx

    def update_priority(self):
        self.priority = self.cost + self.heuristic

class AStarPlanner:
    def __init__(self, map, goal_config, epsilon=10):
        self.goal_config = goal_config
        self.epsilon = 10
        self.nodes = {}
        self.heap = []
        self.map = map

    def swap_node(self, a, b):
        self.heap[a].heap_idx = b
        self.heap[b].heap_idx = a
        self.heap[a], self.heap[b] = self.heap[b], self.heap[a]

    def percolate_up(self, idx):
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[parent].priority <= self.heap[idx].priority:
                break
            self.swap_node(idx, parent)
            idx = parent

    def percolate_down(self, idx):
        while True:
            child = idx * 2 + 1
            if child >= len(self.heap):
                break
            if child + 1 < len(self.heap) and self.heap[child + 1].priority < self.heap[child].priority:
                child += 1
            if self.heap[idx].priority <= self.heap[child].priority:
                break
            self.swap_node(idx, child)
            idx = child

    def open_node(self, node):
        node.open = True
        node.update_priority()
        node.heap_idx = len(self.heap)
        self.heap.append(node)
        self.percolate_up(node.heap_idx)

    def add_node(self, node):
        node.heuristic = np.linalg.norm(node.coord * self.map.resolution - self.goal_config) * self.epsilon
        self.nodes[(node.coord.item(0), node.coord.item(1))] = node
        self.open_node(node)

    def pop_node(self):
        best = self.heap[0]
        best.open = False
        self.swap_node(0, len(self.heap) - 1)
        self.heap.pop()
        self.percolate_down(0)
        return best

    def update_node(self, node):
        old_priority = node.priority
        node.update_priority()
        if node.priority < old_priority:
            self.percolate_up(node.heap_idx)
        else:
            self.percolate_down(node.heap_idx)

    def update_neighbor(self, node, dx, dy):
        delta_coord = np.array([dx, dy])
        neighbor_coord = node.coord + delta_coord
        new_cost = node.cost + np.linalg.norm(delta_coord) * self.map.resolution
        if (neighbor_coord.item(0), neighbor_coord.item(1)) in self.nodes:
            neighbor = self.nodes[(neighbor_coord.item(0), neighbor_coord.item(1))]
            if new_cost < neighbor.cost:
                neighbor.cost = new_cost
                if neighbor.open:
                    self.update_node(neighbor)
                else:
                    self.open_node(neighbor)
        else:
            neighbor = AStarNode(neighbor_coord)
            neighbor.cost = new_cost
            self.add_node(neighbor)

    def plan(self, start_config, max_iter):
        node = AStarNode(np.rint(start_config / self.map.resolution).astype(np.int))
        node.cost = 0.0
        self.add_node(node)
        success = False
        state_count = 0
        while len(self.heap) > 0 and state_count < max_iter:
            node = self.pop_node()
            state_count += 1
            if car.collision_violation(node.coord * self.map.resolution, self.map):
                continue
            if np.sum(np.abs(node.coord * self.map.resolution - self.goal_config)) < self.map.resolution:
                success = True
                break
            self.update_neighbor(node, 1, 0)
            self.update_neighbor(node, 1, 1)
            self.update_neighbor(node, 0, 1)
            self.update_neighbor(node, -1, 1)
            self.update_neighbor(node, -1, 0)
            self.update_neighbor(node, -1, -1)
            self.update_neighbor(node, 0, -1)
            self.update_neighbor(node, 1, -1)
        if not success:
            return -1
        return node.cost
