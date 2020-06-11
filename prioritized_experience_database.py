# Database for experiences and sum-tree for prioritized experience replay.
# Yibo Cao
# 2020/3/4: Created
# 2020/6/1: Added getPriority

import sqlite3
import random
import pickle

class Node:
  def __init__(self, id, priority):
    self.id = id
    self.sum = priority
    self.max = priority
    self.nLeaf = 1
    self.height = 0
    self.parent = None
    self.left = None
    self.right = None

class Tree:
  def __init__(self):
    self.map = dict()
    self.root = None

  def prop(self, node):
    while node is not None:
      if node.left is None:
        node.sum = node.right.sum
        node.max = node.right.max
        node.nLeaf = node.right.nLeaf
        node.height = node.right.height + 1
      elif node.right is None:
        node.sum = node.left.sum
        node.max = node.left.max
        node.nLeaf = node.left.nLeaf
        node.height = node.left.height + 1
      else:
        node.sum = node.left.sum + node.right.sum
        node.max = max(node.left.max, node.right.max)
        node.nLeaf = node.left.nLeaf + node.right.nLeaf
        node.height = max(node.left.height, node.right.height) + 1
      node = node.parent

  def insert(self, newLeaf):
    if self.root is None:
      self.root = newLeaf
      return
    node = None
    leaf = self.root
    while leaf.id is None:
      node = leaf
      if node.left is None or node.right is not None and node.left.height > node.right.height:
        leaf = node.right
        isLeft = False
      else:
        leaf = node.left
        isLeft = True
    newNode = Node(None, None)
    newNode.left = leaf
    newNode.right = newLeaf
    newLeaf.parent = newNode
    if node is None:
      self.root = newNode
    else:
      newNode.parent = node
      if isLeft:
        node.left = newNode
      else:
        node.right = newNode
    self.prop(newNode)

  def set(self, id, priority):
    if id in self.map:
      node = self.map[id]
      node.sum = priority
      node.max = priority
      self.prop(node.parent)
    else:
      node = Node(id, priority)
      self.map[id] = node
      self.insert(node)

  def max(self):
    return 1.0 if self.root is None else self.root.max

  def sum(self):
    return 0.0 if self.root is None else self.root.sum

  def nLeaf(self):
    return 0 if self.root is None else self.root.nLeaf

  def sample(self):
    node = self.root
    while node.id is None:
      if node.left is None or random.uniform(0.0, node.sum) > node.left.sum:
        node = node.right
      else:
        node = node.left
    return node.id

class Sample:
  def __init__(self, pool, id, blob):
    self.pool = pool
    self.id = id
    self.o, self.a, self.r, self.oNext = pickle.loads(blob)

  def update(self, priority):
    self.pool.cursor.execute('update pool set priority=? where id=?', (priority, self.id))
    self.pool.tree.set(self.id, priority)

  def getPriority(self):
    return self.pool.tree.map[self.id].sum

class Pool:
  def __init__(self, fn):
    self.db = sqlite3.connect(fn)
    self.cursor = self.db.cursor()
    self.cursor.execute('pragma synchronous=off')
    self.cursor.execute('''
      create table if not exists pool (
        id integer primary key,
        priority real not null,
        data blob not null
      )
    ''')

    self.tree = Tree()
    self.cursor.execute('select id, priority from pool')
    row = self.cursor.fetchone()
    while row is not None:
      self.tree.set(row[0], row[1])
      row = self.cursor.fetchone()

  def addSample(self, o, a, r, oNext):
    priority = self.tree.max()
    self.cursor.execute('insert into pool(priority, data) values(?, ?)', (priority, pickle.dumps((o, a, r, oNext))))
    self.tree.set(self.cursor.lastrowid, priority)

  def sample(self):
    id = self.tree.sample()
    self.cursor.execute('select data from pool where id=?', (id,))
    return Sample(self, id, self.cursor.fetchone()[0])

  def size(self):
    return self.tree.nLeaf()

  def sum(self):
    return self.tree.sum()

  def max(self):
    return self.tree.max()

  def commit(self):
    self.db.commit()
