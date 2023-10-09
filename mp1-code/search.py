# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

import sys


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
        "astar_mul": astar_mul # For execution of multi-dot A*
    }.get(searchMethod)(maze)

#############################
## BFS Search by Leixin Chang
#############################
def bfs(maze):
    # 广度优先搜索算法
    # This bfs func is implemented by Leixin Chang

    # 创建所需要的储存空间
    frontier = []  # 搜索list
    # explored = {} # 检索过的点
    explored = []  # 储存探索过的点
    parent = {}  # 通过父子对节点记录路径序列
    num_explored = 0
    goal = None
    path = []  # 储存总目标按照 parent 追踪回来的路径
    explored_filtered = []  # 储存 path 以外的 探索过的点

    # 初始化
    current_state = maze.getStart()  # 读取起点
    frontier.append(current_state)  # 将起点加入当前搜索序列
    parent[current_state] = None
    explored.append(current_state)  # 将当前点（起点）加入已被探索点集合

    # 搜索
    while frontier:  # 当搜索序列为空时 结束循环
        current_state = frontier.pop(0)  # 拿出序列第一个，并将其从搜索序列汇总删除掉
        # 先进先出保证了"一层一层搜索"，从而保证了最优性
        current_neighbors = maze.getNeighbors(current_state[0], current_state[1])  # 获取当前点的相邻点（小于等于4）

        # 遍历邻近点
        for neighbor in current_neighbors:
            if neighbor not in explored:  # 如果该点未被搜索过
                frontier.append(neighbor)
                # explored[neighbor] = True # 将该点标记为 explored
                explored.append(neighbor)
                parent[neighbor] = current_state  # 标记该点的父子对
                num_explored += 1

                if maze.isObjective(neighbor[0], neighbor[1]):  # 检查是否为目标点
                    goal = neighbor
                    break

    # 从终点开始回溯
    child = goal
    while parent[child] is not None:
        path.append(child)  # 从 goal 往 start 追踪，但是goal的索引在前，start 在后， 后续可以反转。
        child = parent[child]

    path.append(maze.getStart())  # 手动添加起点
    path.reverse()  # 将goal在前，start在后的情况反转一下

    # 提取出遍历过的点，对于用字典储存点来说
    # for key in explored:
    #     if explored[key]:
    #         if key not in path: # 排除掉在 path中的点
    #             explored_points.append(key)
    for i in range(len(explored)):
        # print(explored[i])
        if explored[i] not in path:
            explored_filtered.append(explored[i])

    # print(explored)
    return path, num_explored
    # return path, num_explored, explored_filtered

###########################
## DFS Search by Chuhan Sun
###########################
def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    explored = list()
    start=maze.getStart()
    frontier = [(start, [start])]  # 使用栈来实现深度优先搜索
    while frontier:
        curr, path = frontier.pop()
        if maze.isObjective(curr[0], curr[1]):
            return path, len(explored)
        current_neighbors = maze.getNeighbors(curr[0], curr[1])
        for neighbor in current_neighbors:
            if neighbor not in explored:
                explored.append(neighbor)
                frontier.append((neighbor, path + [neighbor]))
    return [],0

#############################
## Greedy Search by Wentao Ni
#############################
def greedy(maze):
    # TODO: Write your code here
    path = []
    explored = []
    record_last_step = {}
    num_states_explored = 0
    step = 0
    obj = maze.getObjectives()[0]
    coord = maze.getStart()
    num_states_explored += 1
    explored.append(coord)
    while explored != []:
        choice = maze.getNeighbors(coord[0],coord[1])
        #遇到死路或鬼打墙，回溯至上一个有未探索路径的节点
        if isAllExplored(choice,explored):
            for i in range(0,num_states_explored-1):
                checkpoint = explored[-i]
                check = maze.getNeighbors(checkpoint[0],checkpoint[1])
                if not isAllExplored(check,explored):
                    coord = checkpoint
                    choice = check
                    explored.append(coord)
                    break
        move = select(choice,obj,explored)
        num_states_explored += 1
        coord = move
        explored.append(move)
        if maze.isObjective(move[0],move[1]):
            break
    #生成path
    for j in explored:
        if not j in record_last_step:
            step += 1
            path.append(j)
            record_last_step[j] = step
        #原路返回时，筛除重复路径
        else:
            step = record_last_step[j]
            del path[step:]
    return path, num_states_explored

def heuristic(item,obj):
    # Manhattan Distance
    d = abs(obj[0]-item[0])+abs(obj[1]-item[1])
    # Euclidean DIstance
    # d = math.sqrt(1.0*((obj[0]-item[0])**2+(obj[1]-item[1])**2))
    return d

def select(choice,obj,explored):
    min_value = sys.maxsize
    min_move = None
    for i in choice:
        v = heuristic(i,obj)
        if i in explored:
            v += 3
        if v < min_value:
            min_value = v
            min_move = i
    return min_move

def isAllExplored(choice,explored):
    flag = True
    for i in choice:
        if i not in explored:
            flag = False
    return flag


#######################################
## A* Single-dot Search by Leixin Chang
#######################################

def astar(maze): # The single-dot A* implemented by Leixin.
    # initialization
    frontier = []  # open_list
    explored = []  # closed_list
    path = []  # 最终路径
    parent = {}  # 记录父子节点关系
    evaluation = {}  # evaluation function
    cost = {}  # 记录每个点的cost 点:cost_value

    start_point = maze.getStart()  # 获取起点
    goal_points = maze.getObjectives()
    goal_point = goal_points[0]

    cost[start_point] = 0  # 设置起点的点cost 为0 （优先级最高）
    evaluation[start_point] = 0  # 设置优先级为0
    parent[start_point] = None  # 设置起点没有父点
    frontier.append(start_point)

    current_point = None
    num_explored = 0  # 搜索次数计数器

    while frontier:
        evaluation_mini = sys.maxsize
        # 找出 frontier 中优先级最高的点，并赋给current_point
        for point in frontier:
            if evaluation[point] < evaluation_mini:
                evaluation_mini = evaluation[point]
                current_point = point

        # 如果当前点是终点
        if current_point[0] == goal_point[0] and \
                current_point[1] == goal_point[1]:
            # 回溯路径
            child = current_point
            while parent[child] is not None:
                path.append(child)  # 从 goal 往 start 追踪，但是goal的索引在前，start 在后， 后续可以反转。
                child = parent[child]

            path.append(maze.getStart())  # 手动添加起点
            path.reverse()  # 将goal在前，start在后的情况反转一下
            return path, num_explored  # return放在程序出口

        # 如果当前点不是终点
        else:
            frontier.remove(current_point)  # 如果该点不是终点， 从 frontier 去掉这个点
            explored.append(current_point)  # 将该点加入 explored

            current_neighbors = maze.getNeighbors(current_point[0], current_point[1])  # 获取当前点的相邻点
            for neighbor in current_neighbors:  # 遍历相邻点
                if neighbor in explored:
                    continue
                if neighbor not in frontier:
                    num_explored += 1
                    parent[neighbor] = current_point
                    cost[neighbor] = cost[current_point] + \
                                     abs(neighbor[0] - current_point[0]) + \
                                     abs(neighbor[1] - current_point[1])
                    evaluation[neighbor] = cost[neighbor] + \
                                           heuristic_calc_manhattan(neighbor, goal_point) # 使用曼哈顿距离的启发函数
                    frontier.append(neighbor)

# 计算当前阶段的启发函数 f(n) = g(n) + h(n) (对角距离)
def heuristic_calc_manhattan(current_point, current_goal):
    weight = 1
    dx = abs(current_point[0] - current_goal[0])
    dy = abs(current_point[1] - current_goal[1])
    h = weight * (dx + dy)  # 使用对角距离，并且添加权重
    return h


######################################
## A* Multi-dot Search by Leixin Chang
######################################

def astar_mul(maze): # The multi-dot A* search implemented by Leixin.

    start_point = maze.getStart()
    goal_points = maze.getObjectives()
    path_total = []
    num_explored_total = 0

    while goal_points:
        goal_point = get_closest_goal(start_point, goal_points)
        print(goal_point)
        path, num_explored = astar_single(maze, start_point, goal_point)
        start_point = goal_point # 将上次的goal_point 变换为star_point
        goal_points.remove(goal_point)

        path_total += path
        path_total.insert(0, maze.getStart())
        num_explored_total += num_explored

    return path_total, num_explored_total

# A* Single-dot Search with Diagonal Distance
def astar_single(maze,start_point,goal_point):
    # initialization
    frontier = []  # open_list
    explored = []  # closed_list
    path = []  # 最终路径
    parent = {}  # 记录父子节点关系
    evaluation = {}  # evaluation function
    cost = {}  # 记录每个点的cost 点:cost_value

    # start_point = maze.getStart()  # 获取起点
    # goal_points = maze.getObjectives()
    # goal_point = goal_points[0]

    cost[start_point] = 0  # 设置起点的点cost 为0 （优先级最高）
    evaluation[start_point] = 0  # 设置优先级为0
    parent[start_point] = None  # 设置起点没有父点
    frontier.append(start_point)

    current_point = None
    num_explored = 0  # 搜索次数计数器

    while frontier:
        evaluation_mini = sys.maxsize
        # 找出 frontier 中优先级最高的点，并赋给current_point
        for point in frontier:
            if evaluation[point] < evaluation_mini:
                evaluation_mini = evaluation[point]
                current_point = point

        # 如果当前点是终点
        if current_point[0] == goal_point[0] and \
                current_point[1] == goal_point[1]:
            # 回溯路径
            child = current_point
            while parent[child] is not None:
                path.append(child)  # 从 goal 往 start 追踪，但是goal的索引在前，start 在后， 后续可以反转。
                child = parent[child]

            # path.append(start_point)  # 手动添加起点
            path.reverse()  # 将goal在前，start在后的情况反转一下
            return path, num_explored  # return放在程序出口

        # 如果当前点不是终点
        else:
            frontier.remove(current_point)  # 如果该点不是终点， 从 frontier 去掉这个点
            explored.append(current_point)  # 将该点加入 explored

            current_neighbors = maze.getNeighbors(current_point[0], current_point[1])  # 获取当前点的相邻点
            for neighbor in current_neighbors:  # 遍历相邻点
                if neighbor in explored:
                    continue
                if neighbor not in frontier:
                    num_explored += 1
                    parent[neighbor] = current_point
                    cost[neighbor] = cost[current_point] + \
                                     abs(neighbor[0] - current_point[0]) + \
                                     abs(neighbor[1] - current_point[1])
                    evaluation[neighbor] = cost[neighbor] + \
                                           heuristic_calc_diagonal(neighbor, goal_point)
                    frontier.append(neighbor)

# Find the closest goal of the goal list
def get_closest_goal(current_point, goals):
    closest_goal = None
    closest_dis = sys.maxsize

    for goal in goals:
        dx = abs(current_point[0] - goal[0])
        dy = abs(current_point[1] - goal[1])
        distance = 1.414 * min(dx, dy) + abs(dx - dy)  # 当前点到目标点的对角距离
        if distance < closest_dis:
            closest_goal = goal

    return closest_goal  # 将最近目标返回

# Heuristics Based on Diagonal Distance
def heuristic_calc_diagonal(current_point, current_goal):
    weight = 1 # 启发h(n)的权重,经过测试，发现权重在4.5 时候 state explored 最少, 但是不符合admissible 的原则
    dx = abs(current_point[0] - current_goal[0])
    dy = abs(current_point[1] - current_goal[1])
    h = weight * (1.414 * min(dx, dy) + abs(dx - dy))  # 使用对角距离，并且添加权重
    return h
