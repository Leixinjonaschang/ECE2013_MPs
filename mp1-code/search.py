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

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)


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


def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0


def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0
