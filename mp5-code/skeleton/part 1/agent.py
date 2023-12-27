import numpy as np
import utils
import random


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.last_state = None
        self.last_action = None
        self.last_points = 0

        self.R_plus = 1.0

        # 使用探索函数 (True) 还是ε-贪婪策略随机动作 (False)
        # 注意：如果使用ε-贪婪策略随机动作，应该重新调整Ne和C的值
        self.explore_or_random = True

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    # def act(self, state, points, dead):
    #     '''
    #     :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
    #     :param points: float, the current points from environment
    #     :param dead: boolean, if the snake is dead
    #     :return: the index of action. 0,1,2,3 indicates up,down,left,right separately
    #
    #     TODO: write your function here.
    #     Return the index of action the snake needs to take, according to the state and points known from environment.
    #     Tips: you need to discretize the state to the state space defined on the webpage first.
    #     (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)
    #
    #     '''
    #
    #     return self.actions[0]

    # def update_q_table(self, state, action, reward, new_state):
    #     """
    #     更新Q表
    #     :param state: 旧状态
    #     :param action: 执行的动作
    #     :param reward: 获得的奖励
    #     :param new_state: 新状态
    #     """
    #     learning_rate = 20
    #
    #     discrete_state = self.discretize_state(state)
    #     discrete_new_state = self.discretize_state(new_state)
    #
    #     # 更新Q表
    #     future_optimal_value = np.max(self.Q[discrete_new_state])
    #     self.Q[discrete_state][action] += learning_rate * (
    #             reward + self.gamma * future_optimal_value - self.Q[discrete_state][action])

    def act(self, state, points, dead):
        """
        选择动作并更新Q表
        """
        discrete_state = self.discretize_state(state)

        if dead:
            return None

        if self._train:
            if self.last_action is not None:
                reward = points - self.last_points  # 计算奖励
            else:
                reward = points
            # 更新Q表和N表
            self.update_q_n_table(self.last_state, self.last_action, reward, discrete_state)

        if self._train:
            if self.explore_or_random:
                # 使用探索函数选择动作
                exploration_values = self.Q[discrete_state] + np.where(self.N[discrete_state] < self.Ne, self.R_plus, 0)
                action = np.argmax(exploration_values)
            else:
                # 使用ε-贪婪策略选择动作
                if random.random() < self.epsilon():
                    action = random.choice(self.actions)
                else:
                    action = np.argmax(self.Q[discrete_state])
        else:
            action = np.argmax(self.Q[discrete_state])

        # 保存当前状态和分数用于下次更新
        self.last_state = discrete_state
        self.last_action = action
        self.last_points = points

        return action

    def update_q_n_table(self, state, action, reward, new_state):
        """
        更新Q表
        :param state: 旧状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param new_state: 新状态
        """
        # learning_rate = 0.1  # 例如：学习率

        learning_rate = self.C / (self.C + self.N[state][action])

        self.N[state][action] += 1

        future_optimal_value = np.max(self.Q[new_state])
        self.Q[state][action] += learning_rate * (reward + self.gamma * future_optimal_value - self.Q[state][action])

    def epsilon(self):
        """
        计算ε值，用于ε-贪婪策略
        """
        return max(self.Ne, 0.01) / (1 + self.C)

    def discretize_state(self, state):
        """
        离散化状态
        """
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state

        adjoining_wall_x = self.get_adjoining_wall_x(snake_head_x)
        adjoining_wall_y = self.get_adjoining_wall_y(snake_head_y)
        food_dir_x = self.get_food_dir_x(snake_head_x, food_x)
        food_dir_y = self.get_food_dir_y(snake_head_y, food_y)
        adjoining_body_top = self.check_adjoining_body(snake_head_x, snake_head_y - utils.GRID_SIZE, snake_body)
        adjoining_body_bottom = self.check_adjoining_body(snake_head_x, snake_head_y + utils.GRID_SIZE, snake_body)
        adjoining_body_left = self.check_adjoining_body(snake_head_x - utils.GRID_SIZE, snake_head_y, snake_body)
        adjoining_body_right = self.check_adjoining_body(snake_head_x + utils.GRID_SIZE, snake_head_y, snake_body)

        return (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom,
                adjoining_body_left, adjoining_body_right)

    # 辅助函数的实现
    def get_adjoining_wall_x(self, snake_head_x):
        # 实现蛇头与X轴方向墙壁的关系
        if snake_head_x == 0:  # 墙在蛇头左边相邻位置
            return 1
        elif snake_head_x == 480:  # 墙在蛇头右边相邻位置
            return 2
        else:  # 蛇头左右方向上的相邻位置无墙壁
            return 0

    def get_adjoining_wall_y(self, snake_head_y):
        # 实现蛇头与Y轴方向墙壁的关系
        if snake_head_y == 0:  # 墙在蛇头上边相邻位置
            return 1
        elif snake_head_y == 480:  # 墙在蛇头下边相邻位置
            return 2
        else:  # 蛇头上下方向上的相邻位置无墙壁
            return 0

    def get_food_dir_x(self, snake_head_x, food_x):
        # 实现食物在X轴方向上相对于蛇头的位置
        if snake_head_x > food_x:  # 食物在蛇头左边
            return 1
        elif snake_head_x < food_x:  # 食物在蛇头右边
            return 2
        else:  # 蛇头与食物在水平方向上重合
            return 0

    def get_food_dir_y(self, snake_head_y, food_y):
        # 实现食物在Y轴方向上相对于蛇头的位置
        if snake_head_y < food_y:  # 食物在蛇头上边
            return 1
        elif snake_head_y > food_y:  # 食物在蛇头下边
            return 2
        else:  # 蛇头与食物在竖直方向上重合
            return 0

    def check_adjoining_body(self, x, y, snake_body):
        # 检查指定位置(x, y)是否有蛇的身体部分
        if (x, y) in snake_body:
            return 1
        else:
            return 0
