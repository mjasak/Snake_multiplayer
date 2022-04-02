import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


def lr(n_game):
    return 0.001
    # if n_game < 40:
    #     return 0.001
    # elif n_game < 80:
    #     return 0.0001
    # else:
    #     return 0.00001


LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # parameter to control randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        # self.model.load_state_dict(torch.load("./model/model.pth"))
        # self.model.eval()
        self.trainer = QTrainer(self.model, lr=lr(self.n_games), gamma=self.gamma)

    def get_state(self, game, snake, otherSnake):
        head = snake.head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and snake.is_collision(otherSnake, point_r))
            or (dir_l and snake.is_collision(otherSnake, point_l))
            or (dir_u and snake.is_collision(otherSnake, point_u))
            or (dir_d and snake.is_collision(otherSnake, point_d)),
            # Danger right
            (dir_u and snake.is_collision(otherSnake, point_r))
            or (dir_d and snake.is_collision(otherSnake, point_l))
            or (dir_l and snake.is_collision(otherSnake, point_u))
            or (dir_r and snake.is_collision(otherSnake, point_d)),
            # Danger left
            (dir_d and snake.is_collision(otherSnake, point_r))
            or (dir_u and snake.is_collision(otherSnake, point_l))
            or (dir_r and snake.is_collision(otherSnake, point_u))
            or (dir_l and snake.is_collision(otherSnake, point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < snake.head.x,  # food left
            game.food.x > snake.head.x,  # food right
            game.food.y < snake.head.y,  # food up
            game.food.y > snake.head.y,  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) >= BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: trade-off
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    agent2 = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game, game.snake, game.snake2)
        state_old2 = agent2.get_state(game, game.snake2, game.snake)

        # get move
        final_move = agent.get_action(state_old)
        final_move2 = agent2.get_action(state_old2)
        moves = [final_move, final_move2]

        # perform move and get reward
        reward1, reward2, done, score = game.play_step(moves)
        state_new = agent.get_state(game, game.snake, game.snake2)
        state_new2 = agent2.get_state(game, game.snake2, game.snake)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward1, state_new, done)
        agent2.train_short_memory(state_old2, final_move2, reward2, state_new2, done)

        # remember
        agent.remember(state_old, final_move, reward1, state_new, done)
        agent2.remember(state_old2, final_move2, reward2, state_new2, done)

        if done:
            # train the long memory plot the results
            game.reset()
            agent.n_games += 1
            agent2.n_games += 1

            agent.trainer.lr = lr(agent.n_games)
            agent2.trainer.lr = lr(agent2.n_games)

            agent.train_long_memory()
            agent2.train_long_memory()

            if abs(score) > record:
                record = abs(score)
                if score > 0:
                    agent.model.save()
                elif score < 0:
                    agent2.model.save()
                # agent.model.save()
            print("Game ", agent.n_games, "\tScore: ", score, "\tRecord: ", record)
            plot_scores.append(abs(score))
            total_score += abs(score)
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
