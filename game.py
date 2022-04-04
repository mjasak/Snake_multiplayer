import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font("arial.ttf", 25)


# font = pygame.font.SysFont('arial', 25)

# reset
# reward
# play(action) -> direction
# game_iteration?
# is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")
import config

# rgb colors
WHITE = config.WHITE
RED = config.RED
BLUE1 = config.BLUE1
BLUE2 = config.BLUE2
BLACK = config.BLACK
RED1 = config.RED1
RED2 = config.RED2
BLOCK_SIZE = config.BLOCK_SIZE
SPEED = config.SPEED
W = config.W
H = config.H


class Snake:
    def __init__(self, posx, posy, direction, w=W, h=H):
        self.head = Point(posx, posy)
        self.body = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.direction = direction
        self.w = w
        self.h = h
        self.color1 = BLUE1
        self.color2 = BLUE2
        self.score = 0
        self.game_over = False
        self.reward = 0
        self.col_with_enemy = False

    def move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0,0,1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir
        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x = (x + BLOCK_SIZE) % self.w
        elif self.direction == Direction.LEFT:
            x = (x - BLOCK_SIZE) % self.w
        elif self.direction == Direction.DOWN:
            y = (y + BLOCK_SIZE) % self.h
        elif self.direction == Direction.UP:
            y = (y - BLOCK_SIZE) % self.h
        self.head = Point(x, y)
        self.body.insert(0, self.head)

    def set_color(self, color1, color2):
        self.color1 = color1
        self.color2 = color2

    def is_collision(self, other, pt=None):
        flag = False
        if pt is None:
            pt = self.head
            flag = True
        if not self.game_over:
            if pt in self.body[1:]:
                if flag:
                    self.game_over = True
                return True
        if not other.game_over:
            if pt in other.body[1:]:
                if flag:
                    self.game_over = True
                    self.col_with_enemy = True
                return True
        return False


class SnakeGameAI:
    def __init__(self, w=W, h=H):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.snake = Snake(self.w / 2, 3 * self.h / 4, Direction.RIGHT)
        self.snake2 = Snake(self.w / 2, self.h / 4, Direction.RIGHT)
        self.snake2.set_color(RED1, RED2)

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if not self.snake.game_over:
            if self.food in self.snake.body:
                self._place_food()
        if not self.snake2.game_over:
            if self.food in self.snake2.body:
                self._place_food()

    def play_step(self, actions):
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        # self._move(action)  # update the head
        # self.snake.insert(0, self.head)

        if not self.snake.game_over:
            self.snake.move(actions[0])
        if not self.snake2.game_over:
            self.snake2.move(actions[1])

        # 3. check if game over
        reward = 0
        self.snake.reward = 0
        self.snake2.reward = 0

        game_over = False
        if not self.snake.game_over:
            if self.snake.is_collision(self.snake2):
                self.snake.body = None
                self.snake.reward = -10
                if self.snake.col_with_enemy:
                    self.snake.reward = -20
                print("snake 1 collision: ", self.snake.reward, self.snake2.reward)

        if not self.snake2.game_over:
            if self.snake2.is_collision(self.snake):
                self.snake2.body = None
                self.snake2.reward = -10
                if self.snake2.col_with_enemy:
                    self.snake2.reward = -20
                print("snake 2 collision: ", self.snake.reward, self.snake2.reward)

        if not self.snake.game_over and not self.snake2.game_over:
            limit = 100 * max(len(self.snake.body), len(self.snake2.body))
        elif not self.snake.game_over:
            limit = 100 * len(self.snake.body)
        elif not self.snake2.game_over:
            limit = 100 * len(self.snake2.body)

        if (
            self.snake.game_over
            and self.snake2.game_over
            or self.frame_iteration > limit
        ):
            game_over = True
            if self.snake.score > self.snake2.score:
                self.score = self.snake.score
                self.snake2.reward = -10
                print("Endgame snake 1 won: ", self.snake.reward, self.snake2.reward)

            elif self.snake.score < self.snake2.score:
                self.score = -self.snake2.score
                self.snake.reward = -10
                print("Endgame snake 2 won: ", self.snake.reward, self.snake2.reward)

            else:
                self.score = self.snake.score
            # score = self.snake.score if self.snake.score > self.snake2.score else self.snake2.score
            return self.snake.reward, self.snake2.reward, game_over, self.score

        # 4. place new food or just move
        if not self.snake.game_over:
            if self.snake.head == self.food:
                self.snake.score += 1
                self.snake.reward = 10
                print("Snake 1 food: ", self.snake.reward, self.snake2.reward)
                self._place_food()
            else:
                self.snake.body.pop()

        if not self.snake2.game_over:
            if self.snake2.head == self.food:
                self.snake2.score += 1
                self.snake2.reward = 10
                print("Snake 2 food: ", self.snake.reward, self.snake2.reward)
                self._place_food()
            else:
                self.snake2.body.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        # print(self.snake.reward, self.snake2.reward)
        return self.snake.reward, self.snake2.reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        # if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
        #     return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        if not self.snake.game_over:
            for pt in self.snake.body:
                pygame.draw.rect(
                    self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
                )
                pygame.draw.rect(
                    self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
                )
        if not self.snake2.game_over:
            for pt in self.snake2.body:
                pygame.draw.rect(
                    self.display,
                    self.snake2.color1,
                    pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
                )
                pygame.draw.rect(
                    self.display,
                    self.snake2.color2,
                    pygame.Rect(pt.x + 4, pt.y + 4, 12, 12),
                )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render(
            "Blue: " + str(self.snake.score) + "   Red: " + str(self.snake2.score),
            True,
            WHITE,
        )

        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0,0,1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x = (x + BLOCK_SIZE) % self.w
        elif self.direction == Direction.LEFT:
            x = (x - BLOCK_SIZE) % self.w
        elif self.direction == Direction.DOWN:
            y = (y + BLOCK_SIZE) % self.h
        elif self.direction == Direction.UP:
            y = (y - BLOCK_SIZE) % self.h

        self.head = Point(x, y)


if __name__ == "__main__":
    game = SnakeGameAI()

    # game loop
    while True:
        actions = [[0, 0, 0], [0, 0, 0]]
        ind1 = random.randint(0, 2)
        ind2 = random.randint(0, 2)
        actions[0][ind1] = 1
        actions[1][ind1] = 1
        rew1, rew2, game_over, score = game.play_step(actions)
        print(rew1, rew2)
        if game_over:
            break

    print("Final Score", score)

    pygame.quit()
