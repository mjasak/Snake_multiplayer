import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font("arial.ttf", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
RED1 = (255, 0, 0)
RED2 = (255, 100, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 10


class Snake:
    def __init__(self, posx, posy, direction, w=640, h=480):
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

    def move(self, direction):
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

    def is_collision(self, other):
        if self.head in self.body[1:]:
            self.game_over = True
            return True
        if not other.game_over:
            if self.head in other.body[1:]:
                self.game_over = True
                return True
        return False


class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.RIGHT

        self.snake = Snake(self.w / 2, 3 * self.h / 4, Direction.RIGHT)
        self.snake2 = Snake(self.w / 2, self.h / 4, Direction.RIGHT)
        self.snake2.set_color(RED1, RED2)

        self.score = 0
        self.food = None
        self._place_food()

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

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                # SNAKE 1 MOVES
                if (
                    self.snake.direction == Direction.LEFT
                    and event.key == pygame.K_RIGHT
                ):
                    pass
                elif (
                    self.snake.direction == Direction.RIGHT
                    and event.key == pygame.K_LEFT
                ):
                    pass
                elif (
                    self.snake.direction == Direction.UP and event.key == pygame.K_DOWN
                ):
                    pass
                elif (
                    self.snake.direction == Direction.DOWN and event.key == pygame.K_UP
                ):
                    pass
                elif event.key == pygame.K_LEFT:
                    self.snake.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.snake.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.snake.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.snake.direction = Direction.DOWN

                # SNAKE 2 MOVES
                if self.snake2.direction == Direction.LEFT and event.key == pygame.K_d:
                    pass
                elif (
                    self.snake2.direction == Direction.RIGHT and event.key == pygame.K_a
                ):
                    pass
                elif self.snake2.direction == Direction.UP and event.key == pygame.K_s:
                    pass
                elif (
                    self.snake2.direction == Direction.DOWN and event.key == pygame.K_w
                ):
                    pass
                elif event.key == pygame.K_a:
                    self.snake2.direction = Direction.LEFT
                elif event.key == pygame.K_d:
                    self.snake2.direction = Direction.RIGHT
                elif event.key == pygame.K_w:
                    self.snake2.direction = Direction.UP
                elif event.key == pygame.K_s:
                    self.snake2.direction = Direction.DOWN

        # 2. move
        # self._move(self.snake.direction)  # update the head
        if not self.snake.game_over:
            self.snake.move(self.snake.direction)
        if not self.snake2.game_over:
            self.snake2.move(self.snake2.direction)
        # self.snake.body.insert(0, self.snake.head)

        # 3. check if game over
        game_over = False
        if not self.snake.game_over:
            if self.snake.is_collision(self.snake2):
                self.snake.body = None

        if not self.snake2.game_over:
            if self.snake2.is_collision(self.snake):
                self.snake2.body = None

        # print(self.snake.game_over, self.snake2.game_over)
        if self.snake.game_over and self.snake2.game_over:
            game_over = True
            # winner = "Blue" if self.snake.score > self.snake2.score else "Red"
            score = (
                self.snake.score
                if self.snake.score > self.snake2.score
                else self.snake2.score
            )
            return game_over, score

        # 4. place new food or just move
        if not self.snake.game_over:
            if self.snake.head == self.food:
                self.snake.score += 1
                self._place_food()
            else:
                self.snake.body.pop()

        if not self.snake2.game_over:
            if self.snake2.head == self.food:
                self.snake2.score += 1
                self._place_food()
            else:
                self.snake2.body.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score

    def _is_collision(self):
        # hits boundary
        # if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
        #     return True
        # hits itself
        if self.snake.head in self.snake.body[1:]:
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

    # def _move(self, direction):
    #     x = self.snake.head.x
    #     y = self.snake.head.y
    #
    #     if self.direction == Direction.RIGHT:
    #         x = (x + BLOCK_SIZE) % self.w
    #     elif self.direction == Direction.LEFT:
    #         x = (x - BLOCK_SIZE) % self.w
    #     elif self.direction == Direction.DOWN:
    #         y = (y + BLOCK_SIZE) % self.h
    #     elif self.direction == Direction.UP:
    #         y = (y - BLOCK_SIZE) % self.h
    #     self.snake.head = Point(x, y)


if __name__ == "__main__":
    game = SnakeGame()

    # game loop
    while True:
        game_over, score = game.play_step()

        if game_over:
            break

    print("Final Score", score)

    pygame.quit()
