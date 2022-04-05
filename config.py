# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
RED1 = (255, 0, 0)
RED2 = (255, 100, 0)

# Game parameters
BLOCK_SIZE = 20
SPEED = 120

# window size
W = 1080
H = 720

# memory settings
MAX_MEMORY = 100_000
BATCH_SIZE = 1000

# learning rate schedule
def lr(n_game):
    return 0.001


# model path
model_path = None

# model parameters
gamma = 0.9 # discount rate
epsilon = 0 # parameter to control randomness
epsilon_zero = 80