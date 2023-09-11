import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIDE = 1000

# Learning Rate
LR = 0.001

class Agent:

    def __init__(self):
        self.number_games = 0

        #epsilon is a paramet to controll randomness
        self.epsilon = 0
        
        # discount rate r
        self.gamma = 0.9

        # memory, if we exceed the memory it automatically popleft()
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        


    def get_state(self, game):
        # grabbing th head of the snake
        head = game.snake[0]

        # creating four points next to the head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # checking which direction we are currenting headings, boolean value
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # checking if the danger is straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # checking if the danger is right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # checking if the danger is left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction, boolean values
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # checking food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        # converting out list to a numpy array, converts true or false booleans to 0 and 1
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # if this exceeds max memory, we just simply pop left
        self.memory.append((state, action, reward, next_state, done))

    #training functions
    def train_long_memory(self):
        # if the memory is larger than the batch size, then we will get 
        # a random sample (mini sample)
        if len(self.memory) > BATCH_SIDE:
            mini_sample = random.sample(self.memory, BATCH_SIDE) # will return a list of tuples
        else: # if we don't have 1000 samples 
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # exploration (exploring the model, making random moves and exploring the environment)
        # exploitation (as agent gets better, we exploit the model)
        
        self.epsilon = 80 - self.number_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediciton = self.model(state0)
            move = torch.argmax(prediciton).item()
            final_move[move] = 1

        return final_move


#globabl function
def train():
    # lists to keep track of score
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get the old (current) state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform the move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memeory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory (replayed/experience memory), plot results
            game.reset()
            agent.number_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.number_games, 'Score', score, 'Record:', record)

            # plotting
            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.number_games
            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score)




    

if __name__ == '__main__':
    train()