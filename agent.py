import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from torchvision import transforms
from model import Linear_QNet, CNN_QNet, QTrainer
from helper import plot

MAX_MEMORY = 500
BATCH_SIZE = 32
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # randomness
        self.gamma = 0.9    # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        # self.model = Linear_QNet(11, 256, 3)
        self.model = CNN_QNet(256, 3)
        self.trainer= QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # return torch.from_numpy(game.state)
        transform = transforms.ToTensor()
        return transform(game.state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Tradeoff between exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        # Random move vs predicted move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
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
    # agent.model.load()
    game = SnakeGameAI()

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()