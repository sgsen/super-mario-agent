import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import distributions

class CartPoleAgent(nn.Module): ## Policy Network, the agent that does stuff
    def __init__(self, num_inputs, hidden_dim, num_actions, dropout):
        super().__init__()
        self.layer1 = nn.Linear(num_inputs, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_actions)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

def forward_pass(env, policy, discount_factor): ## making the agent and env interact
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0
    observation, info = env.reset()

    while not done:
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_pred = policy(observation)
        action_prob = F.softmax(action_pred, dim=-1)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()

        log_prob_action = dist.log_prob(action)
        observation, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward

        return episode_return, rewards, log_prob_actions

## now the learning function
def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()
    return loss

def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



def main():
    env = gym.make("CartPole-v1", render_mode="human")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("CartPole-v1 environment loaded")


    # Test running one episode
    state, info = env.reset()
    print(state, info)
    print("Episode started")

    done = False
    
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(state, reward, done, truncated, info)
        done = done or truncated
    
    print("Episode finished")
    env.close()

if __name__ == "__main__":
    main() 