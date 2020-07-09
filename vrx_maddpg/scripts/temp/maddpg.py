import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
START_LEARN = 5000
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, env, n_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            env: environment class, two functions are required as follows:
                -env.get_state_dim(m)
                -env.get_action_dim(m)
            n_agents(int): number of agents
            random_seed (int): random seed
        """
        self.env = env
        self.n_agents = n_agents
        self.seed = random.seed(random_seed)

        # Actor
        self.actor_local = []
        self.actor_target = []
        self.actor_optimizer = []
        # Critic
        self.critic_local = []
        self.critic_target = []
        self.critic_optimizer = []
        # noise
        self.noise = []

        self.all_state_size = 0
        self.all_action_size = 0
        for m in range(n_agents):
            self.all_state_size += self.env.get_state_dim(m)
            self.all_action_size += self.env.get_action_dim(m)

        for m in range(n_agents):
            # Actor Network (w/ Target Network)
            state_size = self.env.get_state_dim(m)
            action_size = self.env.get_action_dim(m)
            self.actor_local.append(
                Actor(state_size, action_size, random_seed, fc1_units=400, fc2_units=300).to(device))
            self.actor_target.append(
                Actor(state_size, action_size, random_seed, fc1_units=400, fc2_units=300).to(device))
            self.actor_optimizer.append(
                optim.Adam(self.actor_local[m].parameters(), lr=LR_ACTOR))

            # Critic Network (w/ Target Network)
            self.critic_local.append(Critic(self.all_state_size, self.all_action_size, random_seed,
                                            fc1_units=400, fc2_units=300).to(device))
            self.critic_target.append(Critic(self.all_state_size, self.all_action_size, random_seed,
                                             fc1_units=400, fc2_units=300).to(device))
            self.critic_optimizer.append(
                optim.Adam(self.critic_local[m].parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY))

            # Noise process
            self.noise.append(OUNoise(action_size, random_seed))

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > START_LEARN:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        actions = np.zeros(self.all_action_size)
        state_index_s = 0
        action_index_s = 0
        for m in range(self.n_agents):
            state_index_e = state_index_s + self.env.get_state_dim(m)
            action_index_e = action_index_s + self.env.get_action_dim(m)

            self.actor_local[m].eval()
            with torch.no_grad():
                action = self.actor_local[m](state[state_index_s: state_index_e]).cpu().data.numpy()
            self.actor_local[m].train()
            if add_noise:
                action += self.noise[m].sample()
            actions[action_index_s: action_index_e] = action

            state_index_s = state_index_e
            action_index_s = action_index_e
        return np.clip(actions, -1, 1)

    def reset(self):
        for m in range(self.n_agents):
            self.noise[m].reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        for m in range(self.n_agents):
            # get the action index of agent m
            action_index_s = 0
            for i in range(m):
                action_index_s += self.env.get_action_dim(i)
            action_index_e = action_index_s + self.env.get_action_dim(m)

            # get the state index of agent m
            state_index_s = 0
            for i in range(m):
                state_index_s += self.env.get_state_dim(i)
            state_index_e = state_index_s + self.env.get_state_dim(m)

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = actions.clone()
            actions_next[:, action_index_s: action_index_e] = \
                self.actor_target[m](next_states[:, state_index_s: state_index_e])

            Q_targets_next = self.critic_target[m](next_states, actions_next)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards[:, m: m + 1] + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local[m](states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer[m].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[m].step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = actions.clone()
            actions_pred[:, action_index_s: action_index_e] = \
                self.actor_local[m](states[:, state_index_s: state_index_e])
            actor_loss = -self.critic_local[m](states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer[m].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[m].step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local[m], self.critic_target[m], TAU)
            self.soft_update(self.actor_local[m], self.actor_target[m], TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)    # change noise source random to np.random, by sw
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # change noise type from random.random to np.random.normal, by sw
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.normal() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)