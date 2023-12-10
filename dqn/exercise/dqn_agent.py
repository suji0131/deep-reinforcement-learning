import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(local_model, target_model, tau):
    """Soft update model parameters. updates w- from w
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Agent():
    def __init__(self, state_size, action_size, seed):
        """
        Interacts with and learns from the environment.
        Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(self.seed)

        # Q-Network
        self.q_network_local = self._set_q_network()
        self.q_network_target = self._set_q_network()
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY step)
        self.t_step = 0

    def _set_q_network(self):
        return QNetwork(self.state_size, self.action_size, self.seed).to(device)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()

        # Epsilon-greedy action selection
        if random.random() <= eps:  # random action with prob eps
            return random.choice(np.arange(self.action_size))
        else:  # greedy action
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (S, A, R, S', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # TD-target
        # real action-value function estimate
        # q_pi = reward + gamma * max_a q(S', a, w-)
        # if done, max_a q(S', a, w-) = 0, that's why we used (1-dones) in below expression.
        q_pi = rewards + (gamma*self.q_network_target.forward(next_states).detach().max(1)[0].unsqueeze(1)*(1-dones))

        # estimated action-value function estimate
        # q_hat = q(S, A, w)
        # q_hat(S, A, w), that's why we used gather(1, actions) in below expression.
        q_hat = self.q_network_local.forward(states).gather(1, actions)

        # loss function
        # in pytorch loss we send predictions as first argument
        # so, q_hat is first argument.
        loss = F.mse_loss(q_hat, q_pi)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        soft_update(self.q_network_local, self.q_network_target, TAU)


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """Fixed-size buffer to store experience tuples.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory: deque = deque(maxlen=buffer_size)
        self.batch_size: int = batch_size
        self.experience: namedtuple = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
        )
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
