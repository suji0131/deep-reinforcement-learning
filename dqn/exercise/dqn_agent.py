import time
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
A_START = 0.01          # When a=0, uniform random sampling. And when a=1, pure priorities
B_START = 0.0           # increase b from a low value to one over time.
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

        # Hyper-parameters
        self.gamma = GAMMA  # discount factor
        # DQN improvement parameters
        self.e = 0.2        # prioritized experience replay sample priority parameter.
        self.a = A_START    # prioritized experience replay sample priority parameter.
        self.b = B_START    # prioritized experience replay sample weight parameter.

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY step)
        self.t_step = 0
        self.step_counter = 1

    def _set_q_network(self):
        return QNetwork(self.state_size, self.action_size, self.seed).to(device)

    def _get_sample_priority(self, state, action, reward, next_state, done):
        """Gets priority for a sample for prioritized experience replay."""
        # convert inputs to torch vectos
        states = torch.from_numpy(np.vstack([state])).float().to(device)
        actions = torch.from_numpy(np.vstack([action])).long().to(device)
        rewards = torch.from_numpy(np.vstack([reward])).float().to(device)
        next_states = torch.from_numpy(np.vstack([next_state])).float().to(device)
        dones = torch.from_numpy(np.vstack([done])).float().to(device)

        # get qpi and qhat for the sample
        q_pi, q_hat = self._get_qpi_qhat(
            states, actions, rewards, next_states, dones,
            no_grad=True
        )
        # calculate priority
        priority = torch.abs(torch.subtract(q_pi, q_hat)).cpu().data.numpy()
        priority = (priority + self.e)
        return priority[0][0]  # since it is only for one element.
    
    def step(self, state, action, reward, next_state, done):
        # calculate sample priority
        sample_priority = self.e
        # self._get_sample_priority(state, action, reward, next_state, done)

        # Save experience in replay memory
        self.memory.add(
            state, action, reward, next_state, done,
            sample_priority
        )
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(aparam=self.a, bparam=self.b)
                self.learn(experiences)

    def get_action(self, state, eps=0.):
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

    def _get_qpi_qhat(self, states, actions, rewards, next_states, dones, no_grad=False):
        """Returns
        q_pi: real action-value function estimate using Fixed Q-target.
        and
        q_hat: estimated action-value function estimate.
        """
        # TD-target
        # real action-value function estimate
        # q_pi = reward + gamma * max_a q(S', a, w-)
        # if done, max_a q(S', a, w-) = 0, that's why we used (1-dones) in below expression.
        q_pi = rewards + (
                    self.gamma *
                    self.q_network_target.forward(next_states).detach().max(1)[0].unsqueeze(1) *
                    (1 - dones)
        )

        # estimated action-value function estimate
        # q_hat = q(S, A, w)
        # q_hat(S, A, w), that's why we used gather(1, actions) in below expression.
        if no_grad:  # don't need this for q_network_target (above) as those weights are only updated.
            self.q_network_local.eval()
            with torch.no_grad():
                q_hat = self.q_network_local.forward(states).gather(1, actions)
            self.q_network_local.train()
        else:
            q_hat = self.q_network_local.forward(states).gather(1, actions)
        return q_pi, q_hat

    def _get_sample_priorities(self, q_pi, q_hat):
        """Gets priority for samples for prioritized experience replay."""
        # calculate priority
        priority = torch.abs(torch.subtract(q_pi, q_hat)).cpu().data.numpy()
        priority = (priority + self.e)
        return priority.flatten()

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (S, A, R, S', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, sample_weights, sample_indices = experiences

        # convert to torch tensors
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        sample_weights = torch.from_numpy(sample_weights).float().to(device)

        # TD-target
        # get q_pi (real action-value function estimate) and q_hat (estimated action-value function estimate)
        # q_pi, q_hat = self._get_qpi_qhat(states, actions, rewards, next_states, dones)

        # TD-target
        # real action-value function estimate
        # q_pi = reward + gamma * max_a q(S', a, w-)
        # if done, max_a q(S', a, w-) = 0, that's why we used (1-dones) in below expression.
        q_pi = rewards + (
                self.gamma *
                self.q_network_target.forward(next_states).detach().max(1)[0].unsqueeze(1) *
                (1 - dones)
        )

        # estimated action-value function estimate
        # q_hat = q(S, A, w)
        # q_hat(S, A, w), that's why we used gather(1, actions) in below expression.
        q_hat = self.q_network_local.forward(states).gather(1, actions)

        # loss function
        # in pytorch loss we send output (predictions) and target (truth)
        # so, q_hat is first argument.
        loss = F.mse_loss(q_hat, q_pi, reduction="none")
        loss = torch.mean(loss*sample_weights)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        soft_update(self.q_network_local, self.q_network_target, TAU)

        # update the priorities in the memory buffer.
        #sample_priorities = self._get_sample_priorities(q_pi, q_hat)
        #self.memory.update_priorities(sample_indices, sample_priorities)

    def update_hyperparameter(self, a, b):
        """Updates the hyperparameter values a and b."""
        self.a = a
        self.b = b
        return 0


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """Fixed-size buffer to store experience tuples.
        Prioritized experience replay.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory: list[namedtuple] = []
        self.sample_priorities: list[float] = []
        self.batch_size: int = batch_size
        self.buffer_size: int = buffer_size  # max len of memory or sample_priorities.
        self.experience: namedtuple = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
        )
        random.seed(seed)

    def add(self, state, action, reward, next_state, done, sample_priority):
        """Add a new experience and associated priority to buffer."""
        e = self.experience(state, action, reward, next_state, done)
        if len(self.sample_priorities) >= self.buffer_size:
            self.memory.pop(0)
            self.sample_priorities.pop(0)

        self.memory.append(e)
        self.sample_priorities.append(sample_priority)

    def update_priorities(self, sample_indices, sample_priorities):
        """Updates the priorities of the samples."""
        assert len(sample_indices) == len(sample_priorities)
        for i in range(len(sample_indices)):
            idx = sample_indices[i]
            self.sample_priorities[idx] = sample_priorities[i]
    
    def sample(self, aparam, bparam):
        """Randomly sample a batch of experiences from memory."""
        # experiences = random.sample(self.memory, k=self.batch_size)
        # calculate sample selection probabilities from sample priorities.
        # raise by power of a
        _sample_priorities = [i**aparam for i in self.sample_priorities]
        # sum all elements to derive divider
        _sum = sum(_sample_priorities)
        # now calculate the probability.
        _sample_probs = []
        for i in _sample_priorities:
            _sample_probs.append(i/_sum)

        # sample experiences using calculated probabilities.
        sample_indices = np.random.choice(
            [i for i in range(len(self.memory))],
            size=self.batch_size,
            p=_sample_probs
        )
        experiences: list[namedtuple] = []
        selected_sample_probs = []
        for idx in sample_indices:
            experiences.append(self.memory[idx])
            selected_sample_probs.append(_sample_probs[idx])

        # convert to torch tensors
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        sample_weights = np.vstack([(1/(self.batch_size*i))**bparam for i in selected_sample_probs])

        return states, actions, rewards, next_states, dones, sample_weights, sample_indices

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
