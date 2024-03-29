import torch
import numpy as np
from models.explainer import MolDQN
from models.explainer.ReplayMemory import ReplayMemory
from torch.nn import functional as F


class Agent(object):
    def __init__(self, num_input, num_output, device, lr, replay_buffer_size):

        self.device = device
        self.num_input = num_input
        self.num_output = num_output

        self.dqn, self.target_dqn = (MolDQN(num_input,
                                            num_output).to(self.device),
                                     MolDQN(num_input,
                                            num_output).to(self.device))

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayMemory(replay_buffer_size)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

    def action_step(self, observations, epsilon_threshold):
        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn.forward(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).detach().numpy()

        return action

    def train_step(self, batch_size, gamma, polyak):

        self.optimizer.zero_grad()
        experience = self.replay_buffer.sample(batch_size)
        states_ = torch.stack([S for S, *_ in experience])

        next_states_ = [S for *_, S, _ in experience]
        q_value = self.dqn(states_).reshape((1, batch_size))

        q_target = torch.stack([
            self.target_dqn(S.to(self.device)).max(dim=0).values.detach() for S in next_states_
        ]).reshape((1, batch_size)).to(self.device)

        rewards = torch.stack([R for _, R, *_ in experience]).reshape(
            (1, batch_size)).to(self.device)
        dones = torch.tensor([D for *_, D in experience]).reshape(
            (1, batch_size)).to(self.device)

        q_target = rewards + gamma * (1 - dones) * q_target

        loss = F.smooth_l1_loss(q_target, q_value, reduction="mean")

        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.dqn.parameters(),
                                           self.target_dqn.parameters()):
                target_param.data.mul_(polyak)
                target_param.data.add_((1 - polyak) * param.data)

        return loss
