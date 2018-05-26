"""A2C agent for SC2."""
import torch
import torch.optim as optim
from agents.policy_net import SpatialPolicyNet

class A2CAgent:
    def __init__(self, screen_channels=5, screen_size=64, n_actions=5, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = SpatialPolicyNet(screen_channels, screen_size, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []
        self.values = []

    def select_action(self, screen):
        screen_t = torch.FloatTensor(screen).unsqueeze(0).to(self.device)
        action_probs, spatial_probs, value = self.policy(screen_t)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        spatial_dist = torch.distributions.Categorical(spatial_probs.view(-1))
        spatial = spatial_dist.sample()
        y, x = spatial.item() // 64, spatial.item() % 64
        log_prob = action_dist.log_prob(action) + spatial_dist.log_prob(spatial)
        self.saved_log_probs.append(log_prob)
        self.values.append(value)
        return action.item(), [y, x]

    def update(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss, value_loss = 0, 0
        for lp, val, R in zip(self.saved_log_probs, self.values, returns):
            advantage = R - val.item()
            policy_loss -= lp * advantage
            value_loss += (val - R) ** 2
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
        self.optimizer.step()
        self.saved_log_probs, self.rewards, self.values = [], [], []
        return loss.item()
