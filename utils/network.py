import torch
import torch.nn as nn


class PolicyWorld(nn.Module):
    def __init__(self, local_dim, state_dim, control_dim):
        super(PolicyWorld, self).__init__()

        # actor
        self.actor = nn.Sequential(
            nn.Linear(local_dim, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, control_dim),
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(local_dim + control_dim, 1024), nn.ELU(), nn.Linear(1024, 1024), nn.ELU(), nn.Linear(1024, state_dim)
        )
    
    # 외부에서 학습되기 때문에 생략함
    def forward(self):
        raise NotImplementedError

    def policy(self, P):
        return self.actor(P)

    def world(self, P, T):
        return self.critic(torch.concat((P, T), -1))

    def save(self, directory):
        torch.save(self.actor.state_dict(), directory + "/policy.w")
        torch.save(self.critic.state_dict(), directory + "/world.w")

    def load(self, directory):
        self.actor.load_state_dict(torch.load(directory + "/policy.w"))
        self.critic.load_state_dict(torch.load(directory + "/world.w"))
