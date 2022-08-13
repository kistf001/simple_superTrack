import numpy
import torch
import torch.nn as nn

################################## set device ##################################
# set device to cpu or cuda
device = torch.device('cpu')
#if(torch.cuda.is_available()): 
#    device = torch.device('cuda:0') 
#    torch.cuda.empty_cache()
#    print("Device set to : " + str(torch.cuda.get_device_name(device)))
#else:
#    print("Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.motion = []
        self.action = []
        self.state_physics = []
        self.state_model = []
        self.pd = []
    
    def clear(self):
        self.motion = []
        self.action = []
        self.state_physics = []
        self.state_model = []
        self.pd = []


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 256),
                        nn.ELU(),
                        nn.Linear(256, 256),
                        nn.ELU(),
                        nn.Linear(256, action_dim),
                    )

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(action_dim+state_dim, 256),
                        nn.ELU(),
                        nn.Linear(256, 256),
                        nn.ELU(),
                        nn.Linear(256, state_dim)
                    )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        return self.actor(state)
    
    def crt(self, state):
        return self.critic(state)


class SUPERTRACK:
    def __init__(
        self,
        state_dim, action_dim,
        lr_actor, lr_critic,
        actor_window, critic_window
    ):
        self.policy = ActorCritic(state_dim, action_dim, ).to(device)

        self.buffer = RolloutBuffer()

        self.optimizer_actor = torch.optim.RAdam(self.policy.actor.parameters(), lr_actor)
        self.optimizer_critic = torch.optim.RAdam(self.policy.critic.parameters(), lr_critic)

        self.loss_actor = nn.L1Loss().cpu()
        self.loss_critic = nn.L1Loss().cpu()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_window = actor_window
        self.critic_window = critic_window

        self.a = torch.Tensor(numpy.zeros(action_dim))
        self.b = torch.tensor(numpy.ones(action_dim)*0.2)

    def select_action(self, state):
        noise = torch.normal(mean=self.a, std=self.b)
        state = torch.FloatTensor(state).to(device)
        # The first state buffer that works before traversing the action network.
        # After that, it works as a Critic network state buffer.
        self.buffer.state_model.append(state)
        action = self.policy.act(state) + noise
        self.buffer.action.append(action)
        return action.detach().cpu().numpy().flatten()

    def update(self):
        
        a = []
        
        agent_buffer = self.buffer

        _pd            = torch.stack(agent_buffer.pd).type(torch.float32)
        _motion        = torch.stack(agent_buffer.motion).type(torch.float32)
        _action        = torch.stack(agent_buffer.action).type(torch.float32)
        _state_physics = torch.stack(agent_buffer.state_physics).type(torch.float32)
        _state_model   = torch.stack(agent_buffer.state_model).type(torch.float32)

        ##############################################################
        # Critic
        ##############################################################
        pd            = _pd.clone().view([-1,self.critic_window,self.action_dim]).transpose(1, 0)
        motion        = _motion.clone().view([-1,self.critic_window,self.state_dim]).transpose(1, 0)
        action        = _action.clone().view([-1,self.critic_window,self.action_dim]).transpose(1, 0)
        state_physics = _state_physics.clone().view([-1,self.critic_window,self.state_dim]).transpose(1, 0)
        state_model   = _state_model.clone().view([-1,self.critic_window,self.state_dim]).transpose(1, 0)

        buffer        = state_model[0].clone()
        
        for step in range(self.critic_window):
            # CRITIC
            _data = torch.cat([buffer,action[step]+pd[step]],1)
            buffer += self.policy.critic(_data)
            state_model[step] = buffer

        self.optimizer_critic.zero_grad()
        loss = self.loss_critic(state_physics,state_model)
        loss.backward(retain_graph=True)
        self.optimizer_critic.step()
        """ Gradient averaging. """
        for param in self.policy.critic.parameters():
            a.append(param.grad.data)
        
        ##############################################################
        # Actor
        ##############################################################
        pd            = _pd.clone().view([-1,self.actor_window,self.action_dim]).transpose(1, 0)
        motion        = _motion.clone().view([-1,self.actor_window,self.state_dim]).transpose(1, 0)
        action        = _action.clone().view([-1,self.actor_window,self.action_dim]).transpose(1, 0)
        state_physics = _state_physics.clone().view([-1,self.actor_window,self.state_dim]).transpose(1, 0)
        state_model   = _state_model.clone().view([-1,self.actor_window,self.state_dim]).transpose(1, 0)
        
        buffer        = state_model[0].clone()
        
        for step in range(self.actor_window):
            # CRITIC
            _data = torch.cat([buffer,action[step]+pd[step]],1)
            buffer += self.policy.critic(_data)
            state_model[step] = buffer

        self.optimizer_actor.zero_grad()
        d = self.loss_actor(motion,state_model)
        d.backward(retain_graph=True)
        self.optimizer_actor.step()
        """ Gradient averaging. """
        for param in self.policy.actor.parameters():
            a.append(param.grad.data)
        
        return a, round(loss.detach().item(),5), round(d.detach().item(),5)

if __name__=="__main__":
    SUPERTRACK()