if __name__ == '__main__':
    import SUPERTRACK
    import env
else:
    from . import SUPERTRACK
    from . import env

import numpy as np
import torch

class Agent(object):
    def __init__(self,sample_size=4096):

        #####################################################
        self.env = env.env()
        #####################################################

        #####################################################
        # hyperparameters
        #####################################################
        self.update_timestep = sample_size       # update policy every n timesteps

        self.lr_actor = 0.0001                   # learning rate for actor network
        self.lr_critic = 0.001                   # learning rate for critic network
        
        self.max_ep_len = 1024

        self.policy_window_size = 32
        self.world_window_size = 8

        #####################################################
        # state space dimension
        #####################################################
        self.state_dim = self.env.observation_space

        #####################################################
        # action space dimension
        #####################################################
        self.action_dim = self.env.action_space

        #####################################################
        # initialize a SUPERTRACK agent
        #####################################################
        self.supertrack_agent = SUPERTRACK.SUPERTRACK(
            self.state_dim, self.action_dim, 
            self.lr_actor, self.lr_critic,
            self.policy_window_size, self.world_window_size
        )

        self.learning_counter = 0

    def run_data(self):

        agent_buffer = self.supertrack_agent.buffer
        agent_buffer.clear()

        ########################################################################
        # buffer clear
        ########################################################################
        list_current_ep_steps = []

        ########################################################################
        # training loop
        ########################################################################

        time_step = 0
        
        while time_step < self.update_timestep:

            state = self.env.reset()
            current_ep_steps = 0
            done = 0

            # 전체 트레블 길이
            while time_step < self.update_timestep:

                # 윈도우 단위로 실행
                for t in range(0, self.policy_window_size):
                    
                    # PD control Value
                    agent_buffer.pd.append(torch.from_numpy(self.env.pd()))
                    agent_buffer.motion.append(torch.from_numpy(self.env.motion()))

                    # select action with policy
                    action = self.supertrack_agent.select_action(state)
                    state, reward, _done, _ = self.env.step(action)

                    done += _done

                    # saving reward and is_terminals
                    agent_buffer.state_physics.append(torch.from_numpy(state))

                    time_step += 1
                    current_ep_steps += 1

                # break; if the episode is over
                if done or (time_step >= self.update_timestep):
                    done = 0
                    break

            list_current_ep_steps.append(current_ep_steps)

        Crit, Act = self.supertrack_agent.update()

        print(Crit, end="||- " )
        print(Act , end="||- " )
        print( np.array(list_current_ep_steps).mean(), end="||$ \n" )


    # weight management

    def param_set(self,policy):
        a = 0
        # set policy 1
        for param in self.supertrack_agent.policy.parameters():
            param.data = torch.tensor(policy[a])
            a += 1
    
    def param_get(self):
        policy = []
        # get policy weight and bias
        for param in self.supertrack_agent.policy.parameters():
            policy.append(param.detach().cpu().numpy())
        #
        return policy

    def param_export(self):
        policy = self.param_get()
        np.save('./trained_weight/policy.npy', policy, allow_pickle=True)
    
    def param_import(self):
        policy = np.load("./trained_weight/policy.npy",allow_pickle=True)
        self.param_set(policy)
