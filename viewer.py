import numpy as np, os
import dm_control_human
import torch, cv2, time
import torch.nn as nn
import torch.nn.functional as F

#
def identity(x):
    """Return input without any change."""
    return x
class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 output_limit=1.0,
                 hidden_sizes=(256,256,256,256), 
                 activation=F.relu, 
                 output_activation=identity,
                 use_output_layer=True,
                 use_actor=False,
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer
        self.use_actor = use_actor

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        x = x.view(-1,self.input_size)
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        # If the network is used as actor network, make sure output is in correct range
        x = x * self.output_limit if self.use_actor else x   
        return x

policy = torch.load(
    os.getcwd()+"\\learning_data\\network_robot_ctrl.pt", 
    map_location=torch.device('cpu')
)

env = dm_control_human.balance_off_line()

while 1:
    policy = torch.load(
        os.getcwd()+"\\learning_data\\network_robot_ctrl.pt", 
        map_location=torch.device('cpu')
    )
    env.humanoid_init()
    env.reset()
    a = 0
    action = np.array([0.0,0.0])
    while not(a>100):
        #
        pos = torch.Tensor([env.humanoid_to_network()])
        ssssss = policy(pos).detach().numpy()[0]
        action += ssssss
        action = np.clip(action,-1,1)
        print(np.array(env.humanoid_to_network()),action)
        #
        env.set_control(action)
        env.step()
        env.humanoid_delta_calc()
        #
        img = env.render(640, 480, camera_id="fixed")
        cv2.imshow('Resized Window', img)
        cv2.waitKey(10)
        a += 1