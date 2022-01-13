import time, numpy as np, os
import torch
import torch.nn as nn
import torch.nn.functional as F

import dm_control_human
env = dm_control_human.balance_off_line()

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

# network
network_world_sim = MLP(6,4,hidden_sizes=(256,256),activation=F.elu)
network_robot_ctrl = MLP(6,2,hidden_sizes=(256,256),activation=F.elu)

if(0):
    network_robot_ctrl = torch.load(
        os.getcwd()+"\\learning_data\\network_robot_ctrl.pt", 
        map_location=torch.device('cpu')
    )
    network_world_sim = torch.load(
        os.getcwd()+"\\learning_data\\network_world_sim.pt", 
        map_location=torch.device('cpu')
    )

# loss
criterion_world = nn.L1Loss().cuda()
criterion_robot = nn.L1Loss().cuda()

# optimizer
optimizer_world = torch.optim.RAdam(network_world_sim.parameters(), lr = 0.001)
optimizer_robot = torch.optim.RAdam(network_robot_ctrl.parameters(), lr = 0.0001)

def learning(
    network_world_sim,network_robot_ctrl,
    criterion_world,criterion_robot,
    optimizer_world,optimizer_robot,
    www
):   

    ##########################################################################
    window_size = 8
    window_size_controller = window_size * 4
    max_time_step = 128*window_size
    length_time_step = 16
    target_radian = 0.15
    angle_limit = 0.32
    ##########################################################################
    # 피직스엔진 관측 값을 폴리시 네트워크에 넣고 제어값을 피직스에 넣음
    ##########################################################################
    if(1):
        # 컨트롤 네트워크를 cpu로
        network_robot_ctrl = network_robot_ctrl.cpu()
        # 
        pose_physix_prv = torch.Tensor()
        pose_physix = torch.Tensor()
        ctrl_neural = torch.Tensor()
        #
        all_time_step_count = 0
        #
        noise = torch.Tensor((np.random.rand(max_time_step,1,2)-0.5)*0.1)
        #
        while not (all_time_step_count>=max_time_step):
            # simulator init
            env.reset()
            env.humanoid_init()
            #
            motor_ctrl = torch.Tensor([[0,0]])
            #
            for c in range(0,length_time_step):
                #
                for d in range(0,window_size*4):
                    # 물건의 조인트 각을 받아서 신경망에
                    pose_physix_prv_buffer = torch.Tensor([env.humanoid_pose_and_delta()])
                    pose_physix_prv = torch.cat((pose_physix_prv,pose_physix_prv_buffer),dim=0)
                    # 로봇 제어기가 내뱉은 값을 적분하여 버퍼에 쌓음
                    motor_ctrl_delta = network_robot_ctrl(torch.Tensor([env.humanoid_to_network()]))
                    motor_ctrl += motor_ctrl_delta + noise[all_time_step_count]
                    motor_ctrl = torch.clip(motor_ctrl,-1,1)
                    ctrl_neural = torch.cat((ctrl_neural,motor_ctrl),dim=0)
                    # physix를 다음 스텝으로
                    env.set_control(motor_ctrl.detach().numpy()[0])
                    env.step()
                    env.humanoid_delta_calc()
                    # 버퍼에 쌓음 다음 피직스 값
                    pose_physix_buffer = torch.Tensor([env.humanoid_pose_and_delta()])
                    pose_physix = torch.cat((pose_physix,pose_physix_buffer),dim=0)
                # 총 타임스텝의 갯수 측정
                all_time_step_count += window_size*4
                if(all_time_step_count>=max_time_step):
                    break
                a = env.humanoid_pose()
                if(  (abs(a[0])>angle_limit)  |  (abs(a[1])>angle_limit)  ):
                    break
    ##########################################################################

    ##########################################################################
    # world 학습
    ##########################################################################
    if(1):
        #
        network_world_sim = network_world_sim.cuda()  
        #
        AAAA, BBBB = torch.Tensor().cuda(), torch.Tensor().cuda()
        #
        _pose_physix_prv = pose_physix_prv.clone().view([-1,window_size,4]).transpose(1, 0).cuda()
        _pose_physix = pose_physix.clone().view([-1,window_size,4]).transpose(1, 0).cuda()
        _ctrl_neural = ctrl_neural.clone().view([-1,window_size,2]).transpose(1, 0).cuda()
        # start of window and bring a physix_simulator start index value
        _pose_neural = _pose_physix_prv[0].cuda()
        for f in range(0,window_size):
            _input_buffer = torch.cat((_pose_neural,_ctrl_neural[f]),dim=1)
            _pose_neural_delta = network_world_sim(_input_buffer)
            _pose_neural += _pose_neural_delta
            AAAA = torch.cat((AAAA,_pose_physix[f]))
            BBBB = torch.cat((BBBB,_pose_neural))
        # 신경망 옵티마이즈
        optimizer_world.zero_grad()
        cost = criterion_world(AAAA,BBBB)
        cost.backward(retain_graph=True )
        torch.nn.utils.clip_grad_norm_(network_world_sim.parameters(),0.5)
        optimizer_world.step()
        print(  "world : ",  cost,  AAAA[16*8-1],  BBBB[16*8-1]  )
    ##########################################################################

    ##########################################################################
    # controller 학습
    ##########################################################################
    if(1):
        #
        network_world_sim = network_world_sim.cuda()  
        # controller window size need 32 timestep
        _window_size = window_size_controller
        #
        AAAA, BBBB = torch.Tensor().cuda(), torch.Tensor().cuda()
        #
        _pose_physix_prv = pose_physix_prv.clone().view([-1,_window_size,4]).transpose(1, 0).cuda()
        _pose_physix = pose_physix.clone().view([-1,_window_size,4]).transpose(1, 0).cuda()
        _ctrl_neural = ctrl_neural.clone().view([-1,_window_size,2]).transpose(1, 0).cuda()
        # 추론 진행
        _pose_neural = _pose_physix_prv[0].cuda()
        dsadasdasdsa = torch.zeros(_pose_neural.shape).cuda()+target_radian
        for f in range(0,_window_size):
            _input_buffer = torch.cat((_pose_neural,_ctrl_neural[f]),dim=1)
            _pose_neural_delta = network_world_sim(_input_buffer)
            _pose_neural += _pose_neural_delta
            AAAA = torch.cat((AAAA,dsadasdasdsa[0:,0:2]))
            BBBB = torch.cat((BBBB,_pose_neural[0:,0:2]))
        # 신경망 옵티마이즈
        optimizer_robot.zero_grad()
        cost = criterion_robot(AAAA, BBBB)
        cost.backward(retain_graph=True )
        #torch.nn.utils.clip_grad_norm_(network_robot_ctrl.parameters(),0.5)
        optimizer_robot.step()
        print(  "pose : ",  cost,  AAAA[16*8-1],  BBBB[16*8-1]  )
    ##########################################################################
    
    ##########################################################################
    # model save
    ##########################################################################
    if(0):
        torch.save(network_robot_ctrl, os.getcwd()+"/learning_data/network_robot_ctrl.pt")
        torch.save(network_world_sim, os.getcwd()+"/learning_data/network_world_sim.pt")
    ##########################################################################
    print(www,"=============================")
    return 0
    ##########################################################################

for w in range(0,1000000):

    learning(
        network_world_sim,network_robot_ctrl,
        criterion_world,criterion_robot,
        optimizer_world,optimizer_robot,
        w
    )