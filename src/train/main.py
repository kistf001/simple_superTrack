import torch
import torch.nn as nn
import torch.multiprocessing as mp

from network import PolicyWorld

import env
import buffer
import learningMLT
import learningSGL

if __name__ == '__main__':

    ####################################################################
    # [CYCLIC BUFFER SIZE], [SPLIT SIZE], [AGENT NUMBER]
    ####################################################################
    buffer.init(512*8*2, 8, 8)
    # buffer.init(1024, 1, 1) <- test and develop

    ####################################################################
    # Select Device
    ####################################################################
    # not working have a problem i tried only CPU
        # try:
        #     import torch_directml
        #     buffer.init_device(torch_directml.device())
        #     print("DirectML working")
        # except:
        #     pass
        
    device = buffer.get_device_type()

    ####################################################################
    # model setting
    ####################################################################
    model_master = PolicyWorld(
        local_dim = 244, 
        state_dim = (16 * 3) + (16 * 3), 
        control_dim = env.get_info()["control_size"]
        )
    model_master.to(device)
    # model_master.load("./train_weight")

    model_slave = PolicyWorld(
        local_dim = 244, 
        state_dim = (16 * 3) + (16 * 3), 
        control_dim = env.get_info()["control_size"]
        )
    model_slave.share_memory()

    model_slave.actor .load_state_dict(model_master.actor .state_dict())
    model_slave.critic.load_state_dict(model_master.critic.state_dict())

    ####################################################################
    # train setting
    ####################################################################
    L1_loss = nn.L1Loss ()
    L2_loss = nn.MSELoss()

    optim_world  = torch.optim.RAdam(model_master.critic.parameters(), 0.0001 )
    optim_policy = torch.optim.RAdam(model_master.actor .parameters(), 0.00001)

    ####################################################################
    # process type
    ####################################################################

    # SINGLE PROCESS
    if buffer.buffer_size_processor == 1:
        learningSGL.start(
            model_master, model_slave,
            L1_loss, L2_loss, 
            optim_world, optim_policy, 
            buffer, env
        )

    # MULTI PROCESS
    else :
        learningMLT.start(
            model_master, model_slave,
            L1_loss, L2_loss, 
            optim_world, optim_policy, 
            buffer, env
            )
