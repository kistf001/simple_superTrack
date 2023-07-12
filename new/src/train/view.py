import time
import mujoco.viewer
import os
import numpy as np
import torch
import env
import algorithm
from network import PolicyWorld
import myQuaternion

POS = 0
VEL = 1
ROT = 2
ANG = 3

m = env.m
d = env.d

model = PolicyWorld(
        local_dim = 244, 
        state_dim = (16 * 3) + (16 * 3), 
        control_dim = env.get_info()["control_size"]
        )
model.load("./train_weight")
count = 0

def test(P_i, T):
    # Predict rigid body accelerations
    buffer = model.world(algorithm.to_serial(
            *algorithm.local(*[a.unsqueeze(-3) for a in P_i])), T.unsqueeze(-2))
    local_vel_dt = buffer[...,  0:48].reshape(1,-1,16,3)
    local_ang_dt = buffer[..., 48:96].reshape(1,-1,16,3)

    # Convert accelerations to world space
    root_mat = algorithm.myQuaternion.q_2_m(P_i[ROT][..., 0:1, :]) # 1, -1, 1, 4
    world_vel_dt = torch.einsum(
        "...ij,...jk->...ik", 
        root_mat, local_vel_dt.unsqueeze(-1)).squeeze(-1)
    world_ang_dt = torch.einsum(
        "...ij,...jk->...ik", 
        root_mat, local_ang_dt.unsqueeze(-1)).squeeze(-1)

    # Integrate rigid body accelerations
    return algorithm.integrate(*P_i, world_vel_dt, world_ang_dt, torch.tensor([0.01]))

with mujoco.viewer.launch_passive(m, d) as viewer:
    with torch.no_grad():
        # algorithm.ewqe.load_network()
        ssssss = env.init()
        wwwwww = env.init()
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < 100.1:
            step_start = time.time()

            for a in range(32):
                time.sleep(0.01)
                
                _T = algorithm.local(*ssssss)
                _T = algorithm.to_serial(*_T)
                _T = model.policy(_T) 
                _T = ((torch.normal(0.0, 1.0, _T.shape) * 0.000005) + _T) *10
                # wwwwww = test(wwwwww,_T)
                ssssss = env.step(_T)
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                # print(_T.std(),ssssss[0][0], wwwwww[0][0], )
                print(_T.std(),ssssss[0][0], wwwwww[0].shape )
            # ssssss = env.reset()
            count += 1
            if(env.is_fail()):
                print(count)
                count = 0
                ssssss = env.reset()