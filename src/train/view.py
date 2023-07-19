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

with mujoco.viewer.launch_passive(m, d) as viewer:
    with torch.no_grad():
        ssssss = env.init()
        start = time.time()
        while viewer.is_running() and time.time() - start < 100.1:
            step_start = time.time()
            for a in range(8):
                time.sleep(0.01)
                _T = algorithm.local(*ssssss)
                _T = algorithm.to_serial(*_T)
                _T = model.policy(_T) 
                _T = ((torch.normal(0.0, 1.0, _T.shape) * 0.0000015) + _T) * 10
                ssssss = env.step(_T)
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
            count += 1
            if(env.is_fail()):
                print(count)
                count = 0
                ssssss = env.reset()