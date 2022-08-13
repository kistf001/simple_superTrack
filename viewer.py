import numpy
from algorithm import agent1 as agent
import torch
import cv2
import time

A = agent.Agent()

A.env.render()

while(1):

    A.param_import()

    state = A.env.reset()

    for d in range(0,512):
        
        action = A.supertrack_agent.select_action(state)
        
        state, reward, done, _ = A.env.step(action)
        
        img = A.env.render()
        
        cv2.imshow('Resized Window', img)
        cv2.waitKey(1)

        if(done):
            print("=",done,d)
            break

    print("=",done,512)
    