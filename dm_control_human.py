# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Cartpole domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
#from dm_control.suite import common
#from dm_control.utils import containers
#from dm_control.utils import rewards
#from lxml import etree
import numpy as np
import os

_DEFAULT_TIME_LIMIT = 100
#SUITE = containers.TaggedTasks()

def xml_string():
    fp = open(os.getcwd()+'/humanoid_test.xml', 'r')
    lines = fp.readlines()
    axml_string = ""
    for itr in lines:
        axml_string += itr
    return axml_string

def balance_off_line():
    return Physics.from_xml_string(xml_string())

class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def humanoid_init(self):
        a = 1
        self.integral_qpos_vel_0 = 0
        self.integral_qpos_vel_1 = 0
        self.integral_qpos_0 = 0
        self.integral_qpos_1 = 0

        self.delta_qpos_acc_0 = 0
        self.delta_qpos_acc_1 = 0
        self.delta_qpos_vel_prev_0 = 0
        self.delta_qpos_vel_prev_1 = 0
        self.delta_qpos_vel_0 = 0
        self.delta_qpos_vel_1 = 0
        self.delta_qpos_prev_0 = 0
        self.delta_qpos_prev_1 = 0

        self.s = 0
        self.d = 0
        
        self.humanoid_delta_calc()
        self.step()
        self.humanoid_delta_calc()
        self.step()
        self.humanoid_delta_calc()
    def humanoid_pose_integral(self,acc_0,acc_1):
        self.integral_qpos_vel_0 += (acc_0/0xFFFF)
        self.integral_qpos_vel_1 += (acc_1/0xFFFF)
        self.integral_qpos_0 += self.integral_qpos_vel_0
        self.integral_qpos_1 += self.integral_qpos_vel_1
        return([
            self.integral_qpos_0, 
            self.integral_qpos_1
        ])
    def humanoid_pose_delta(self):

        self.delta_qpos_vel_0 = self.named.data.qpos[0] - self.delta_qpos_prev_0
        self.delta_qpos_vel_1 = self.named.data.qpos[1] - self.delta_qpos_prev_1

        self.delta_qpos_acc_0 = self.delta_qpos_vel_0 - self.delta_qpos_vel_prev_0
        self.delta_qpos_acc_1 = self.delta_qpos_vel_1 - self.delta_qpos_vel_prev_1
 
        self.delta_qpos_vel_prev_0 = self.delta_qpos_vel_0
        self.delta_qpos_vel_prev_1 = self.delta_qpos_vel_1

        self.delta_qpos_prev_0 = self.named.data.qpos[0]
        self.delta_qpos_prev_1 = self.named.data.qpos[1]

        return ([
            self.delta_qpos_vel_0,#*0xFF,
            self.delta_qpos_vel_1,#*0xFF
        ])   
    def humanoid_pose(self):
        #print(self.named.data.sensordata)
        return ([
            self.named.data.qpos[0],#*0xFF, 
            self.named.data.qpos[1],#*0xFF
        ])
    def humanoid_pose_and_delta(self):
        s = self.s
        d = self.d
        return([  s[0],s[1],  d[0],d[1]  ])
    def humanoid_to_network(self):
        s = self.s
        d = self.d
        return([  s[0],s[1],  d[0],d[1],  0,0  ])
    def humanoid_delta_calc(self):
        self.s = self.humanoid_pose()
        self.d = self.humanoid_pose_delta()

