from dm_control import mujoco
import os
import numpy

class Physics0(mujoco.Physics):
    
    def init(self):
        #
        self.ObservationDimension = 4
        self.ActionDimension = 2
        self.ActionLimit = 1

        self.ctrl_integral = numpy.zeros(self.ActionDimension)
        self.allArrayValue = self._observation()
        self.initialValue = self._observation() # 목표로 해야 하는 값

    def _observation(self):
        return numpy.concatenate([self.data.qpos,self.data.qvel],0)
    def _reward(self):
        return 1-numpy.absolute(self._observation()).sum()
    def _done(self):
        return int(numpy.any(numpy.absolute(self.data.qpos)>0.2))

    def reset0(self):
        self.reset()
        self.allArrayValue = self._observation()
        return self.allArrayValue
    def step0(self,action):
        action += self.ctrl_integral
        self.set_control(action)
        self.step()
        self.allArrayValue = self._observation()
        return self.allArrayValue, self._reward(), self._done(), 0

    # review
    def review(self):
        return self.reset0()

class env():

    def __init__(self,select=0):

        self.inside = []

        if(select==0):
            fp = open(os.getcwd()+'/model/pole.xml', 'r')
            lines = fp.readlines()
            axml_string = ""
            for itr in lines:
                axml_string += itr
            self.inside = Physics0.from_xml_string(axml_string)

        self.inside.init()

        self.observation_space = self.inside.ObservationDimension
        self.action_space = self.inside.ActionDimension
    
    def render(self):
        return self.inside.render(640, 480, camera_id=0)

    def reset(self):
        return self.inside.reset0()

    def step(self, a):
        return self.inside.step0(a)
    
    def review(self):
        return self.inside.review()
    
    def motion(self):
        return self.inside.initialValue
        
    def pd(self):
        return self.inside.ctrl_integral

