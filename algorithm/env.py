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

class Physics1(mujoco.Physics):

    #
    def init(self):

        # mujoco data select array
        self.qposSelectArray = [
            #0,     #            root [ 0       ]
            #1,     #            root [ 0       ]
            #2,     #            root [ 1.28    ]
            #3,     #            root [ 1       ]
            #4,     #            root [ 0       ]
            #5,     #            root [ 0       ]
            #6,     #            root [ 0       ]
             7,     #       abdomen_z [ 0       ]
             8,     #       abdomen_y [ 0       ]
             9,     #       abdomen_x [ 0       ]
            10, 16, #     right_hip_x [ 0       ]   #      left_hip_x [ 0       ]
            11, 17, #     right_hip_z [ 0       ]   #      left_hip_z [ 0       ]
            12, 18, #     right_hip_y [ 0       ]   #      left_hip_y [ 0       ]
            13, 19, #      right_knee [ 0       ]   #       left_knee [ 0       ]
            14, 20, #   right_ankle_y [ 0       ]   #    left_ankle_y [ 0       ]
            15, 21, #   right_ankle_x [ 0       ]   #    left_ankle_x [ 0       ]
            22, 25, # right_shoulder1 [ 0       ]   #  left_shoulder1 [ 0       ]
            23, 26, # right_shoulder2 [ 0       ]   #  left_shoulder2 [ 0       ]
            24, 27, #     right_elbow [ 0       ]   #      left_elbow [ 0       ]
        ]
        self.qvelSelectArray = [
            #0,     #            root [ 0       ]
            #1,     #            root [ 0       ]
            #2,     #            root [ 0       ]
            #3,     #            root [ 0       ]
            #4,     #            root [ 0       ]
            #5,     #            root [ 0       ]
             6,     #       abdomen_z [ 0       ]
             7,     #       abdomen_y [ 0       ]
             8,     #       abdomen_x [ 0       ]
             9, 15, #     right_hip_x [ 0       ]    #      left_hip_x [ 0       ]
            10, 16, #     right_hip_z [ 0       ]    #      left_hip_z [ 0       ]
            11, 17, #     right_hip_y [ 0       ]    #      left_hip_y [ 0       ]
            12, 18, #      right_knee [ 0       ]    #       left_knee [ 0       ]
            13, 19, #   right_ankle_y [ 0       ]    #    left_ankle_y [ 0       ]
            14, 20, #   right_ankle_x [ 0       ]    #    left_ankle_x [ 0       ]
            21, 24, # right_shoulder1 [ 0       ]    #  left_shoulder1 [ 0       ]
            22, 25, # right_shoulder2 [ 0       ]    #  left_shoulder2 [ 0       ]
            23, 26, #     right_elbow [ 0       ]    #      left_elbow [ 0       ]
        ]
        self.xposSelectArray = [
            #                                 x         y         z
            #0,  1,  2, #00           world [ 0         0         0       ]
             3,  4,  5, #01           torso [ 0         0         1.28    ]
             6,  7,  8, #02            head [ 0         0         1.47    ]
             9, 10, 11, #03     lower_waist [-0.01      0         1.02    ]
            12, 13, 14, #04          pelvis [-0.01      0         0.86    ]
            15, 16, 17, #05     right_thigh [-0.01     -0.1       0.82    ]
            18, 19, 20, #06      right_shin [-0.01     -0.09      0.417   ]
            21, 22, 23, #07      right_foot [-0.01     -0.09      0.027   ]
            24, 25, 26, #08      left_thigh [-0.01      0.1       0.82    ]
            27, 28, 29, #09       left_shin [-0.01      0.09      0.417   ]
            30, 31, 32, #10       left_foot [-0.01      0.09      0.027   ]
            33, 34, 35, #11 right_upper_arm [ 0        -0.17      1.34    ]
            36, 37, 38, #12 right_lower_arm [ 0.18     -0.35      1.17    ]
            39, 40, 41, #13      right_hand [ 0.36     -0.17      1.34    ]
            42, 43, 44, #14  left_upper_arm [ 0         0.17      1.34    ]
            45, 46, 47, #15  left_lower_arm [ 0.18      0.35      1.17    ]
            48, 49, 50, #16       left_hand [ 0.36      0.17      1.34    ]
        ]
        self.xmatSelectArray = [ 
        # FieldIndexer(xmat):
        #    xx   xy   xz   yx   yy   yz   zx   zy   zz
           #  0,   1,   2,   3,   4,   5,   6,   7,   8, # 0           world 
              9,  10,  11,  12,  13,  14,  15,  16,  17, # 1           torso 
             18,  19,  20,  21,  22,  23,  24,  25,  26, # 2            head 
             27,  28,  29,  30,  31,  32,  33,  34,  35, # 3     lower_waist 
             36,  37,  38,  39,  40,  41,  42,  43,  44, # 4          pelvis 
             45,  46,  47,  48,  49,  50,  51,  52,  53, # 5     right_thigh 
             54,  55,  56,  57,  58,  59,  60,  61,  62, # 6      right_shin 
             63,  64,  65,  66,  67,  68,  69,  70,  71, # 7      right_foot 
             72,  73,  74,  75,  76,  77,  78,  79,  80, # 8      left_thigh 
             81,  82,  83,  84,  85,  86,  87,  88,  89, # 9       left_shin 
             90,  91,  92,  93,  94,  95,  96,  97,  98, #10       left_foot 
             99, 100, 101, 102, 103, 104, 105, 106, 107, #11 right_upper_arm 
            108, 109, 110, 111, 112, 113, 114, 115, 116, #12 right_lower_arm 
            117, 118, 119, 120, 121, 122, 123, 124, 125, #13      right_hand 
            126, 127, 128, 129, 130, 131, 132, 133, 134, #14  left_upper_arm 
            135, 136, 137, 138, 139, 140, 141, 142, 143, #15  left_lower_arm 
            144, 145, 146, 147, 148, 149, 150, 151, 152  #16       left_hand 
        ] 

        # 상태
        self.xposBuffer = self.get_xpos()
        self.xmatBuffer = self.get_xmat()
        self.xposVel = numpy.zeros(self.get_xpos().shape)
        self.xmatVel = numpy.zeros(self.get_xmat().shape)
        
        #
        self.allArrayValue = self.get_array()
        self.initialValue = self.get_array() # 목표로 해야 하는 값
        self.stateValue = numpy.zeros(self.allArrayValue.shape)    # 현제값과 목표값의 차이

        reward_select, done_select = self.reward_and_done_select(
            self.qposSelectArray, self.qvelSelectArray, self.xposSelectArray, self.xmatSelectArray)
        self.rewardSelectArray_a = reward_select[0]
        self.rewardSelectArray_b = reward_select[1]
        self.rewardSelectArray_c = reward_select[2]
        self.rewardSelectArray_d = reward_select[3]
        self.doneSelectArray = done_select

        # 네트워크 구조
        self.ObservationDimension = len(self.initialValue)
        self.ActionDimension = 42
        self.ActionLimit = 2

        self.ctrl_integral = numpy.concatenate([self.get_qpos(),numpy.zeros(21)])
    
    #
    def reward_and_done_select(self,qposSelect,qvelSelect,xposSelect,xmatSelect):

        qpos = []
        qvel = []
        xpos = [
           #                                  x         y         z
           # 0,  1,  2, #00           world [ 0         0         0       ]
           # 3,  4,  5, #01           torso [ 0         0         1.28    ]
             6,  7,  8, #02            head [ 0         0         1.47    ]
           # 9, 10, 11, #03     lower_waist [-0.01      0         1.02    ]
           #12, 13, 14, #04          pelvis [-0.01      0         0.86    ]
           #15, 16, 17, #05     right_thigh [-0.01     -0.1       0.82    ]
           #18, 19, 20, #06      right_shin [-0.01     -0.09      0.417   ]
            21, 22, 23, #07      right_foot [-0.01     -0.09      0.027   ]
           #24, 25, 26, #08      left_thigh [-0.01      0.1       0.82    ]
           #27, 28, 29, #09       left_shin [-0.01      0.09      0.417   ]
            30, 31, 32, #10       left_foot [-0.01      0.09      0.027   ]
           #33, 34, 35, #11 right_upper_arm [ 0        -0.17      1.34    ]
           #36, 37, 38, #12 right_lower_arm [ 0.18     -0.35      1.17    ]
            39, 40, 41, #13      right_hand [ 0.36     -0.17      1.34    ]
           #42, 43, 44, #14  left_upper_arm [ 0         0.17      1.34    ]
           #45, 46, 47, #15  left_lower_arm [ 0.18      0.35      1.17    ]
            48, 49, 50, #16       left_hand [ 0.36      0.17      1.34    ]
        ]
        xvel = xpos#[]
        xmat = [
           #  FieldIndexer(xmat):
           # xx   xy   xz   yx   yy   yz   zx   zy   zz
           #  0,   1,   2,   3,   4,   5,   6,   7,   8, # 0           world 
           #  9,  10,  11,  12,  13,  14,  15,  16,  17, # 1           torso 
             18,  19,  20,  21,  22,  23,  24,  25,  26, # 2            head 
           # 27,  28,  29,  30,  31,  32,  33,  34,  35, # 3     lower_waist 
           # 36,  37,  38,  39,  40,  41,  42,  43,  44, # 4          pelvis 
           # 45,  46,  47,  48,  49,  50,  51,  52,  53, # 5     right_thigh 
           # 54,  55,  56,  57,  58,  59,  60,  61,  62, # 6      right_shin 
             63,  64,  65,  66,  67,  68,  69,  70,  71, # 7      right_foot 
           # 72,  73,  74,  75,  76,  77,  78,  79,  80, # 8      left_thigh 
           # 81,  82,  83,  84,  85,  86,  87,  88,  89, # 9       left_shin 
             90,  91,  92,  93,  94,  95,  96,  97,  98, #10       left_foot 
           # 99, 100, 101, 102, 103, 104, 105, 106, 107, #11 right_upper_arm 
           #108, 109, 110, 111, 112, 113, 114, 115, 116, #12 right_lower_arm 
            117, 118, 119, 120, 121, 122, 123, 124, 125, #13      right_hand 
           #126, 127, 128, 129, 130, 131, 132, 133, 134, #14  left_upper_arm 
           #135, 136, 137, 138, 139, 140, 141, 142, 143, #15  left_lower_arm 
            144, 145, 146, 147, 148, 149, 150, 151, 152  #16       left_hand 
        ]
        xang = xmat#[]

        f = []
        f_xpos = []
        f_xvel = []
        f_xmat = []
        f_xang = []
        dd = []

        A_0 = len(qposSelect)
        A_1 = len(qvelSelect)
        A_2 = len(xposSelect)
        A_3 = len(xposSelect)
        A_4 = len(xmatSelect)
        
        # array making
        if(1):
            if(qpos!=[]):
                for _a in qpos:
                    f.append(qposSelect.index(_a))

            if(qvel!=[]):
                for _a in qvel:
                    f.append(A_0 + qvelSelect.index(_a))

            if(xpos!=[]):
                for _a in xpos:
                    f_xpos.append(A_0 + A_1 + xposSelect.index(_a))

            if(xvel!=[]):
                for _a in xvel:
                    f_xvel.append(A_0 + A_1 + A_2 + xposSelect.index(_a))

            if(xmat!=[]):
                for _a in xmat:
                    f_xmat.append(A_0 + A_1 + A_2 + A_3 + xmatSelect.index(_a))

            if(xang!=[]):
                for _a in xang:
                    f_xang.append(A_0 + A_1 + A_2 + A_3 + A_4 + xmatSelect.index(_a))
        
        # array making
        if(1):
            if(qpos!=[]):
                dd.append([[ 
                    qposSelect.index(_a) for _a in qpos],0])
            if(qvel!=[]):
                dd.append([[ 
                    A_0 + qvelSelect.index(_a) for _a in qvel],0])
            if(xpos!=[]):
                dd.append([[ 
                    A_0 + A_1 + xposSelect.index(_a) for _a in xpos],0.4])
            if(xvel!=[]):
                dd.append([[ 
                    A_0 + A_1 + A_2 + xposSelect.index(_a) for _a in xvel],0.2])
            if(xmat!=[]):
                dd.append([[ 
                    A_0 + A_1 + A_2 + A_3 + xmatSelect.index(_a) for _a in xmat],0.4])
            if(xang!=[]):
                dd.append([[ 
                    A_0 + A_1 + A_2 + A_3 + A_4 + xmatSelect.index(_a) for _a in xang],0.2])

        return ( f_xpos, f_xvel, f_xmat, f_xang ), dd
    
    #
    def get_qpos(self):
        return self.data.qpos[self.qposSelectArray]
    def get_qvel(self):
        return self.data.qvel[self.qvelSelectArray]
    def get_xpos(self):
        #return (self.data.xpos-self.data.xpos[1]).reshape((-1))[self.xposSelectArray]
        return (self.data.xpos).reshape((-1))[self.xposSelectArray]
    def get_xvel(self):
        return self.xposVel
    def get_xmat(self):
        #root_rotation = numpy.linalg.inv(self.data.xmat[1].reshape(3,3))
        #a = numpy.array([
        #    root_rotation.dot(s) for s in self.data.xmat.reshape(-1,3,3)
        #])
        a = self.data.xmat
        a = a.reshape(-1)[self.xmatSelectArray]
        return a
    def get_xang(self):
        return self.xmatVel
    def get_array(self):
        return numpy.clip(numpy.concatenate([
            self.get_qpos(),
            self.get_qvel(),
            self.get_xpos(),
            self.get_xvel(),
            self.get_xmat(),
            self.get_xang(),
        ],axis=0),-3.14,3.14)

    # for step0
    def _observation(self):
        return self.allArrayValue
    def _reward(self):
        a = (
            0.15*(40*numpy.exp(self.stateValue[ self.rewardSelectArray_a].sum())) + 
            0.1 *(10*numpy.exp(self.stateValue[ self.rewardSelectArray_b].sum())) + 
            0.65*(2*numpy.exp(self.stateValue[ self.rewardSelectArray_c].sum())) + 
            0.1 *(0.1*numpy.exp(self.stateValue[ self.rewardSelectArray_d].sum()))
        )
        return max(1/a,0.000001)
    def _done(self):
        for a in self.doneSelectArray:
            for b in a[0]:
                if (self.stateValue[b]>a[1]):
                    return 1
        return 0

    # main
    def delta(self):
        self.xposVel = self.get_xpos() - self.xposBuffer
        self.xposBuffer = self.get_xpos()
        self.xmatVel = self.get_xmat() - self.xmatBuffer
        self.xmatBuffer = self.get_xmat()
    def reset0(self):
        random = (numpy.random.rand(len(self.allArrayValue))-0.5)*0.2
        self.reset()
        self.allArrayValue = self.get_array() + random
        return self.allArrayValue
    def step0(self,action):
        action += self.ctrl_integral
        self.set_control(action)
        self.step()
        self.set_control(action)
        self.step()
        self.delta()
        self.allArrayValue = self.get_array()
        # 시뮬레이션 결과값에서 정상자세를 뺌
        self.stateValue = (self.allArrayValue - self.initialValue)**2
        return self._observation(), self._reward(), self._done(), 0

    # review
    def review(self):
        return self.reset0()

class env():

    def __init__(self,select=1):

        self.inside = []

        if(select==0):
            fp = open(os.getcwd()+'/model/pole.xml', 'r')
            lines = fp.readlines()
            axml_string = ""
            for itr in lines:
                axml_string += itr
            self.inside = Physics0.from_xml_string(axml_string)
        
        if(select==1):
            fp = open(os.getcwd()+'/model/humanoid.xml', 'r')
            lines = fp.readlines()
            axml_string = ""
            for itr in lines:
                axml_string += itr
            self.inside = Physics1.from_xml_string(axml_string)

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


if __name__ == "__main__":
    for ii in range(8):
        print(env())
    #a = inside.rewardSelectArray
    #b = inside.get_xpos()#[a]
    ##print(a)
    #print(b)
    #print(inside.reset0(),len(inside.reset0()))
    #print(inside.initialValue)

