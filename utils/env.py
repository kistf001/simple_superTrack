import time
import mujoco
import mujoco.viewer
import os
import numpy as np
import torch
import utils.myQuaternion as myQuaternion

np.random.seed((os.getpid() * int(time.time())) % 123456789)

############################################################
#
############################################################
# FieldIndexer(geom_xpos):
#                       x         y         z
#  0            floor [ 0         0         0       ]
#  1            torso [ 0         0         1.28    ]
#  2      upper_waist [-0.01      0         1.17    ]
#  3             head [ 0         0         1.47    ]
#  4      lower_waist [-0.01      0         1.02    ]
#  5             butt [-0.0293    0         0.86    ]
#  6      right_thigh [-0.00766  -0.095     0.65    ]
#  7       right_shin [-0.0046   -0.09      0.267   ]
#  8 right_right_foot [ 0.0323   -0.12      0.0273  ]
#  9  left_right_foot [ 0.0323   -0.08      0.0273  ]
# 10       left_thigh [-0.00766   0.095     0.65    ]
# 11        left_shin [-0.0046    0.09      0.267   ]
# 12   left_left_foot [ 0.0323    0.12      0.0273  ]
# 13  right_left_foot [ 0.0323    0.08      0.0273  ]
# 14  right_upper_arm [ 0.08     -0.25      1.26    ]
# 15  right_lower_arm [ 0.27     -0.26      1.26    ]
# 16       right_hand [ 0.36     -0.17      1.34    ]
# 17   left_upper_arm [ 0.08      0.25      1.26    ]
# 18   left_lower_arm [ 0.27      0.26      1.26    ]
# 19        left_hand [ 0.36      0.17      1.34    ]
############################################################

############################################################
#
############################################################
m = mujoco.MjModel.from_xml_path(os.getcwd() + "/src/train/humanoid.xml")
d = mujoco.MjData(m)

############################################################
#
############################################################
pos_prv, quat_prv = [], []


############################################################
#
############################################################
@torch.jit.script
def step_backend(pos, quat, pos_prv, quat_prv):
    """
    input : pos, quat, pos_prv, quat_prv
    return : buffer, pos_prv, quat_prv
    """
    pos = pos.to(torch.float32)
    quat = quat.to(torch.float32)
    buffer = (
        pos.clone(),
        ((pos - pos_prv) * 100).clone(),
        quat.clone(),
        myQuaternion.quat_differentiate_angular_velocity(quat, quat_prv, torch.tensor([100])).clone(),
    )
    pos_prv = pos.clone()
    quat_prv = quat.clone()
    return buffer, pos_prv, quat_prv


############################################################
#
############################################################
def init():
    """
    return : pos, vel, rot, ang
    """
    return reset()


def reset():
    """
    return : pos, vel, rot, ang
    """
    global pos_prv, quat_prv, m, d

    mujoco.mj_resetData(m, d)
    mujoco.mj_step(m, d)

    pos_prv = torch.from_numpy(d.xpos[1:].astype(np.float32)).clone()
    quat_prv = torch.from_numpy(d.xquat[1:].astype(np.float32)).clone()

    np.copyto(d.qvel[6:], d.qvel[6:] + (np.random.rand((21)) - 0.5) * 2)

    return step()


def reset_view():
    """
    return : pos, vel, rot, ang
    """
    global pos_prv, quat_prv, m, d

    mujoco.mj_resetData(m, d)
    mujoco.mj_step(m, d)

    pos = torch.from_numpy(d.xpos[1:].astype(np.float32))
    quat = torch.from_numpy(d.xquat[1:].astype(np.float32))

    pos_prv = pos.clone()
    quat_prv = quat.clone()

    return step()


def step(ctrl=None):
    """
    arg : [42.]
    return : pos, vel, rot, ang
    """
    global pos_prv, quat_prv, m, d

    if ctrl != None:
        np.copyto(d.ctrl, ctrl.detach().numpy())

    mujoco.mj_step(m, d, 2)

    buffer, pos_prv, quat_prv = step_backend(torch.from_numpy(d.xpos[1:]), torch.from_numpy(d.xquat[1:]), pos_prv, quat_prv)
    return buffer


def is_fail():
    global d
    if d.xpos[1][2] < 1.25:
        return True
    return False


def get_info():
    global pos_prv, quat_prv, m, d
    return {"control_size": d.ctrl.shape[0], "body": 16}


if __name__ == "__main__":
    s = init()
    print(s)
