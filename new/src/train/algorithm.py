import myQuaternion
import time
import torch

POS = 0
VEL = 1
ROT = 2
ANG = 3

######################################################################
#
######################################################################
@torch.jit.script
def local(P0, P1, P2, P3):
    """
    input : (pos, vel, rot, ang)
    return : (pos, vel, rot, ang, hei, up)
    """

    # print([a.shape for a in [P0, P1, P2, P3]])
    rot_mat = myQuaternion.q_2_m(P2).clone()
    root_mat_inv = (rot_mat[...,0:1,:,:]).inverse()

    pos = (P0 - P0[..., 0:1, :]).unsqueeze(-1)
    pos = torch.einsum("...ij,...jk->...ik",root_mat_inv, pos) .squeeze(-1)

    vel = P1.unsqueeze(-1)
    vel = torch.einsum("...ij,...jk->...ik",root_mat_inv, vel) .squeeze(-1)

    rot = rot_mat
    rot = torch.einsum("...ij,...jk->...ik",root_mat_inv, rot)[..., 0:2, :]

    ang = P3.unsqueeze(-1)
    ang = torch.einsum("...ij,...jk->...ik",root_mat_inv, ang) .squeeze(-1)

    hei = P0[..., 0:1, 2:3]

    up = torch.zeros(P0[..., 0:1, :].shape)
    up[..., 2:3] += 1
    up = up.unsqueeze(-1)
    up = torch.einsum("...ij,...jk->...ik",root_mat_inv, up).squeeze(-1)

    # print("==============================")
    # print(root_mat_inv.shape, pos.shape)
    # print(root_mat_inv.shape, vel.shape)
    # print(root_mat_inv.shape, rot.shape)
    # print(root_mat_inv.shape, ang.shape)
    # print(root_mat_inv.shape, hei.shape)
    # print(root_mat_inv.shape, up.shape)
    # print("==============================")

    return (pos, vel, rot, ang, hei, up)
######################################################################
@torch.jit.script
def to_serial(a0, a1, a2, a3, a4, a5):
    """
    input : {pos, vel, rot, ang, hei, up}
    return : 
    """
    # print([a.shape for a in [a0, a1, a2, a3, a4, a5]])
    # print(a4.flatten(-2).shape)
    buffer = torch.concat((
        a0.flatten(-2),
        a1.flatten(-2), 
        a2.flatten(-3), 
        a3.flatten(-2), 
        a4.flatten(-2), 
        a5.flatten(-2),
        ),-1)
    # print(buffer.shape)
    return buffer
######################################################################
@torch.jit.script
def integrate(P_pos, P_vel, P_rot, P_ang, vel_of_dt, ang_of_dt, dt):
    """
    input : (
        P_pos, P_vel, P_rot, P_ang, 
        vel_of_dt, ang_of_dt, 
        dt
        )
    return : (y_pos, y_vel, y_rot, y_ang)
    """
    y_vel = (dt * vel_of_dt) + P_vel
    y_ang = (dt * ang_of_dt) + P_ang
    y_pos = (dt * y_vel) + P_pos
    y_rot = myQuaternion.quat_integrate_angular_velocity(y_ang, P_rot, dt)
    return (y_pos, y_vel, y_rot, y_ang)
######################################################################

######################################################################
#
######################################################################
ADD_NOISE = 0.05
SCALE = 10

######################################################################
#
######################################################################
def gether(rank, model, env, buffer):
    #############################################################
    # setting
    #############################################################
    with torch.no_grad():
    #############################################################
        # Simulator Initialize
        _S, _K, _T = env.init(), env.init(), torch.zeros(21)
        #
        start = 0
        end = buffer.get_allocated_size()
        #
        buffer.start(rank)
        #
        while start != end:
            # Fail detaect
            if (start % 64) == 0 :
                _S = env.reset()
            # if (start % 32) == 0 :
            #     if(env.is_fail()):
            #         _S = env.reset()
            # Converting to Local
            _T = model.policy(to_serial(*local(*_S)))
            _T = (torch.normal(0.0, 1.0, _T.shape) * ADD_NOISE) + (_T) 
            _T *= SCALE
            # Stacking to Circular Buffer
            buffer.insert(_S, _K, _T)
            # Simulator next step
            _S = env.step(_T)
            # count
            start += 1
        # End Step Size
        buffer.next()
    ######################################################################
    return
######################################################################

######################################################################
#
######################################################################
def train_world (model, env, __buffer, L1, L2, optim_world):

    __S, __K, __T = __buffer.get()

    # init
    S = (
        __S[POS].detach().clone().reshape(-1,8,16,3).transpose(1, 0), 
        __S[VEL].detach().clone().reshape(-1,8,16,3).transpose(1, 0), 
        __S[ROT].detach().clone().reshape(-1,8,16,4).transpose(1, 0), 
        __S[ANG].detach().clone().reshape(-1,8,16,3).transpose(1, 0),
        )
    T = __T.detach().clone().reshape(-1,8,21).transpose(1, 0)
    P = (
        S[POS].detach().clone(), 
        S[VEL].detach().clone(), 
        S[ROT].detach().clone(), 
        S[ANG].detach().clone(),
        )
    P_i = (
        P[POS][0:1,...].detach().clone(), 
        P[VEL][0:1,...].detach().clone(), 
        P[ROT][0:1,...].detach().clone(), 
        P[ANG][0:1,...].detach().clone(),
        )

    # Predict P over a window of 𝑁 frames
    for i in range(8):

        # Predict rigid body accelerations
        buffer = model.world(to_serial(*local(*P_i)), T[i:i+1])
        local_vel_dt = buffer[...,  0:48].reshape(1,-1,16,3)
        local_ang_dt = buffer[..., 48:96].reshape(1,-1,16,3)

        # Convert accelerations to world space
        root_mat = myQuaternion.q_2_m(P_i[ROT][..., 0:1, :]) # 1, -1, 1, 4
        world_vel_dt = torch.einsum(
            "...ij,...jk->...ik", 
            root_mat, local_vel_dt.unsqueeze(-1)).squeeze(-1)
        world_ang_dt = torch.einsum(
            "...ij,...jk->...ik", 
            root_mat, local_ang_dt.unsqueeze(-1)).squeeze(-1)

        # Integrate rigid body accelerations
        P_i = integrate(*P_i, world_vel_dt, world_ang_dt, torch.tensor([0.01]))

        # P_i integrate to P
        P[POS][i:i+1] = P_i[POS].clone()
        P[VEL][i:i+1] = P_i[VEL].clone()
        P[ROT][i:i+1] = P_i[ROT].clone()
        P[ANG][i:i+1] = P_i[ANG].clone()

    # Compute losses
    loss_pos = L1(P[POS], S[POS])
    loss_vel = L1(P[VEL], S[VEL])
    loss_rot = myQuaternion.quat_differentiate_angular_velocity(
            P[ROT], S[ROT], torch.tensor([1]) ).abs().mean()
    loss_ang = L1(P[ANG], S[ANG])

    # Update network parameters
    optim_world.zero_grad()
    loss_sum = loss_pos + loss_vel + loss_rot + loss_ang
    loss_sum.backward()
    optim_world.step()

    return "train_world ", float(loss_sum.detach().numpy())
def train_policy(model, env, __buffer, L1, L2, optim_policy):
    
    __S, __K, __T = __buffer.get()

    # init
    S = (
        __S[POS].detach().clone().reshape(-1,32,16,3).transpose(1, 0), 
        __S[VEL].detach().clone().reshape(-1,32,16,3).transpose(1, 0), 
        __S[ROT].detach().clone().reshape(-1,32,16,4).transpose(1, 0), 
        __S[ANG].detach().clone().reshape(-1,32,16,3).transpose(1, 0),
        )
    K = (
        __K[POS].detach().clone().reshape(-1,32,16,3).transpose(1, 0), 
        __K[VEL].detach().clone().reshape(-1,32,16,3).transpose(1, 0), 
        __K[ROT].detach().clone().reshape(-1,32,16,4).transpose(1, 0), 
        __K[ANG].detach().clone().reshape(-1,32,16,3).transpose(1, 0),
        )
    P = (
        S[POS].detach().clone(), 
        S[VEL].detach().clone(), 
        S[ROT].detach().clone(), 
        S[ANG].detach().clone(),
        )
    P_i = (
        P[POS][0:1,...].detach().clone(), 
        P[VEL][0:1,...].detach().clone(), 
        P[ROT][0:1,...].detach().clone(), 
        P[ANG][0:1,...].detach().clone(),
        )
    TT = __T.detach().clone().reshape(-1,32,21).transpose(1, 0)
    
    # Predict P over a window of 𝑁 frames
    for i in range(32):

        # Predict PD offset
        OUT = model.policy(to_serial(*local(*P_i)))
        TT[i:i+1] = OUT.clone()

        # Add noise to offset
        OUT_hat = (torch.normal(0.0, 1.0, OUT.shape) * ADD_NOISE) + (OUT)

        # Compute PD target is skiped 
        """ In mujoco ALL joint have one Parameter  """
        OUT_hat *= SCALE

        # Pass through world mod
        buffer = model.world(to_serial(*local(*P_i)), OUT_hat)
        local_vel_dt = buffer[...,  0:48].reshape(1,-1,16,3)
        local_ang_dt = buffer[..., 48:96].reshape(1,-1,16,3)

        # Convert accelerations to world space
        root_mat = myQuaternion.q_2_m(P_i[ROT][..., 0:1, :])
        world_vel_dt = torch.einsum(
            "...ij,...jk->...ik", 
            root_mat, local_vel_dt.unsqueeze(-1)).squeeze(-1)
        world_ang_dt = torch.einsum(
            "...ij,...jk->...ik", 
            root_mat, local_ang_dt.unsqueeze(-1)).squeeze(-1)

        # Integrate rigid body accelerations
        P_i = integrate(*P_i, world_vel_dt, world_ang_dt, torch.tensor([0.01]))

        # P_i integrate to P
        P[POS][i:i+1] = P_i[POS].clone()
        P[VEL][i:i+1] = P_i[VEL].clone()
        P[ROT][i:i+1] = P_i[ROT].clone()
        P[ANG][i:i+1] = P_i[ANG].clone()

    # Compute Local Spaces
    P_loss = local(*P)
    K_loss = local(*K)

    # Compute losses
    loss_pos = L1(P_loss[0], K_loss[0])
    loss_vel = L1(P_loss[1], K_loss[1])
    loss_rot = L1(P_loss[2], K_loss[2])
    loss_ang = L1(P_loss[3], K_loss[3])
    loss_hei = L1(P_loss[4], K_loss[4])
    loss_up  = L1(P_loss[5], K_loss[5])
    loss_lreg = (TT**2).mean().sqrt()
    loss_sreg = TT.abs().mean()
    
    # Update network parameters
    optim_policy.zero_grad()
    loss_sum = (
        loss_pos + loss_vel + 
        loss_rot + loss_ang + 
        loss_hei + loss_up + 
        loss_lreg + loss_sreg
        )
    print(loss_hei , loss_up ,
        loss_lreg , loss_sreg)
    loss_sum.backward()
    optim_policy.step()
    return "train_policy", float(loss_sum.detach().numpy())
######################################################################

if "__main__" == __name__ :
    from pyquaternion import Quaternion

    rot_quat = Quaternion(w=1, x=0, y=0, z=0) * Quaternion(w=0, x=1, y=0, z=0)
    rot_mat = rot_quat.rotation_matrix

    rot_quat = torch.from_numpy(rot_quat.elements).to(torch.float32)
    rot_mat = torch.from_numpy(rot_mat).to(torch.float32)

    local = local(
        torch.tensor([[0., 2., 2.], [0., 3., 1.]]).reshape(-1,3),
        torch.tensor([[0., 1., 0.], [0., 1., 0.]]).reshape(-1,3),
        torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]).reshape(-1,4),
        torch.tensor([[0., 0., 0.], [0., 0., 0.]]).reshape(-1,3),
        )
    
    assert torch.all(local[0] == torch.tensor([[0., 0., 0.], [0., 1., -1.]]))
    assert torch.all(local[1] == torch.tensor([[0., 1., 0.], [0., 1.,  0.]]))
    assert torch.all(local[2] == torch.tensor([
        [[1., 0., 0.], [0.,  1., 0.]], 
        [[1., 0., 0.], [0., -1., 0.]]
        ]))
    assert torch.all(local[3] == torch.tensor([[0., 0., 0.], [0., 0., 0.]]))
    assert torch.all(local[4] == torch.tensor([[2.]]))
    assert torch.all(local[5] == torch.tensor([[0., 0., 1.]]))

    integram = integrate(
        torch.tensor([0., 1., 0.]),
        torch.tensor([1., 0., 0.]),
        torch.tensor([1., 0., 0., 0.]),
        torch.tensor([torch.pi, 0., 0.]), 
        torch.tensor([0., 0., 0.]), torch.tensor([0., 0., 0.]), 
        torch.tensor([1])
        )

    assert torch.all(integram[0] == torch.tensor([1., 1., 0.]))
    assert torch.all(integram[1] == torch.tensor([1., 0., 0.]))
    assert torch.all(torch.round(integram[2], decimals=3) == torch.tensor([0., 1., 0., 0.]))
    assert torch.all(integram[3] == torch.tensor([torch.pi, 0., 0.]))