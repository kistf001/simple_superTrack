import utils.myQuaternion as myQuaternion
import torch
import time

POS, VEL, ROT, ANG = 0, 1, 2, 3


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
    root_mat_inv = (rot_mat[..., 0:1, :, :]).inverse()

    pos = (P0 - P0[..., 0:1, :]).unsqueeze(-1)
    pos = torch.einsum("...ij,...jk->...ik", root_mat_inv, pos).squeeze(-1)

    vel = P1.unsqueeze(-1)
    vel = torch.einsum("...ij,...jk->...ik", root_mat_inv, vel).squeeze(-1)

    rot = rot_mat
    rot = torch.einsum("...ij,...jk->...ik", root_mat_inv, rot)[..., 0:2, :]

    ang = P3.unsqueeze(-1)
    ang = torch.einsum("...ij,...jk->...ik", root_mat_inv, ang).squeeze(-1)

    hei = P0[..., 0:1, 2:3]

    up = P0[..., 0:1, :].clone().detach() * 0.0
    up[..., 2:3] += 1
    up = up.unsqueeze(-1)
    up = torch.einsum("...ij,...jk->...ik", root_mat_inv, up).squeeze(-1)

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
    buffer = torch.concat(
        (
            a0.flatten(-2),
            a1.flatten(-2),
            a2.flatten(-3),
            a3.flatten(-2),
            a4.flatten(-2),
            a5.flatten(-2),
        ),
        -1,
    )
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
def gather(rank, model, buffer, env):
    """
    충분한 데이터를 수집할 수 있도록 DeepMimic [Peng et al. 2018a].
    모션 캡처 데이터베이스에서 가져온 해당 운동학 애니메이션을 추적하기 위해
    PD 컨트롤러에 의해 구동되는 각 관절의 모터와 병렬로 256개의 서로 다른 관절
    캐릭터를 시뮬레이션합니다.
    최대 에피소드 길이를 초과하거나 최소 에피소드 길이를 확보하고 캐릭터의 머리 높이가
    기준에서 25cm 이상 벗어날 때마다 기준 상태 초기화 [Peng et al. 2018a]는
    시뮬레이트된 캐릭터를 키네마틱 애니메이션 데이터베이스의 무작위 애니메이션에서
    무작위 포즈로 재설정하는 데 사용되며, 지면 관통을 피하기 위해 수직 보정이 적용됩니다.
    최대 에피소드 길이는 512개, 최소 에피소드 길이는 48개를 사용하고
    애니메이션에 대한 일부 사용자 제공 확률 분포와 관련하여 재설정을 적용하는
    옵션을 제공합니다.
    에피소드가 종료되면 환경의 샘플이 대규모 순환 데이터 버퍼에
    추가됩니다(자세한 내용은 섹션 3.5 참조).
    이 설정은 초당 ~5000개 샘플의 속도로 데이터를 생성합니다.
    시뮬레이션 설정에 대한 자세한 내용은 섹션 5.3을 참조하십시오.
    """

    ADD_NOISE = buffer.get_noise_gain()
    SCALE = buffer.get_action_gain()

    START, END_SIZE = 0, buffer.get_allocated_size()

    #############################################################
    # setting
    #############################################################
    with torch.no_grad():
        model.eval()
        #############################################################
        # Simulator Initialize
        try:
            _S, _K, _T = gather._S, gather._K, gather._T
            fail_count = gather.fail_count
            timestep = gather.timestep
        except:
            _S, _K, _T = env.init(), env.init(), torch.zeros(21)
            fail_count = 0
            timestep = 0
        # Generate Normaldistributed Noise
        noise = torch.normal(0.0, 1.0, (END_SIZE, 21)) * ADD_NOISE
        # RESET load data
        buffer.start(rank)
        # START LOOP
        while START != END_SIZE:
            # Reset if the head position is lower than the standard.
            if timestep == 128:
                _S = env.reset()
                timestep = 0
                fail_count = False
            elif (timestep % 32) == 0:
                if fail_count:
                    _S = env.reset()
                    timestep = 0
                    fail_count = False
            # Converting to Local
            _T = model.policy(to_serial(*local(*_S)).unsqueeze(0))
            # Add noise
            _T = (_T + noise[START]) * SCALE
            # Stack to Circular Buffer
            buffer.insert(_S, _K, _T)
            # Simulator next step
            _S = env.step(_T)
            # count
            START += 1
            timestep += 1
            fail_count |= env.is_fail()
        # End Step Size
        buffer.next()
        gather._S, gather._K, gather._T = _S, _K, _T
        gather.fail_count = fail_count
        gather.timestep = timestep
    ######################################################################
    return


######################################################################
#
######################################################################
def train_world(model, __buffer, L1, L2, optim_world):
    model.train()

    __S, __K, __T = __buffer.get()
    device = __buffer.get_device_type()

    S = (
        __S[POS].clone().detach().transpose(1, 0).to(device),
        __S[VEL].clone().detach().transpose(1, 0).to(device),
        __S[ROT].clone().detach().transpose(1, 0).to(device),
        __S[ANG].clone().detach().transpose(1, 0).to(device),
    )
    T = __T.clone().detach().transpose(1, 0).to(device)
    P = (
        S[POS].clone().detach(),
        S[VEL].clone().detach(),
        S[ROT].clone().detach(),
        S[ANG].clone().detach(),
    )
    P_i = (
        P[POS][0, ...].clone().detach(),
        P[VEL][0, ...].clone().detach(),
        P[ROT][0, ...].clone().detach(),
        P[ANG][0, ...].clone().detach(),
    )

    delta_time = torch.tensor([0.01]).to(device)
    angular_velocity_time = torch.tensor([1]).to(device)

    optim_world.zero_grad()

    # Predict P over a window of 𝑁 frames
    for i in range(8):
        # Predict rigid body accelerations
        buffer = model.world(to_serial(*local(*P_i)), T[i])
        local_vel_dt = buffer[..., 0:48].reshape(-1, 16, 3)
        local_ang_dt = buffer[..., 48:96].reshape(-1, 16, 3)

        # Convert accelerations to world space
        root_mat = myQuaternion.q_2_m(P_i[ROT][..., 0:1, :])  # 1, -1, 1, 4
        world_vel_dt = torch.einsum("...ij,...jk->...ik", root_mat, local_vel_dt.unsqueeze(-1)).squeeze(-1)
        world_ang_dt = torch.einsum("...ij,...jk->...ik", root_mat, local_ang_dt.unsqueeze(-1)).squeeze(-1)

        # Integrate rigid body accelerations
        P_i = integrate(*P_i, world_vel_dt, world_ang_dt, delta_time)

        # P_i integrate to P
        P[POS][i] = P_i[POS].clone()
        P[VEL][i] = P_i[VEL].clone()
        P[ROT][i] = P_i[ROT].clone()
        P[ANG][i] = P_i[ANG].clone()

    # Compute losses
    loss_pos = L1(P[POS], S[POS])
    loss_vel = L1(P[VEL], S[VEL])
    loss_rot = myQuaternion.quat_differentiate_angular_velocity(P[ROT], S[ROT], angular_velocity_time).abs().mean()
    loss_ang = L1(P[ANG], S[ANG])

    # Update network parameters
    loss_sum = loss_pos + loss_vel + loss_rot + loss_ang
    loss_sum.backward()
    optim_world.step()

    return "train_world ", float(loss_sum.detach().cpu().numpy())


def train_policy(model, __buffer, L1, L2, optim_policy):
    """
    정책 Π를 훈련하기 위해 먼저 데이터 버퍼에서 𝑁Π = 32 프레임의 창을
    샘플링하여 초기 시뮬레이션 상태 S0과 목표 운동학적 상태 K를 추출합니다.
    그런 다음 시뮬레이션된 상태와 초기 목표 운동학적 상태가 정책에 공급되어
    PD 오프셋 o를 얻습니다.
    𝜎 = 0.1로 스케일링된 가우시안 노이즈는 오프셋에 추가되어 상태 및 동작
    궤적 탐색을 장려하고 결과 오프셋 o^는 쿼터니언으로 변환되고 운동학적
    캐릭터 관절 회전 k 𝑡를 곱하기 전에 전체 스케일링 인수 𝛼로 스케일링됩니다.
    그런 다음 운동학적 캐릭터 관절 회전 속도 ¤k 𝑡와 쌍을 이루어 최종 PD 목표
    T를 생성합니다.
    이러한 PD 목표는 예측된 시뮬레이션 상태와 함께 세계 모델에 입력되어 다음
    시뮬레이션 상태를 생성합니다.
    이 프로세스는 예측된 상태 P의 전체 창이 생성될 때까지 반복됩니다.
    이 예측과 목표 운동학적 상태 K 간의 차이는 로컬 공간에서 계산되고 손실은
    정책의 가중치를 업데이트하는 데 사용됩니다.
    자세한 내용은 Algorithm 2와 Fig 3을 참조하십시오.
    𝑤𝑢𝑝는 훈련 시작 시 모든 손실에서 거의 동일한 기여도를 제공하도록 설정되는
    반면 정규화 가중치 𝑤𝑙𝑟ม𝑔 및 𝑤𝑠𝑟ม𝑔은 두 자릿수만큼 더 작은 기여도를
    부여하고 큰 PD 오프셋에 페널티를 주는 데 사용됩니다.
    이 알고리즘은 단일 샘플에 대해 제시되지만 훈련은 미니 배치에서 수행됩니다.
    이 알고리즘은 또한 훈련장에서 훈련 데이터를 수집하기 위해 수행되는 절차와
    일치하지만 실제 물리 시뮬레이터를 사용하여 세계 모델이 아닌 다음 시뮬레이션
    상태를 얻습니다.
    """

    model.train()

    __S, __K, __T = __buffer.get()
    device = __buffer.get_device_type()

    # init
    K = (
        __K[POS].clone().detach().transpose(1, 0).to(device),
        __K[VEL].clone().detach().transpose(1, 0).to(device),
        __K[ROT].clone().detach().transpose(1, 0).to(device),
        __K[ANG].clone().detach().transpose(1, 0).to(device),
    )
    P = (
        __S[POS].clone().detach().transpose(1, 0).to(device),
        __S[VEL].clone().detach().transpose(1, 0).to(device),
        __S[ROT].clone().detach().transpose(1, 0).to(device),
        __S[ANG].clone().detach().transpose(1, 0).to(device),
    )
    P_i = (
        P[POS][0, ...].clone().detach(),
        P[VEL][0, ...].clone().detach(),
        P[ROT][0, ...].clone().detach(),
        P[ANG][0, ...].clone().detach(),
    )
    TT = __T.detach().clone().transpose(1, 0).to(device)

    Noise = torch.normal(0.0, 1.0, TT.shape)
    Noise *= __buffer.get_noise_gain().clone()  # MULTIPLY NOISE GAIN 0.1
    Noise = Noise.to(device)

    SCALE = __buffer.get_action_gain().clone().to(device)  # CONTROL SCALE

    delta_time = torch.tensor([0.01]).to(device)

    optim_policy.zero_grad()

    # Predict P over a window of 𝑁 frames
    for i in range(32):
        # Predict PD offset
        OUT = model.policy(to_serial(*local(*P_i)))
        TT[i] = OUT.clone()

        # Add noise to offset
        OUT_hat = OUT + Noise[i]

        # Compute PD target
        """ Papers and implementations are different. """
        OUT_hat *= SCALE

        # Pass through world mod
        buffer = model.world(to_serial(*local(*P_i)), OUT_hat)
        local_vel_dt = buffer[..., 0:48].reshape(-1, 16, 3)
        local_ang_dt = buffer[..., 48:96].reshape(-1, 16, 3)

        # Convert accelerations to world space
        root_mat = myQuaternion.q_2_m(P_i[ROT][..., 0:1, :])
        world_vel_dt = torch.einsum("...ij,...jk->...ik", root_mat, local_vel_dt.unsqueeze(-1)).squeeze(-1)
        world_ang_dt = torch.einsum("...ij,...jk->...ik", root_mat, local_ang_dt.unsqueeze(-1)).squeeze(-1)

        # Integrate rigid body accelerations
        P_i = integrate(*P_i, world_vel_dt, world_ang_dt, delta_time)

        # P_i integrate to P
        P[POS][i] = P_i[POS].clone()
        P[VEL][i] = P_i[VEL].clone()
        P[ROT][i] = P_i[ROT].clone()
        P[ANG][i] = P_i[ANG].clone()

    # Compute Local Spaces
    P_loss, K_loss = local(*P), local(*K)

    # Compute losses
    loss_pos = L1(P_loss[0], K_loss[0])
    loss_vel = L1(P_loss[1], K_loss[1])
    loss_rot = L1(P_loss[2], K_loss[2])
    loss_ang = L1(P_loss[3], K_loss[3])
    loss_hei = L1(P_loss[4], K_loss[4])
    loss_up = L1(P_loss[5], K_loss[5])
    loss_lreg = (TT**2).mean().sqrt()
    loss_sreg = TT.abs().mean()

    # Update network parameters
    loss_sum = loss_pos + loss_vel + loss_rot + loss_ang + loss_hei + loss_up + loss_lreg + loss_sreg
    loss_sum.backward()
    optim_policy.step()
    return "train_policy", float(loss_sum.cpu().detach().numpy())


if "__main__" == __name__:
    from pyquaternion import Quaternion

    rot_quat = Quaternion(w=1, x=0, y=0, z=0) * Quaternion(w=0, x=1, y=0, z=0)
    rot_mat = rot_quat.rotation_matrix

    rot_quat = torch.from_numpy(rot_quat.elements).to(torch.float32)
    rot_mat = torch.from_numpy(rot_mat).to(torch.float32)

    local = local(
        torch.tensor([[0.0, 2.0, 2.0], [0.0, 3.0, 1.0]]).reshape(-1, 3),
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]).reshape(-1, 3),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]).reshape(-1, 4),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).reshape(-1, 3),
    )

    assert torch.all(local[0] == torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0]]))
    assert torch.all(local[1] == torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
    assert torch.all(local[2] == torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]]))
    assert torch.all(local[3] == torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert torch.all(local[4] == torch.tensor([[2.0]]))
    assert torch.all(local[5] == torch.tensor([[0.0, 0.0, 1.0]]))

    integram = integrate(
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([torch.pi, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1]),
    )

    assert torch.all(integram[0] == torch.tensor([1.0, 1.0, 0.0]))
    assert torch.all(integram[1] == torch.tensor([1.0, 0.0, 0.0]))
    assert torch.all(torch.round(integram[2], decimals=3) == torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.all(integram[3] == torch.tensor([torch.pi, 0.0, 0.0]))
