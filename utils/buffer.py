"""
utils/buffer.py

supertrack의 순환 버퍼를 구현합니다.
init함수를 이용하여 버퍼의 크기를 설정하고, insert, next함수를 
이용하여 데이터를 삽입하고 다음 인덱스로 넘어갑니다.
"""

import torch

action_gain = torch.tensor([10]).share_memory_()
noise_gain = torch.tensor([0.015]).share_memory_()
noise_decay_rate = torch.tensor([0.9999]).share_memory_()
noise_decay_value = torch.tensor([1]).share_memory_()

gain_rate = 0.001


def noise_init(action, noise, decay):
    global noise_gain
    noise_gain


def noise_step():
    global noise_gain, gain_rate, noise_decay_rate
    # try:
    #     noise_step.count += 1
    #     if (noise_step.count % 100) == 0:
    #         noise_gain -= gain_rate
    # except:
    #     noise_step.count = 0
    # if noise_gain > 0.01:
    #     noise_gain *= noise_decay_rate


def get_action_gain():
    return action_gain


def get_noise_gain():
    return noise_gain


"""
buffer size : 
[                               ]
buffer split size : 4
[       |       |       |       ]
buffer togather size : 4
[ | | | |       |       |       ]
"""

S_buffer = None
K_buffer = None
T_buffer = None

buffer_size = 0
buffer_size_split = 4
buffer_size_processor = 4

index_step = 0
index_split = 0

offset_processor = 0
offset_split = 0

working_step_size = 0

device = "cpu"

POS = 0
VEL = 1
ROT = 2
ANG = 3


def init(_buffer_size, _buffer_split, _process_num, _type="master"):
    global S_buffer, K_buffer, T_buffer
    global buffer_size, buffer_size_split, buffer_size_processor
    global offset_processor, offset_split
    global working_step_size

    buffer_size = _buffer_size
    buffer_size_split = _buffer_split
    buffer_size_processor = _process_num

    offset_split = int(buffer_size / buffer_size_split)
    offset_processor = int(offset_split / _process_num)

    # 프로세스당 최대로 할당된 스탭횟수
    working_step_size = int(buffer_size / buffer_size_processor / buffer_size_split)

    if _type == "master":
        S_buffer = [
            torch.zeros((buffer_size, 16, 3)).share_memory_(),
            torch.zeros((buffer_size, 16, 3)).share_memory_(),
            torch.zeros((buffer_size, 16, 4)).share_memory_(),
            torch.zeros((buffer_size, 16, 3)).share_memory_(),
        ]
        S_buffer[ROT][..., 0] = 1.0
        K_buffer = [
            torch.zeros((buffer_size, 16, 3)).share_memory_(),
            torch.zeros((buffer_size, 16, 3)).share_memory_(),
            torch.zeros((buffer_size, 16, 4)).share_memory_(),
            torch.zeros((buffer_size, 16, 3)).share_memory_(),
        ]
        K_buffer[ROT][..., 0] = 1.0
        T_buffer = torch.zeros((buffer_size, 21)).share_memory_()


def init_device(_device="cpu"):
    global device
    device = _device


def start(_rank):
    global index_step, index_split
    index_step = 0 + (offset_split * index_split) + (_rank * offset_processor)


def insert(_S, _K, _T):
    global S_buffer, K_buffer, T_buffer
    global index_step
    S_buffer[POS][index_step] = _S[POS]
    S_buffer[VEL][index_step] = _S[VEL]
    S_buffer[ROT][index_step] = _S[ROT]
    S_buffer[ANG][index_step] = _S[ANG]
    K_buffer[POS][index_step] = _K[POS]
    K_buffer[VEL][index_step] = _K[VEL]
    K_buffer[ROT][index_step] = _K[ROT]
    K_buffer[ANG][index_step] = _K[ANG]
    T_buffer[index_step] = _T
    index_step += 1


def next():
    global index_split, buffer_size_split
    global index_step
    index_split = (index_split + 1) % buffer_size_split


def get_allocated_size():
    global working_step_size
    return working_step_size


def get_device_type():
    global device
    return device


data_buffer = ()


def refrash(
    window_size,
    batch_size=32,
):
    global S_buffer, K_buffer, T_buffer
    global data_buffer

    S_pos = S_buffer[0].view(-1, window_size, 16, 3)
    S_vel = S_buffer[1].view(-1, window_size, 16, 3)
    S_rot = S_buffer[2].view(-1, window_size, 16, 4)
    S_ang = S_buffer[3].view(-1, window_size, 16, 3)

    K_pos = K_buffer[0].view(-1, window_size, 16, 3)
    K_vel = K_buffer[1].view(-1, window_size, 16, 3)
    K_rot = K_buffer[2].view(-1, window_size, 16, 4)
    K_ang = K_buffer[3].view(-1, window_size, 16, 3)

    T = T_buffer.view(-1, window_size, 21)

    dataset = torch.utils.data.TensorDataset(S_pos, S_vel, S_rot, S_ang, K_pos, K_vel, K_rot, K_ang, T)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    for key, data in enumerate(loader):
        data_buffer = ((data[0], data[1], data[2], data[3]), (data[4], data[5], data[6], data[7]), data[8])
        yield


def refrash_all(
    window_size=-1,
    batch_size=32,
):
    global S_buffer, K_buffer, T_buffer
    global data_buffer

    S_pos = S_buffer[0].view(-1, window_size, 16, 3)
    S_vel = S_buffer[1].view(-1, window_size, 16, 3)
    S_rot = S_buffer[2].view(-1, window_size, 16, 4)
    S_ang = S_buffer[3].view(-1, window_size, 16, 3)

    K_pos = K_buffer[0].view(-1, window_size, 16, 3)
    K_vel = K_buffer[1].view(-1, window_size, 16, 3)
    K_rot = K_buffer[2].view(-1, window_size, 16, 4)
    K_ang = K_buffer[3].view(-1, window_size, 16, 3)

    T = T_buffer.view(-1, window_size, 21)
    data_buffer = ((S_pos, S_vel, S_rot, S_ang), (K_pos, K_vel, K_rot, K_ang), T)


def get():
    """
    return : S_buffer, K_buffer, T_buffer
    """
    global data_buffer
    return data_buffer


def _export():
    global S_buffer, K_buffer, T_buffer
    global buffer_size, buffer_size_split, buffer_size_processor
    global device
    global action_gain, noise_gain, noise_decay_rate, noise_decay_value
    return (
        S_buffer,
        K_buffer,
        T_buffer,
        buffer_size,
        buffer_size_split,
        buffer_size_processor,
        device,
        action_gain,
        noise_gain,
        noise_decay_rate,
        noise_decay_value,
    )


def _import(SKT):
    global S_buffer, K_buffer, T_buffer
    global buffer_size, buffer_size_split, buffer_size_processor
    global device
    global action_gain, noise_gain, noise_decay_rate, noise_decay_value
    (
        S_buffer,
        K_buffer,
        T_buffer,
        buffer_size,
        buffer_size_split,
        buffer_size_processor,
        device,
        action_gain,
        noise_gain,
        noise_decay_rate,
        noise_decay_value,
    ) = SKT
    init(buffer_size, buffer_size_split, buffer_size_processor, "slave")


if __name__ == "__main__":
    init(32 * 8 * 4, 4, 2)
    start(0)
    for a in range(get_allocated_size()):
        insert(
            (
                torch.tensor(
                    [
                        a,
                        a,
                        a,
                    ]
                ),
                torch.tensor(
                    [
                        a,
                        a,
                        a,
                    ]
                ),
                torch.tensor(
                    [
                        a,
                        a,
                        a,
                        a,
                    ]
                ),
                torch.tensor(
                    [
                        a,
                        a,
                        a,
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        a,
                        a,
                        a,
                    ]
                ),
                torch.tensor(
                    [
                        a,
                        a,
                        a,
                    ]
                ),
                torch.tensor(
                    [
                        a,
                        a,
                        a,
                        a,
                    ]
                ),
                torch.tensor(
                    [
                        a,
                        a,
                        a,
                    ]
                ),
            ),
            torch.tensor([a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a]),
        )
    next()

    S, K, T = get()

    print(S[0][31])
