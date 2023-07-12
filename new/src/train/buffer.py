import torch

"""
buffer size : 
[                               ]
buffer split size : 4
[       |       |       |       ]
buffer together size : 4
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

POS = 0
VEL = 1
ROT = 2
ANG = 3

def init(_buffer_size, _buffer_split, _process_num):
    global S_buffer, K_buffer, T_buffer
    global buffer_size, buffer_size_split, buffer_size_processor
    global offset_processor, offset_split
    global working_step_size

    buffer_size           = _buffer_size
    buffer_size_split     = _buffer_split
    buffer_size_processor = _process_num

    offset_split = int(buffer_size / buffer_size_split)
    offset_processor = int(offset_split / _process_num)

    # 프로세스당 최대로 할당된 스탭횟수
    working_step_size = int(
        buffer_size / buffer_size_processor / buffer_size_split
        )

    S_buffer = [
        torch.zeros((buffer_size,16,3)).share_memory_(),
        torch.zeros((buffer_size,16,3)).share_memory_(),
        torch.zeros((buffer_size,16,4)).share_memory_(),
        torch.zeros((buffer_size,16,3)).share_memory_()
        ]
    S_buffer[ROT][..., 0] = 1.0
    K_buffer = [
        torch.zeros((buffer_size,16,3)).share_memory_(),
        torch.zeros((buffer_size,16,3)).share_memory_(),
        torch.zeros((buffer_size,16,4)).share_memory_(),
        torch.zeros((buffer_size,16,3)).share_memory_()
        ]
    K_buffer[ROT][..., 0] = 1.0
    T_buffer = torch.zeros((buffer_size,21)).share_memory_()

def get():
    """
    return (S_buffer, K_buffer, T_buffer) 
    """
    return (S_buffer, K_buffer, T_buffer)

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

if __name__ == "__main__":
    init(32*8*4, 4, 2)
    start(0)
    for a in range(get_allocated_size()):
        insert(
            (
                torch.tensor([a,a,a,]),
                torch.tensor([a,a,a,]),
                torch.tensor([a,a,a,a,]),
                torch.tensor([a,a,a,]),
                ),
            (
                torch.tensor([a,a,a,]),
                torch.tensor([a,a,a,]),
                torch.tensor([a,a,a,a,]),
                torch.tensor([a,a,a,]),
                ),
            torch.tensor([
                a,a,a,a,a,
                a,a,a,a,a,
                a,a,a,a,a,
                a,a,a,a,a,a])
            )
    next()

    S,K,T = get()

    print(S[0][31])