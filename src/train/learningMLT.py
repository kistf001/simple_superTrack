from algorithm import gether, train_policy, train_world
import torch.multiprocessing as mp
import time
import torch

model_master, model_slave = None, None
loss1       , loss2       = None, None
optim_w     , optim_p     = None, None
buffer                    = None
env                       = None

def init(
    _model_master, _model_slave, 
    _loss1, _loss2, _optim_w, _optim_p, _env, _buffer):
    """
    input : model, loss, optim_w, optim_p, env, buffer, step_size
    """
    global model_master, model_slave
    global loss1, loss2
    global optim_w, optim_p
    global buffer
    global env

    model_master, model_slave = _model_master, _model_slave
    loss1       , loss2       = _loss1, _loss2
    optim_w     , optim_p     = _optim_w, _optim_p
    buffer                    = _buffer
    env                       = _env

def start():
    global model_master, model_slave
    global loss1, loss2
    global optim_w, optim_p
    global buffer
    global env

    counter = 0

    muchine_num = buffer.buffer_size_processor

    start_flag = torch.zeros((muchine_num)).share_memory_()

    ##########################################################
    # data gethering
    ##########################################################
    for i in range(muchine_num):
        p = mp.Process(
            target = loop_slave_process, 
            args = (
                i, model_slave, env, buffer, start_flag
                ))
        p.start()

    ##########################################################
    # learning
    ##########################################################
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    while True:
        #
        counter += 1
        # 
        while(start_flag.sum()!=muchine_num):
            time.sleep(0.05)
        # process
        result_w = train_world(model_master, env, buffer, loss1, loss2, optim_w)
        result_p = train_policy(model_master, env, buffer, loss1, loss2, optim_p)
        model_slave.actor .load_state_dict(model_master.actor.state_dict())
        model_slave.critic.load_state_dict(model_master.critic.state_dict())
        writer.add_scalar(result_w[0], result_w[1], counter)
        writer.add_scalar(result_p[0], result_p[1], counter)
        start_flag *= 0; #time.sleep(2)
        print(result_w, result_p, counter)
        if(counter%25) == 0:
            # logging
            model_master.save("./train_weight")

def loop_slave_process(rank, model_slave, env, buffer, start_flag):
    # important in multi processing agent
    torch.manual_seed(torch.rand(1) * (rank * 1000000000) * time.time())
    #
    gether(rank, model_slave, env, buffer)
    gether(rank, model_slave, env, buffer)
    gether(rank, model_slave, env, buffer)
    gether(rank, model_slave, env, buffer)
    gether(rank, model_slave, env, buffer)
    gether(rank, model_slave, env, buffer)
    gether(rank, model_slave, env, buffer)
    gether(rank, model_slave, env, buffer)
    #
    while True:
        while(start_flag[rank]!=0):
            time.sleep(0.05)
        gether(rank, model_slave, env, buffer)
        start_flag[rank] += 1
