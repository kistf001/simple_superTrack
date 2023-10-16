from algorithm import gather, train_policy, train_world
import time
import torch
import torch.multiprocessing as mp


def start(model_master, model_slave, loss1, loss2, optim_w, optim_p, _buffer, _env):
    muchine_num = _buffer.buffer_size_processor
    start_flag = torch.zeros((muchine_num)).share_memory_()

    ##########################################################
    # data gathering
    ##########################################################
    mp.set_start_method("spawn")

    for i in range(muchine_num):
        p = mp.Process(target=loop_slave_process, args=(i, model_slave, _buffer._export(), start_flag))
        p.start()

    ##########################################################
    # learning
    ##########################################################
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()

    counter = 0

    while True:
        # multiprocess flow control
        while start_flag.sum() != muchine_num:
            time.sleep(0.05)

        # variable
        result_w, result_p = ["train_world", 0], ["train_policy", 0]

        # For mini-batch
        # for key in buffer.refrash(8): # for mini-batch loop
        #     result_w = train_world (model_master, env, buffer, loss1, loss2, optim_w)
        # for key in buffer.refrash(32): # for mini-batch loop
        #     result_p = train_policy(model_master, env, buffer, loss1, loss2, optim_p)
        # For batch
        _buffer.refrash_all(8)
        result_w = train_world(model_master, _buffer, loss1, loss2, optim_w)
        _buffer.refrash_all(32)
        result_p = train_policy(model_master, _buffer, loss1, loss2, optim_p)

        #
        model_slave.actor.load_state_dict(model_master.actor.state_dict())
        model_slave.critic.load_state_dict(model_master.critic.state_dict())

        #
        _buffer.noise_step()

        #
        start_flag *= 0

        ##########################################################################
        #
        ##########################################################################

        writer.add_scalar(result_w[0], result_w[1], counter)
        writer.add_scalar(result_p[0], result_p[1], counter)
        writer.add_scalar("noise", _buffer.noise_gain, counter)

        print(result_w, result_p, counter, _buffer.get_noise_gain())

        counter += 1

        if (counter % 25) == 0:
            model_master.save("./train_weight")


def loop_slave_process(rank, model_slave, buffer_data, start_flag):
    import utils.buffer as buffer, utils.env as env

    ##########################################################################
    # Buffer setting
    ##########################################################################
    buffer._import(buffer_data)
    while buffer.S_buffer == None:
        time.sleep(1)

    ##########################################################################
    # Seed setting
    ##########################################################################
    # important in multi processing agent
    torch.manual_seed(torch.rand(1) * (rank * 1000000000) * time.time())
    time.sleep(float(torch.rand([1])))
    torch.manual_seed(torch.rand(1) * (rank * 1000000000) * time.time())

    ##########################################################################
    # DATA FILLING
    ##########################################################################
    gather(rank, model_slave, buffer, env)
    gather(rank, model_slave, buffer, env)
    gather(rank, model_slave, buffer, env)
    gather(rank, model_slave, buffer, env)
    gather(rank, model_slave, buffer, env)
    gather(rank, model_slave, buffer, env)
    gather(rank, model_slave, buffer, env)
    gather(rank, model_slave, buffer, env)

    ##########################################################################
    #
    ##########################################################################
    while True:
        while start_flag[rank] != 0:
            time.sleep(0.05)
        gather(rank, model_slave, buffer, env)
        start_flag[rank] += 1
