from algorithm import gether, train_policy, train_world

def start(model_master, model_slave, loss1, loss2, optim_w, optim_p, buffer, env):
    
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter()

    counter = 0

    while True:
        ###################################################
        # process
        ###################################################
        # first : S, T, K get function
        gether(0, model_slave, buffer, env)

        # second : learing algorithm
        # for key in buffer.refrash(8): # for mini-batch loop
        #     result_w = train_world (model, buffer, loss1, loss2, optim_w)
        # for key in buffer.refrash(32): # for mini-batch loop
        #     result_p = train_policy(model, buffer, loss1, loss2, optim_p)

        buffer.refrash_all(8) # for mini-batch loop
        result_w = train_world (model_master, buffer, loss1, loss2, optim_w)
        buffer.refrash_all(32) # for mini-batch loop
        result_p = train_policy(model_master, buffer, loss1, loss2, optim_p)

        model_slave.actor .load_state_dict(model_master.actor.state_dict())
        model_slave.critic.load_state_dict(model_master.critic.state_dict())

        print("end")

        ###################################################
        # logging
        ###################################################
        print(
            result_w,
            # result_p,
            counter
            )

        # # writer.add_scalar(result_w[0], result_w[1], counter)
        # # writer.add_scalar(result_p[0], result_p[1], counter)

        # counter += 1

        # # process
        # env.noise_step()

        # # if (counter%100)==0:
        # #     model.save("./train_weight")