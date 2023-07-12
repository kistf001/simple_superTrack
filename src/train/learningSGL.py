from algorithm import gether, train_policy, train_world

env = None
model = None
loss1 = None
loss2 = None
optim_w = None
optim_p = None
buffer = None

def init(_model, _loss1, _loss2, _optim_w, _optim_p, _env, _buffer):
    """
    input : model, loss, optim_w, optim_p, env, buffer, step_size
    """
    global model, loss1, loss2, optim_w, optim_p, env, buffer
    model            = _model
    loss1, loss2     = _loss1, _loss2
    optim_w, optim_p = _optim_w, _optim_p
    env              = _env
    buffer           = _buffer

def start():
    global model, loss1, loss2, optim_w, optim_p, env, buffer

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter()

    counter = 0

    while True:
        ###################################################
        # process
        ###################################################
        # first. S, T, K get funktion
        gether(0, model, env, buffer)

        # second. learing algorithm
        result_w = train_world (model, env, buffer, loss1, loss2, optim_w)
        result_p = train_policy(model, env, buffer, loss1, loss2, optim_p)

        ###################################################
        # logging
        ###################################################
        print(
            result_w,
            # result_p,
            counter
            )

        # writer.add_scalar(result_w[0], result_w[1], counter)
        # writer.add_scalar(result_p[0], result_p[1], counter)

        counter += 1

        # if (counter%100)==0:
        #     model.save("./train_weight")