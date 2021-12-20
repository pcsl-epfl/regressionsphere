import math

def loss_checkpoint(init, end=None):
    l = init
    while end is None:
        yield l
        l /= math.sqrt(10.0)

def ckp_init(args, init_loss):

    freq = args.savefreq
    max_step = int(args.maxstep)

    step = freq
    space = step ** (1.0 / 10)
    start = 1.0
    checkpoints = []
    for i in range(9):
        start *= space
        checkpoints.append(int(start))
    while step <= 10 * max_step:
        checkpoints.append(step)
        step += freq

    timeckpt_gen = iter(checkpoints)
    lossckpt_gen = loss_checkpoint(args.alpha * init_loss)

    return timeckpt_gen, lossckpt_gen