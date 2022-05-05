"""
    Checkpoints generators and time formatting.
"""

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
    lossckpt_gen = loss_checkpoint(init_loss)

    return timeckpt_gen, lossckpt_gen

# timing function
def format_time(elapsed_time):
    """
    format time into hours, minutes, seconds.
    :param float elapsed_time: elapsed time in seconds
    :return str: time formatted as `{hrs}h{mins}m{secs}s`
    """

    elapsed_seconds = round(elapsed_time)

    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)

    elapsed_time = []
    if h > 0:
        elapsed_time.append(f'{h}h')
    if not (h == 0 and m == 0):
        elapsed_time.append(f'{m:02}m')
    elapsed_time.append(f'{s:02}s')

    return ''.join(elapsed_time)
