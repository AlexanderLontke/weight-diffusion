import os


def get_resume_checkpoint_path(opt):
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        paths = opt.resume.split("/")
        # idx = len(paths)-paths[::-1].index("logs")+1
        # logdir = "/".join(paths[:idx])
        logdir = "/".join(paths[:-2])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), opt.resume
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
    _tmp = logdir.split("/")
    nowname = _tmp[-1]
    return nowname, logdir, ckpt


def create_new_logdir_and_nowname(opt, now):
    if opt.name:
        name = "_" + opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = ""
    nowname = now + name + opt.postfix
    logdir = os.path.join(opt.logdir, nowname)
    return nowname, logdir
