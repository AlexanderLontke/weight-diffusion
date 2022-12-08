import os
import argparse
from packaging import version
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger


from ldm.util import instantiate_from_config


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def get_model_trainer_and_data_from_config(
    opt, unknown, logdir, nowname, now, cfgdir, ckptdir
):
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)

    # model
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()

    # Add WandB logging
    trainer_kwargs["logger"] = WandbLogger(
        name=nowname,
        save_dir=logdir,
        offline=opt.debug,
        id=nowname,
    )

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
            "every_n_epochs": 1,
        },
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse("1.4.0"):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "weight_diffusion.execution.setup_callback.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            },
        },
        "cuda_callback": {"target": "weight_diffusion.execution.cuda_callback.CUDACallback"},
    }
    if version.parse(pl.__version__) >= version.parse("1.4.0"):
        default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
        print(
            "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 10000,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if "ignore_keys_callback" in callbacks_cfg and hasattr(
        trainer_opt, "resume_from_checkpoint"
    ):
        callbacks_cfg.ignore_keys_callback.params[
            "ckpt_path"
        ] = trainer_opt.resume_from_checkpoint
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]

    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    # data.prepare_data()
    # data.setup()
    # print("#### Data #####")
    # for k in data.datasets:
    #    print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        # TODO fix
        ngpu = 1
    else:
        ngpu = 1
    if "accumulate_grad_batches" in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
            )
        )
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    return data, model, trainer
