import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os

os.environ["NCCL_BLOCKING_WAIT"] = "0"
os.environ['DS_SKIP_CUDA_CHECK'] = '1'
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HYDRA_FULL_ERROR"] = "1"

import logging
from datetime import timedelta

import hydra
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import DictConfig, OmegaConf

from tril import tril_run
from tril.algorithms import AlgorithmRegistry
from tril.logging import Tracker

'''
@hydra.main(version_base=None, config_path="cfgs", config_name="config")
hydra config file: cfgs/config.yaml
running main.py task=imdb alg=ppo woule replace task and alg in config.
the input to main (cfg) contains all the info in config+task+alg.

@tril_run
decorator: main(cfg) becomes tril_run(main)
'''
@hydra.main(version_base=None, config_path="cfgs", config_name="config")
@tril_run
def main(cfg: DictConfig):
    # init accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=720000))
    accelerator = Accelerator(
        dispatch_batches=False,
        gradient_accumulation_steps=cfg.alg.args.grad_accumulation,
        kwargs_handlers=[kwargs],
    )

    if accelerator.state.deepspeed_plugin is not None:
        if "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config:
            accelerator.state.deepspeed_plugin.deepspeed_config["fp16"][
                "auto_cast"
            ] = False

    save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    tracker = Tracker(
        save_path, # outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
        OmegaConf.to_container(cfg, resolve=True), # convert cfg to dict
        cfg.project_name, # TRIL
        cfg.experiment_name, # tril_experiment
        cfg.entity_name, # null
        cfg.log_to_wandb, # false
        log_level=logging.INFO, # log general info about the progress
        is_main_process=accelerator.is_main_process, # bool for do something once
    )

    # Initialize algorithm
    try:
        alg_cls = AlgorithmRegistry.get(cfg.alg.id)
    except:
        raise NotImplementedError(
            f"Algorithm {cfg.alg.id} is not supported yet. If implemented, please regist in 'tril.algorithms'."
        )

    alg = alg_cls(cfg, accelerator, tracker)

    # Start Program
    alg.learn()


if __name__ == "__main__":
    main()
