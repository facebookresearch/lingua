# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

from lm_eval import simple_evaluate

from omegaconf import OmegaConf
import torch

from apps.main.eval import (
    ValidationArgs,
    EvalHarnessLM,
    LMHarnessArgs,
    eval_on_val,
)
from apps.fastRNN.generate import (
    PackedRNNGenerator,
    PackedRNNGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)

from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import DistributedArgs, get_global_rank, setup_torch_distributed

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


@dataclass
class EvalArgs:
    name: str = "evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    generator: PackedRNNGeneratorArgs = field(default_factory=PackedRNNGeneratorArgs)
    harness: Optional[LMHarnessArgs] = None
    validation: Optional[ValidationArgs] = None

    wandb: Optional[Any] = None

    global_step: Optional[int] = None  # for in-training evaluation


def launch_eval(cfg: EvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
    if (
        Path(cfg.ckpt_dir).exists()
        and (Path(cfg.ckpt_dir) / "params.json").exists()
        and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(cfg.ckpt_dir)
    else:
        consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
        if not consolidate_path.exists() and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    consolidate_path = str(consolidate_path)
    torch.distributed.barrier()
    logger.info("Loading model")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(consolidate_path)
    logger.info("Model loaded")
    model.eval()
    generator = PackedRNNGenerator(cfg.generator, model, tokenizer)

    results = None
    if cfg.harness:
        wrap = EvalHarnessLM(generator)
        results = simple_evaluate(wrap, **asdict(cfg.harness))
    val_results =  None
    if cfg.validation:
        val_results = eval_on_val(generator, cfg.validation, train_cfg)
    if get_global_rank() == 0:
        if results is not None:
            with open(Path(cfg.dump_dir) / "results.json", "w") as f:
                f.write(json.dumps(results))
            logger.info(f"All evaluation results: {results['results']}")
        if val_results is not None:
            with open(Path(cfg.dump_dir) / "validation.json", "w") as f:
                f.write(json.dumps(val_results))
            logger.info(f"All validation results: {val_results}")
    if cfg.metric_log_dir and get_global_rank() == 0:
        metric_log_path = Path(cfg.metric_log_dir) / "metrics.eval.jsonl"

        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if cfg.global_step is not None:
            timestamp["global_step"] = cfg.global_step
        if results is not None:
            print(
                json.dumps(timestamp | results["results"]),
                file=open(metric_log_path, mode="a"),
                flush=True,
            )

        val_log_path = Path(cfg.metric_log_dir) / "metrics.validation.jsonl"
        if val_results is not None:
            print(
                json.dumps(timestamp | val_results),
                file=open(val_log_path, mode="a"),
                flush=True,
            )
    
    del generator


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        mode: LMMambaArg

    @dataclass
    class LMMambaArgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMMambaArgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate EvalArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call eval.py with eval.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in EvalArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_eval(cfg)


if __name__ == "__main__":
    main()
