import argparse
import os

import pytorch_lightning as pl
import torch

import benchmark as bench
from benchmark.utils.config_utils import (
    load_config,
    merge_args_to_config,
    override_config,
    print_config,
)
from benchmark.utils.get_callbacks import get_callbacks
from benchmark.utils.get_dataloader import get_dataloaders


def get_model(cfg):
    cfg._runtime.task_type = bench.TASK_TYPE_MAPPER[cfg.dataset.dataset]
    prober = eval(f"bench.{cfg.dataset.dataset}Prober")
    return prober(cfg)


def main(args):

    cfg = load_config(args.config, namespace=True)
    if args.override is not None and args.override.lower() != "none":
        override_config(args.override, cfg)
    cfg = merge_args_to_config(args, cfg)
    print_config(cfg)

    cfg._runtime = argparse.Namespace()  # runtime info

    assert cfg.trainer.paradigm == "probe", "paradigm must be probe for probe.py"
    pl.seed_everything(cfg.trainer.seed)

    logger = None
    model = get_model(cfg)
    train_loader, valid_loader, test_loader = get_dataloaders(cfg)
    callbacks = get_callbacks(cfg)
    trainer = pl.Trainer.from_argparse_args(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        default_root_dir="./data/lightning_logs",
        # amp_backend='apex',
    )

    trainer.tune(model, train_loader, valid_loader)
    trainer.model.save_hyperparameters()
    trainer.fit(model, train_loader, valid_loader)
    if cfg.trainer.fast_dev_run:
        return

    # force single gpu test to avoid error
    strategy = cfg.trainer.strategy
    if strategy is not None:
        assert (
            "ddp" in strategy
        ), "only support ddp strategy for now, other strategies may not get the right numbers"
        torch.distributed.destroy_process_group()

    if trainer.global_rank == 0:
        # save best ckpt
        best_ckpt_path = trainer.checkpoint_callback.best_model_path

        cfg.trainer.devices = 1
        cfg.trainer.num_nodes = 1
        cfg.trainer.strategy = None

        trainer = pl.Trainer.from_argparse_args(cfg.trainer, logger=logger)
        trainer.validate(
            model=model, dataloaders=valid_loader, ckpt_path=best_ckpt_path
        )
        os.makedirs("./result", exist_ok=True)
        result = []
        ans = []
        for feature, label, _ in test_loader:
            output = model(feature)
            _, out_label = torch.max(output, 1)
            result += out_label.cpu().tolist()
            ans += label.cpu().tolist()

        with open("./result/prediction.txt", "w") as f:
            for i in range(len(result)):
                f.write(str(result[i]) + "\n")
