import argparse
import datetime
import os
import shutil
import sys
import time

import matplotlib
import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml

from policy import config
from policy.checkpoints import CheckpointIO
from policy.dataset import loader as loader_utils
from policy.perception.scene_vae.training import Trainer
from policy.utils.io import ENV_NAMES, create_folder, get_asset_path

matplotlib.use("Agg")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        choices=ENV_NAMES,
        required=True,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Turning on the debugging mode",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Removing old output directory",
    )
    parser.add_argument(
        "--max_it",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="custom config file",
    )

    args = parser.parse_args()
    return args


def main(args):
    env_name = args.env_name
    stage = args.stage

    default_cfg_path = get_asset_path("perception/default.yaml")

    if len(args.config) > 0:
        user_cfg_path = args.config
        if not os.path.exists(user_cfg_path):
            raise FileExistsError("user cfg does not exist!" f"==> {user_cfg_path}")
    elif stage == 1:
        user_cfg_path = get_asset_path(f"perception/{env_name}/vae_stage1.yaml")
    elif stage == 2:
        user_cfg_path = get_asset_path(f"perception/{env_name}/vae_stage2.yaml")
    else:
        raise ValueError("Stage number is incorrect!")

    assert all(os.path.exists(elem) for elem in (default_cfg_path, user_cfg_path))

    cfg = config.load_config(user_cfg_path, default_cfg_path)

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    # Set t0
    t0 = time.time()

    # Shorthands
    out_dir = cfg["training"]["out_dir"]
    vis_dir = os.path.join(out_dir, "vis")
    if args.debug:
        cfg["training"]["batch_size"] = 2
        cfg["training"]["vis_n_outputs"] = 1
        cfg["training"]["print_every"] = 1
        cfg["training"]["backup_every"] = 1
        cfg["training"]["validate_every"] = 1
        cfg["training"]["visualize_every"] = 1
        cfg["training"]["checkpoint_every"] = 1
        cfg["training"]["visualize_total"] = 1
        cfg["training"]["max_it"] = 1

    wandb.init(
        project="dexterous-hand-implicit",
        config=cfg,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb.run.name = "-".join(
        [env_name, f"stage{stage}", wandb.run.name.split("-")[-1]]
    )
    wandb.run.save()
    yaml.dump(cfg, sys.stdout)

    backup_every = cfg["training"]["backup_every"]
    max_it = cfg["training"]["max_it"]
    if args.max_it > 0:
        max_it = args.max_it

    model_selection_metric = cfg["training"]["model_selection_metric"]
    if cfg["training"]["model_selection_mode"] == "maximize":
        model_selection_sign = 1
    elif cfg["training"]["model_selection_mode"] == "minimize":
        model_selection_sign = -1
    else:
        raise ValueError("model_selection_mode must be " "either maximize or minimize.")

    create_folder(out_dir, remove_exists=args.fresh_start)
    create_folder(vis_dir, remove_exists=args.fresh_start)

    # copy config to output directory
    shutil.copyfile(user_cfg_path, os.path.join(out_dir, "config.yaml"))

    # Dataset
    train_loader = loader_utils.get_plab_loader(cfg, env_name, "train")
    val_loader = loader_utils.get_plab_loader(cfg, env_name, "val")

    # Model
    model = config.get_model(cfg, device=device)
    print(model)

    # Generator
    generator = config.get_generator(model, cfg, device=device)

    # Intialize training
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer: Trainer = config.get_trainer(model, optimizer, cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

    if stage == 2:
        ckpt_file = cfg["training"]["ckpt_file"]
        if cfg["training"].get("load_model_only", True):
            checkpoint_io.load_model_only(ckpt_file)
        else:
            load_dict = checkpoint_io.load(ckpt_file)
            it = load_dict.get("it", 0)
            print(f"==> loading vae stage 1 iteration: {it}")
            print(f"==> current learning rate: {optimizer.param_groups[0]['lr']}")

    try:
        load_dict = checkpoint_io.load("model.pt")
    except FileExistsError:
        load_dict = dict()

    epoch_it = load_dict.get("epoch_it", 0)
    it = load_dict.get("it", 0)

    metric_val_best = load_dict.get("loss_val_best", -model_selection_sign * np.inf)

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf
    print(
        "Current best validation metric (%s): %.8f"
        % (model_selection_metric, metric_val_best)
    )

    # Shorthands
    print_every = cfg["training"]["print_every"]
    checkpoint_every = cfg["training"]["checkpoint_every"]
    validate_every = cfg["training"]["validate_every"]
    visualize_every = cfg["training"]["visualize_every"]

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())

    print("Total number of parameters: %d" % nparameters)
    print("output path: ", cfg["training"]["out_dir"])

    # Visualizations
    data_vis_list = []
    vis_dataset = loader_utils.get_plab_dataset(cfg, env_name, split="vis")
    # Build a data dictionary for visualization
    np.random.seed(0)
    vis_inds = np.random.randint(
        len(vis_dataset), size=cfg["training"]["visualize_total"]
    )
    for i, idx in enumerate(vis_inds):
        data_vis = loader_utils.collate_pair_fn([vis_dataset[idx]])
        data_vis_list.append({"it": i, "data": data_vis})

    while True:
        epoch_it += 1
        trainer.epoch_step()

        for batch in train_loader:
            it += 1
            losses = trainer.train_step(batch)

            metrics = {f"train/{k}": v for k, v in losses.items()}
            wandb.log(metrics)

            # Print output
            if (it % print_every) == 0:
                t = datetime.datetime.now()
                print_str = f"[Epoch {epoch_it:04d}] it={it:04d}, time: {time.time() - t0:.3f}, "
                print_str += f"{t.hour:02d}:{t.minute:02d}, "
                for k, v in losses.items():
                    print_str += f"{k}:{v:.4f}, "
                print(print_str)

            # Visualize output
            if visualize_every > 0 and (it % visualize_every) == 0:
                print("Visualizing...")
                for data_vis in data_vis_list:
                    out = generator.generate_mesh(
                        data_vis["data"],
                        use_vae=cfg["training"].get("train_vae", False),
                    )
                    # Get statistics
                    try:
                        meshes, stats_dict = out
                    except TypeError:
                        meshes, stats_dict = out, {}
                    for mesh_name, mesh in meshes.items():
                        filename = f'{data_vis["it"]}_{mesh_name}.off'
                        mesh.export(os.path.join(vis_dir, filename))

            # Save checkpoint
            if checkpoint_every > 0 and (it % checkpoint_every) == 0:
                print("Saving checkpoint")
                checkpoint_io.save(
                    "model.pt", epoch_it=epoch_it, it=it, loss_val_best=metric_val_best
                )

            # Backup if necessary
            if backup_every > 0 and (it % backup_every) == 0:
                print("Backup checkpoint")
                checkpoint_io.save(
                    "model_%d.pt" % it,
                    epoch_it=epoch_it,
                    it=it,
                    loss_val_best=metric_val_best,
                )

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0:
                eval_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
                print(
                    "Validation metric (%s): %.4f"
                    % (model_selection_metric, metric_val)
                )

                metrics = {f"val/{k}": v for k, v in eval_dict.items()}
                wandb.log(metrics)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    print("New best model (loss %.4f)" % metric_val_best)
                    checkpoint_io.save(
                        "model_best.pt",
                        epoch_it=epoch_it,
                        it=it,
                        loss_val_best=metric_val_best,
                    )

            if args.debug:
                exit(0)

        # Exit if necessary
        if trainer.step_it >= max_it:
            exit(0)


if __name__ == "__main__":
    args = get_args()
    main(args)
