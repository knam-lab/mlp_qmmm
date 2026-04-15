#!/usr/bin/env python3
# src/mlp_qmmm/training.py

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import random_split

from mlp_qmmm.a2_parser import parse_dataset
from mlp_qmmm.b_nn_loader import load_model
from mlp_qmmm.trainer_utils import (
    EV_TO_KCALMOL,
    ProgressTee,
    QMMMDataset,
    build_mask_picker,
    build_scheduler,
    clip_gradients,
    compute_losses,
    eval_split,
    forward_batch,
    init_csv,
    make_group_weight_fn,
    make_loss,
    make_standard_loader,
    make_variable_qm_loader,
    move_batch_to_device,
    prepare_frames_with_model,
    resolve_train_dir,
    sanitize_padded_mm_inplace,
    save_script_checkpoint,
    set_seed,
    step_scheduler,
    write_csv_row,
    write_training_summary,
    _unit_multiplier,
)


def run(cfg: Dict[str, Any]) -> None:  # noqa: C901
    trcfg = cfg.get("trainer", {}) or {}

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    if "torch_num_threads" in trcfg:
        torch.set_num_threads(int(trcfg["torch_num_threads"]))
    if "torch_interop_threads" in trcfg:
        torch.set_num_interop_threads(int(trcfg["torch_interop_threads"]))

    device = torch.device(trcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(trcfg.get("seed", 42))
    set_seed(seed)

    out_dir = resolve_train_dir(cfg)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    tee = ProgressTee(os.path.join(out_dir, "progress.txt"))
    tee.log(f"[train] out_dir  = {out_dir}")
    tee.log(f"[train] device   = {device}")
    tee.log(f"[train] seed     = {seed}")

    print_units = str(trcfg.get("print_units", "eV"))
    print_detail = str(trcfg.get("print_detail", "short")).lower()
    print_keys: Optional[List[str]] = trcfg.get("print_keys", None)

    mm_equal_frame_weight = bool(trcfg.get("mm_equal_frame_weight", True))
    qm_equal_frame_weight = bool(trcfg.get("qm_equal_frame_weight", False))
    variable_qm = bool(trcfg.get("variable_qm", False))
    tee.log(f"[train] variable_qm = {variable_qm}")

    tee.log("[train] loading model ...")
    mio = load_model(cfg, device=device, verbose=True)
    model = mio.model
    core = mio.core
    required_keys = list(mio.required_keys)
    output_keys = list(mio.output_keys)
    allowed_loss_keys = list(mio.allowed_loss_keys)

    initial_stats = {
        "parameters_total": sum(p.numel() for p in core.parameters()),
        "parameters_trainable": sum(p.numel() for p in core.parameters() if p.requires_grad),
        "buffers": sum(b.numel() for b in core.buffers()),
    }

    tee.log(
        f"[train] model '{(cfg.get('model', {}) or {}).get('name', '?')}' "
        f"params={initial_stats['parameters_total']:,} "
        f"trainable={initial_stats['parameters_trainable']:,}"
    )
    tee.log(f"[train] required_keys      : {required_keys}")
    tee.log(f"[train] output_keys        : {output_keys}")
    tee.log(f"[train] allowed_loss_keys  : {allowed_loss_keys}")

    prep_output_keys = list(getattr(type(core), "PREP_OUTPUT_KEYS", ()) or [])
    if mio.prepare_inputs is not None:
        tee.log(f"[train] prepare_inputs available; prep_output_keys={prep_output_keys}")

    losses_cfg: Dict[str, Dict[str, Any]] = cfg.get("losses", {}) or {}
    if not losses_cfg:
        raise ValueError("Config missing 'losses'. Specify at least one target key.")

    requested_loss_keys = sorted(losses_cfg.keys())
    bad_model = [k for k in requested_loss_keys if k not in set(allowed_loss_keys)]
    if bad_model:
        raise KeyError(
            "Requested loss keys not in model allowed_loss_keys:\n"
            f"  requested : {bad_model}\n"
            f"  available : {sorted(set(allowed_loss_keys))}"
        )

    mcfg = cfg.get("model", {}) or {}
    max_qm_cfg = int(mcfg.get("max_qm", 100) or 100)
    if "max_qm" not in mcfg:
        warnings.warn(
            "[train] model.max_qm not set; defaulting to 100. "
            "Set cfg['model']['max_qm'] explicitly for reproducibility."
        )

    mm_pad_to = (cfg.get("adapter_kwargs", {}) or {}).get("max_mm", None)
    if mm_pad_to is None:
        mm_pad_to = 5000
        warnings.warn(
            "[train] adapter_kwargs.max_mm not set; parser will default MM padding to 5000."
        )
    else:
        mm_pad_to = int(mm_pad_to)
        if mm_pad_to <= 0:
            raise ValueError(f"adapter_kwargs.max_mm must be > 0, got {mm_pad_to}.")

    tee.log("[train] parsing dataset ...")
    frames = parse_dataset(
        adapter=str(cfg.get("adapter", "")),
        path=cfg.get("input"),
        adapter_kwargs=cfg.get("adapter_kwargs", {}),
        postprocess_kwargs=cfg.get("postprocess", {}),
        out_dir=out_dir,
        verbose=bool(trcfg.get("verbose_parse", False)),
        required_keys_all_frames=requested_loss_keys,
        required_keys_warn_only=bool(trcfg.get("warn_partial_loss_keys", False)),
        qm_pad_to=(max_qm_cfg if max_qm_cfg > 0 else None),
        mm_pad_to=mm_pad_to,
    )
    tee.log(f"[train] parsed {len(frames)} frames")

    if mio.prepare_inputs is not None:
        tee.log("[train] running model-specific input preparation ...")
        prep_info = prepare_frames_with_model(
            frames,
            mio.prepare_inputs,
            cfg,
            prep_output_keys=prep_output_keys,
            verbose=bool(trcfg.get("verbose_prepare", False)),
        )
        tee.log(
            f"[train] prepare_inputs wrote {prep_info.get('keys_written', [])} "
            f"on {prep_info.get('frames_prepared', 0)} frame(s)"
        )

    missing_input_counts = {
        k: sum(1 for f in frames if k not in f) for k in required_keys
    }
    missing_inputs = {k: c for k, c in missing_input_counts.items() if c > 0}
    if missing_inputs:
        raise KeyError(
            "Required model input keys missing after parsing/preparation:\n"
            + "\n".join(
                f"  - {k}: missing in {c}/{len(frames)} frames"
                for k, c in sorted(missing_inputs.items())
            )
        )

    dataset_mode = str(trcfg.get("dataset_mode", "stacked")).lower()
    dataset = QMMMDataset(frames, input_keys=required_keys, mode=dataset_mode)
    n_total = len(dataset)
    if n_total < 3:
        raise ValueError(f"Need at least 3 frames for train/val/test split; got {n_total}.")

    have_targets = set(dataset.available_target_keys())
    bad_data = [k for k in requested_loss_keys if k not in have_targets]
    if bad_data:
        raise KeyError(
            "Requested loss keys not found in dataset targets present in all frames:\n"
            f"  missing   : {bad_data}\n"
            f"  available : {sorted(have_targets)}"
        )

    active_loss_keys = [k for k in requested_loss_keys if (k in have_targets and k in set(output_keys))]
    inactive_loss_keys = [k for k in requested_loss_keys if k not in active_loss_keys]

    tee.log(f"[train] requested loss keys : {requested_loss_keys}")
    tee.log(f"[train] dataset target keys : {sorted(have_targets)}")
    tee.log(f"[train] active loss keys    : {active_loss_keys}")

    if inactive_loss_keys:
        raise KeyError(
            "Under exact-match policy, some requested loss keys are inactive:\n"
            + "\n".join(f"  - {k}" for k in inactive_loss_keys)
        )

    val_frac = float(trcfg.get("val_fraction", 0.2))
    test_frac = float(trcfg.get("test_fraction", 0.1))
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    n_test = int(round(n_total * test_frac))
    n_val = int(round(n_total * val_frac))
    n_test = min(max(n_test, 1), max(n_total - 2, 1))
    n_val = min(max(n_val, 1), max(n_total - n_test - 1, 1))
    n_train = n_total - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError(f"Could not form a valid split for n_total={n_total}.")

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)
    tee.log(f"[train] split: train={n_train} val={n_val} test={n_test}")

    bs = int(trcfg.get("batch_size", 16))
    nw = int(trcfg.get("num_workers", 0))
    pin = bool(trcfg.get("pin_memory", True))
    pw = bool(nw > 0) and bool(trcfg.get("persistent_workers", False))

    if dataset_mode == "lazy" and nw > 0:
        warnings.warn(
            "[train] dataset_mode='lazy' with num_workers > 0 may replicate the "
            "in-memory frame list across workers and increase RAM use."
        )

    train_indices = list(train_set.indices) if hasattr(train_set, "indices") else list(range(len(train_set)))
    val_indices = list(val_set.indices) if hasattr(val_set, "indices") else list(range(len(val_set)))
    test_indices = list(test_set.indices) if hasattr(test_set, "indices") else list(range(len(test_set)))

    if not variable_qm:
        tee.log("[train] loader mode: standard")
        train_loader = make_standard_loader(
            train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=pw
        )
        val_loader = make_standard_loader(
            val_set, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=pw
        )
        test_loader = make_standard_loader(
            test_set, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=pw
        )
    else:
        tee.log("[train] loader mode: variable_qm / bucketed")
        train_loader = make_variable_qm_loader(
            train_set, dataset,
            local_indices=train_indices, batch_size=bs, shuffle=True, seed=seed,
            num_workers=nw, pin_memory=pin, persistent_workers=pw
        )
        val_loader = make_variable_qm_loader(
            val_set, dataset,
            local_indices=val_indices, batch_size=bs, shuffle=False, seed=seed,
            num_workers=nw, pin_memory=pin, persistent_workers=pw
        )
        test_loader = make_variable_qm_loader(
            test_set, dataset,
            local_indices=test_indices, batch_size=bs, shuffle=False, seed=seed,
            num_workers=nw, pin_memory=pin, persistent_workers=pw
        )

    criterions: Dict[str, nn.Module] = {
        n: make_loss(spec.get("type", "mse")) for n, spec in losses_cfg.items()
    }
    base_weights: Dict[str, float] = {
        n: float(spec.get("weight", 1.0)) for n, spec in losses_cfg.items()
    }

    pick_mask = build_mask_picker(cfg)

    ocfg = cfg.get("optim", {}) or {}
    lr0 = float(ocfg.get("lr", 5e-4))
    wd = float(ocfg.get("weight_decay", 0.0))
    opt_name = str(ocfg.get("name", "adam")).lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(core.parameters(), lr=lr0, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(core.parameters(), lr=lr0, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'. Use 'adam' or 'adamw'.")

    epochs = int(trcfg.get("epochs", 500))
    sched_bundle = build_scheduler(
        optimizer, ocfg, epochs=epochs, steps_per_epoch=max(1, len(train_loader)), start_lr=lr0
    )
    start_lr = sched_bundle.start_lr
    group_weight_fn = make_group_weight_fn(cfg, start_lr=start_lr)

    amp_cfg = str(trcfg.get("amp", "off")).lower()
    use_amp = False
    amp_mode = "off"
    amp_dtype = None

    if device.type == "cuda":
        if amp_cfg == "bf16" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            use_amp, amp_mode, amp_dtype = True, "bf16", torch.bfloat16
        elif amp_cfg == "fp16":
            use_amp, amp_mode, amp_dtype = True, "fp16", torch.float16
            warnings.warn(
                "[train] amp='fp16' with autograd.grad models is high risk; prefer bf16.",
                stacklevel=2,
            )
    elif amp_cfg in ("bf16", "fp16"):
        tee.log("[warn] AMP requested on non-CUDA device → disabled.")

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_mode == "fp16"))
    tee.log(f"[train] amp={amp_mode}")

    ckpt_path = os.path.join(out_dir, "best.pt")
    restart_path = os.path.join(out_dir, "restart.pt")
    save_restart_every = int(trcfg.get("save_restart_every", 0))

    best_val_unw = float("inf")
    es_pat = int(trcfg.get("early_stop_patience", 20))
    es_min_delta = float(trcfg.get("early_stop_min_delta", 1e-4))
    no_improve = 0
    best_epoch: Optional[int] = None
    start_epoch = 1

    restart_ckpt = str(trcfg.get("restart_ckpt", "") or "").strip()
    if restart_ckpt and os.path.exists(restart_ckpt):
        tee.log(f"[train] restarting from {restart_ckpt}")
        rs = torch.load(restart_ckpt, map_location=device)
        core.load_state_dict(rs["model_state"])
        optimizer.load_state_dict(rs["optimizer_state"])
        if sched_bundle.scheduler is not None and "scheduler_state" in rs:
            sched_bundle.scheduler.load_state_dict(rs["scheduler_state"])
        start_epoch = int(rs.get("epoch", 0)) + 1
        best_val_unw = float(rs.get("best_val_unw", float("inf")))
        no_improve = int(rs.get("no_improve", 0))
        best_epoch = rs.get("best_epoch", None)

    goal_cfg = trcfg.get("goal_stop", {}) or {}
    goal_enabled = bool(goal_cfg.get("enabled", False))
    goal_every = int(goal_cfg.get("check_every", 1))
    goal_pat = int(goal_cfg.get("patience", 1))
    goal_units = str(goal_cfg.get("units", "kcal")).lower()
    goal_raw = {k: float(v) for k, v in (goal_cfg.get("thresholds", {}) or {}).items()}
    goal_thr_ev = {
        k: (v / EV_TO_KCALMOL if goal_units.startswith("kcal") else v)
        for k, v in goal_raw.items()
    }
    goal_counters: Dict[str, int] = {k: 0 for k in goal_thr_ev.keys()}

    csv_path = os.path.join(logs_dir, "train_val_metrics.csv")
    init_csv(csv_path, criterions)

    with open(os.path.join(out_dir, "model_io.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "required_keys": required_keys,
                "output_keys": output_keys,
                "allowed_loss_keys": allowed_loss_keys,
                "loss_keys": sorted(criterions.keys()),
                "prep_output_keys": prep_output_keys,
                "has_prepare_inputs": bool(mio.prepare_inputs),
                "masks": cfg.get("masks", {}) or {},
                "amp": amp_mode,
                "device": str(device),
                "variable_qm": variable_qm,
                "dataset_mode": dataset_mode,
            },
            f,
            indent=2,
        )

    max_norm = float(trcfg.get("grad_clip", 5.0))
    save_script_every = int(trcfg.get("save_script_every", 0))

    wall_start = time.time()
    per_epoch_seconds: List[float] = []

    def _fmt_key(k: str, rmse_d: Dict[str, float], mean_d: Dict[str, float]) -> str:
        mul = _unit_multiplier(k, print_units)
        rmse = rmse_d.get(k, float("nan")) * mul
        unit = "kcal/mol" if print_units.lower().startswith("kcal") else "eV"
        if "grad" in k.lower():
            unit += "(grad)"
        if print_detail == "full":
            mean = mean_d.get(k, float("nan")) * mul
            return f"{k}:RMSE={rmse:.5f} {unit} (mean={mean:.3e})"
        return f"{k}:RMSE={rmse:.5f} {unit}"

    tee.log(f"[train] starting epoch {start_epoch} → {epochs}")

    for epoch in range(start_epoch, epochs + 1):
        if variable_qm:
            for loader in (train_loader, val_loader, test_loader):
                sampler = getattr(loader, "batch_sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

        epoch_t0 = time.time()
        core.train()

        tr_loss_sum = 0.0
        tr_steps = 0
        tr_raw_acc: Dict[str, float] = {k: 0.0 for k in criterions.keys()}
        tr_rmse_acc: Dict[str, float] = {k: 0.0 for k in criterions.keys()}
        tr_seen: Dict[str, int] = {k: 0 for k in criterions.keys()}
        clip_hits = 0
        last_total_norm = float("nan")

        for cpu_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            batch = move_batch_to_device(cpu_batch, device)
            sanitize_padded_mm_inplace(batch)

            lr_now = float(optimizer.param_groups[0]["lr"])

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = forward_batch(core, batch, required_keys)
                    loss, raw_logs, rmses, _ = compute_losses(
                        pred,
                        batch["targets"],
                        batch,
                        criterions,
                        base_weights,
                        group_weight_fn,
                        pick_mask,
                        lr_now=lr_now,
                        mm_equal_frame_weight=mm_equal_frame_weight,
                        qm_equal_frame_weight=qm_equal_frame_weight,
                        device=device,
                    )
            else:
                pred = forward_batch(core, batch, required_keys)
                loss, raw_logs, rmses, _ = compute_losses(
                    pred,
                    batch["targets"],
                    batch,
                    criterions,
                    base_weights,
                    group_weight_fn,
                    pick_mask,
                    lr_now=lr_now,
                    mm_equal_frame_weight=mm_equal_frame_weight,
                    qm_equal_frame_weight=qm_equal_frame_weight,
                    device=device,
                )

            if use_amp and amp_mode == "fp16":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                last_total_norm, hit = clip_gradients(optimizer, max_norm)
                if hit:
                    clip_hits += 1
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                last_total_norm, hit = clip_gradients(optimizer, max_norm)
                if hit:
                    clip_hits += 1
                optimizer.step()

            if sched_bundle.sched_type == "onecycle" and sched_bundle.scheduler is not None:
                sched_bundle.scheduler.step()

            tr_loss_sum += float(loss.detach().cpu().item())
            tr_steps += 1
            for k, v in raw_logs.items():
                tr_raw_acc[k] += v
                tr_rmse_acc[k] += rmses[k]
                tr_seen[k] += 1

        tr_loss = tr_loss_sum / max(1, tr_steps)
        tr_logs_avg = {k: tr_raw_acc[k] / max(1, tr_seen[k]) for k in tr_raw_acc}
        tr_rmse_avg = {k: tr_rmse_acc[k] / max(1, tr_seen[k]) for k in tr_rmse_acc}

        lr_now = float(optimizer.param_groups[0]["lr"])
        va_w, va_unw, va_logs_avg, va_rmse_avg = eval_split(
            val_loader,
            core,
            required_keys,
            criterions,
            base_weights,
            group_weight_fn,
            pick_mask,
            device,
            lr_now=lr_now,
            mm_equal_frame_weight=mm_equal_frame_weight,
            qm_equal_frame_weight=qm_equal_frame_weight,
        )

        step_scheduler(sched_bundle, optimizer, val_unw=va_unw, tee=tee)

        goal_stop = False
        if goal_enabled and epoch % goal_every == 0:
            all_met = True
            for key, thr in goal_thr_ev.items():
                cur = va_rmse_avg.get(key, float("nan"))
                if math.isnan(cur):
                    all_met = False
                    continue
                if cur <= thr:
                    goal_counters[key] = goal_counters.get(key, 0) + 1
                else:
                    goal_counters[key] = 0
                if goal_counters[key] < goal_pat:
                    all_met = False
            if goal_thr_ev and all_met:
                goal_stop = True
                tee.log(f"[GoalStop] All targets met for {goal_pat} consecutive checks → stopping.")

        lr_now = float(optimizer.param_groups[0]["lr"])
        write_csv_row(
            csv_path, epoch, "train", lr_now, tr_loss, float("nan"),
            tr_logs_avg, tr_rmse_avg, criterions, group_weight_fn, base_weights
        )
        write_csv_row(
            csv_path, epoch, "val", lr_now, va_w, va_unw,
            va_logs_avg, va_rmse_avg, criterions, group_weight_fn, base_weights
        )

        shown_keys = sorted(criterions.keys()) if not print_keys else [k for k in sorted(criterions.keys()) if k in print_keys]
        parts = [
            f"[{epoch:03d}/{epochs}] lr={lr_now:.3e}",
            f"train={tr_loss:.6f}",
            f"val={va_w:.6f}",
            f"val_unw={va_unw:.6f}",
        ]
        parts += [_fmt_key(k, va_rmse_avg, va_logs_avg) for k in shown_keys]
        if clip_hits > 0:
            parts.append(f"[grad-clip hits={clip_hits}, last_total_norm={last_total_norm:.2f} > {max_norm}]")
        tee.log("  ".join(parts))

        improved = va_unw + es_min_delta < best_val_unw
        if improved:
            best_val_unw = va_unw
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": core.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg,
                },
                ckpt_path,
            )
            tee.log(f"[epoch {epoch}] decision: improved (val_unw) → checkpoint saved")
        else:
            no_improve += 1
            tee.log(f"[epoch {epoch}] decision: no_improve (val_unw) = {no_improve}/{es_pat}")

        save_restart = ((save_restart_every > 0 and epoch % save_restart_every == 0) or improved)
        if save_restart:
            rs_state: Dict[str, Any] = {
                "epoch": epoch,
                "model_state": core.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_unw": best_val_unw,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
            }
            if sched_bundle.scheduler is not None:
                rs_state["scheduler_state"] = sched_bundle.scheduler.state_dict()
            torch.save(rs_state, restart_path)

        if save_script_every > 0 and (epoch % save_script_every == 0):
            core.eval()
            save_script_checkpoint(core, out_dir, epoch, tee)
            core.train()

        per_epoch_seconds.append(time.time() - epoch_t0)

        if (no_improve >= es_pat) or goal_stop:
            msg = "[EarlyStop] " + ("goal targets satisfied." if goal_stop else f"no improvement in {es_pat} epochs.")
            tee.log(msg)
            break

    if os.path.exists(ckpt_path):
        tee.log(f"[test] reloading best checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        core.load_state_dict(ckpt["model_state"])

    lr_now = float(optimizer.param_groups[0]["lr"])
    test_w, test_unw, test_metrics, test_rmse = eval_split(
        test_loader,
        core,
        required_keys,
        criterions,
        base_weights,
        group_weight_fn,
        pick_mask,
        device,
        lr_now=lr_now,
        mm_equal_frame_weight=mm_equal_frame_weight,
        qm_equal_frame_weight=qm_equal_frame_weight,
    )

    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss_weighted": test_w,
                "loss_unweighted": test_unw,
                "per_key": test_metrics,
                "per_key_rmse": test_rmse,
            },
            f,
            indent=2,
        )

    core.eval()
    torch.save(core.state_dict(), os.path.join(out_dir, "model_state.pt"))

    script_path = os.path.join(out_dir, "model_script.pt")
    try:
        torch.jit.script(core).save(script_path)
        tee.log(f"[save] TorchScript → {script_path}")
    except Exception as exc:
        tee.log(f"[warn] TorchScript export failed: {type(exc).__name__}: {exc}")
        script_path = ""

    with open(os.path.join(out_dir, "config_used.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    wall_seconds = time.time() - wall_start
    peak_cuda: Optional[int] = None
    if device.type == "cuda":
        try:
            peak_cuda = int(torch.cuda.max_memory_allocated(device))
        except Exception:
            peak_cuda = None

    write_training_summary(
        out_dir,
        cfg,
        device=device,
        amp_mode=amp_mode,
        wall_seconds=wall_seconds,
        wall_start=wall_start,
        per_epoch_seconds=per_epoch_seconds,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_total=n_total,
        batch_size=bs,
        num_workers=nw,
        val_frac=val_frac,
        test_frac=test_frac,
        initial_stats=initial_stats,
        required_keys=required_keys,
        output_keys=output_keys,
        allowed_loss_keys=allowed_loss_keys,
        criterions=criterions,
        optimizer=optimizer,
        sched_bundle=sched_bundle,
        best_epoch=best_epoch,
        best_val_unw=best_val_unw,
        test_w=test_w,
        test_unw=test_unw,
        peak_cuda_bytes=peak_cuda,
        tee=tee,
    )

    tee.log(f"Done. Best val_unw={best_val_unw:.6f}")
    tee.log(f"Artifacts saved in: {out_dir}")
    tee.log(f" - Logs: {csv_path}")
    tee.log(f" - Test metrics: {os.path.join(out_dir, 'test_metrics.json')}")
    tee.log(f" - Model state: {os.path.join(out_dir, 'model_state.pt')}")
    tee.log(f" - TorchScript: {script_path or '(export failed)'}")
    tee.log(f" - Training summary: {os.path.join(out_dir, 'training_summary.json')}")
    tee.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="mlp_qmmm training script")
    ap.add_argument("config", type=str, help="Path to YAML config file")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == "__main__":
    main()