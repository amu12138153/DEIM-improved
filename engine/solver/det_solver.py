"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


def get_map_50_95(test_stats):
    """Read COCO mAP50:95 from the new or legacy evaluation output."""
    if "map_50_95" in test_stats:
        return float(test_stats["map_50_95"])

    if "coco_eval_bbox" in test_stats:
        bbox_stats = test_stats["coco_eval_bbox"]
        if isinstance(bbox_stats, (list, tuple)) and bbox_stats:
            return float(bbox_stats[0])

    raise KeyError(
        "test_stats does not contain map_50_95 or coco_eval_bbox."
    )


class DetSolver(BaseSolver):

    def fit(self):
        self.train()
        args = self.cfg#自定义的yaml 参数

        _, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)

        for index, (name, parameter) in enumerate(
            self.model.named_parameters()
        ):
            if index in [194, 195]:
                print(
                    f"Index {index}: {name} - "
                    f"requires_grad: {parameter.requires_grad}"
                )

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print(
                "     ## Using Self-defined Scheduler-{} ## ".format(
                    args.lrsheduler
                )
            )
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                args.lr_gamma,
                iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch,
            )
            self.self_lr_scheduler = True

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        non_trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if not parameter.requires_grad
        )

        print(
            f"number of trainable parameters: {trainable_parameters}"
        )
        print(
            "number of non-trainable parameters: "
            f"{non_trainable_parameters}"
        )
        #         # =====================================================
        # # 3.1 模型各部分参数量统计
        # # =====================================================

        # # 兼容普通模型和 DistributedDataParallel
        # base_model = (
        #     self.model.module
        #     if hasattr(self.model, "module")
        #     else self.model
        # )

        # def count_module_parameters(module):
        #     total = sum(
        #         parameter.numel()
        #         for parameter in module.parameters()
        #     )

        #     trainable = sum(
        #         parameter.numel()
        #         for parameter in module.parameters()
        #         if parameter.requires_grad
        #     )

        #     return total, trainable

        # print(
        #     "\n========== PARAMETER BREAKDOWN =========="
        # )

        # for module_name in (
        #     "backbone",
        #     "encoder",
        #     "decoder",
        # ):
        #     module = getattr(
        #         base_model,
        #         module_name
        #     )

        #     module_total, module_trainable = (
        #         count_module_parameters(module)
        #     )

        #     print(
        #         f"{module_name:10s}"
        #         f" total: {module_total:10,d}"
        #         f"  ({module_total / 1e6:.6f} M)"
        #         f" | trainable: {module_trainable:10,d}"
        #         f"  ({module_trainable / 1e6:.6f} M)"
        #     )

        # model_total, model_trainable = (
        #     count_module_parameters(
        #         base_model
        #     )
        # )

        # print("-" * 76)

        # print(
        #     f"{'model':10s}"
        #     f" total: {model_total:10,d}"
        #     f"  ({model_total / 1e6:.6f} M)"
        #     f" | trainable: {model_trainable:10,d}"
        #     f"  ({model_trainable / 1e6:.6f} M)"
        # )

        # # =====================================================
        # # 3.2 检查是否仍然存在 water 参数
        # # =====================================================
        # water_parameter_rows = [
        #     (
        #         name,
        #         parameter.numel(),
        #         parameter.requires_grad,
        #     )
        #     for name, parameter
        #     in base_model.named_parameters()
        #     if "water" in name.lower()
        # ]

        # water_total = sum(
        #     parameter_count
        #     for (
        #         _,
        #         parameter_count,
        #         _,
        #     ) in water_parameter_rows
        # )

        # water_trainable = sum(
        #     parameter_count
        #     for (
        #         _,
        #         parameter_count,
        #         requires_grad,
        #     ) in water_parameter_rows
        #     if requires_grad
        # )

        # print("-" * 76)

        # print(
        #     f"{'water':10s}"
        #     f" total: {water_total:10,d}"
        #     f"  ({water_total / 1e6:.6f} M)"
        #     f" | trainable: {water_trainable:10,d}"
        #     f"  ({water_trainable / 1e6:.6f} M)"
        # )

        # if water_parameter_rows:
        #     print("\nWater parameters:")

        #     for (
        #         name,
        #         parameter_count,
        #         requires_grad,
        #     ) in water_parameter_rows:
        #         print(
        #             f"{parameter_count:10,d}"
        #             f"  trainable={requires_grad}"
        #             f"  {name}"
        #         )
        # else:
        #     print(
        #         "\nNo water parameters exist "
        #         "in the current model."
        #     )

        # # =====================================================
        # # 3.3 找出参数量最大的具体层
        # # =====================================================
        # direct_parameter_rows = []

        # for name, module in (
        #     base_model.named_modules()
        # ):
        #     direct_total = sum(
        #         parameter.numel()
        #         for parameter
        #         in module.parameters(
        #             recurse=False
        #         )
        #     )

        #     direct_trainable = sum(
        #         parameter.numel()
        #         for parameter
        #         in module.parameters(
        #             recurse=False
        #         )
        #         if parameter.requires_grad
        #     )

        #     if direct_total > 0:
        #         direct_parameter_rows.append(
        #             (
        #                 direct_total,
        #                 direct_trainable,
        #                 name,
        #                 module.__class__.__name__,
        #             )
        #         )

        # direct_parameter_rows.sort(
        #     key=lambda row: row[0],
        #     reverse=True,
        # )

        # print(
        #     "\n========== TOP PARAMETER LAYERS =========="
        # )

        # for (
        #     direct_total,
        #     direct_trainable,
        #     name,
        #     class_name,
        # ) in direct_parameter_rows[:40]:
        #     print(
        #         f"{direct_total:10,d}"
        #         f" | trainable: {direct_trainable:10,d}"
        #         f" | {class_name:25s}"
        #         f" | {name}"
        #     )

        # print(
        #     "==========================================\n"
        # )

        metric_key = "coco_eval_bbox"
        top1 = 0.0
        best_stat = {"epoch": -1}

        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, _ = evaluate(
                model=module,
                criterion=self.criterion,
                postprocessor=self.postprocessor,
                data_loader=self.val_dataloader,
                coco_evaluator=self.evaluator,
                device=self.device,
                epoch=self.last_epoch,
                writer=self.writer,
            )

            current_score = get_map_50_95(test_stats)
            best_stat["epoch"] = self.last_epoch
            best_stat[metric_key] = current_score
            top1 = current_score
            print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        for epoch in range(start_epoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)

            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            stop_epoch = self.train_dataloader.collate_fn.stop_epoch

            if epoch == stop_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()

                if self.output_dir:
                    self.load_resume_state(
                        str(self.output_dir / "best_stg1.pth")
                    )

                if self.ema is not None:
                    self.ema.decay = (
                        self.train_dataloader
                        .collate_fn
                        .ema_restart_decay
                    )
                    print(
                        f"Refresh EMA at epoch {epoch} "
                        f"with decay {self.ema.decay}"
                    )

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
            )

            if not self.self_lr_scheduler:
                if (
                    self.lr_warmup_scheduler is None
                    or self.lr_warmup_scheduler.finished()
                ):
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]

                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch:04}.pth"
                    )

                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(
                        self.state_dict(),
                        checkpoint_path,
                    )

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                model=module,
                criterion=self.criterion,
                postprocessor=self.postprocessor,
                data_loader=self.val_dataloader,
                coco_evaluator=self.evaluator,
                device=self.device,
                epoch=epoch,
                writer=self.writer,
            )

            current_score = get_map_50_95(test_stats)

            if (
                self.writer is not None
                and dist_utils.is_main_process()
                and "coco_eval_bbox" in test_stats
            ):
                for index, value in enumerate(
                    test_stats["coco_eval_bbox"]
                ):
                    self.writer.add_scalar(
                        f"Test/coco_eval_bbox_{index}",
                        float(value),
                        epoch,
                    )

            if metric_key in best_stat:
                if current_score > best_stat[metric_key]:
                    best_stat["epoch"] = epoch

                best_stat[metric_key] = max(
                    best_stat[metric_key],
                    current_score,
                )
            else:
                best_stat["epoch"] = epoch
                best_stat[metric_key] = current_score

            if best_stat[metric_key] > top1:
                best_stat_print["epoch"] = epoch
                top1 = best_stat[metric_key]

                if self.output_dir:
                    checkpoint_name = (
                        "best_stg2.pth"
                        if epoch >= stop_epoch
                        else "best_stg1.pth"
                    )
                    dist_utils.save_on_master(
                        self.state_dict(),
                        self.output_dir / checkpoint_name,
                    )

            best_stat_print[metric_key] = max(
                best_stat[metric_key],
                top1,
            )
            print(
                f"best_stat: {best_stat_print}, "
                f"current mAP50:95: {current_score:.6f}"
            )

            if best_stat["epoch"] == epoch and self.output_dir:
                if epoch >= stop_epoch:
                    if current_score > top1:
                        top1 = current_score
                        dist_utils.save_on_master(
                            self.state_dict(),
                            self.output_dir / "best_stg2.pth",
                        )
                else:
                    top1 = max(current_score, top1)
                    dist_utils.save_on_master(
                        self.state_dict(),
                        self.output_dir / "best_stg1.pth",
                    )

            elif epoch >= stop_epoch:
                best_stat = {"epoch": -1}

                if self.ema is not None:
                    self.ema.decay -= 0.0001

                if self.output_dir:
                    self.load_resume_state(
                        str(self.output_dir / "best_stg1.pth")
                    )

                if self.ema is not None:
                    print(
                        f"Refresh EMA at epoch {epoch} "
                        f"with decay {self.ema.decay}"
                    )

            log_stats = {
                **{
                    f"train_{key}": value
                    for key, value in train_stats.items()
                },
                **{
                    f"test_{key}": value
                    for key, value in test_stats.items()
                },
                "epoch": epoch,
                "n_parameters": trainable_parameters,
                "non_trainable_parameters": (
                    non_trainable_parameters
                ),
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open(
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(
                        exist_ok=True
                    )

                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]

                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")

                        for name in filenames:
                            torch.save(
                                coco_evaluator
                                .coco_eval["bbox"]
                                .eval,
                                self.output_dir / "eval" / name,
                            )

            if self.writer is not None and dist_utils.is_main_process():
                self.writer.flush()

        total_time = time.time() - start_time
        total_time_str = str(
            datetime.timedelta(seconds=int(total_time))
        )
        print(f"Training time {total_time_str}")

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        eval_epoch = max(int(getattr(self, "last_epoch", 0)), 0)

        test_stats, coco_evaluator = evaluate(
            model=module,
            criterion=self.criterion,
            postprocessor=self.postprocessor,
            data_loader=self.val_dataloader,
            coco_evaluator=self.evaluator,
            device=self.device,
            epoch=eval_epoch,
            writer=self.writer,
        )

        if (
            self.output_dir
            and coco_evaluator is not None
            and "bbox" in coco_evaluator.coco_eval
        ):
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval,
                self.output_dir / "eval.pth",
            )

        if self.writer is not None and dist_utils.is_main_process():
            self.writer.flush()

        return test_stats
