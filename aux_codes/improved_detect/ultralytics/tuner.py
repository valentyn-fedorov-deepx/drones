import optuna
import yaml
from ultralytics.utils import IterableSimpleNamespace

from aux_codes.improved_detect.ultralytics.trainer import CustomTrainer


def run_detector(trial: optuna.Trial):
    with open('aux_codes/improved_detect/train.yaml', 'r') as f:
        train_cfg = yaml.safe_load(f)

    overrides = dict()

    overrides["batch"] = 32
    overrides["data"] = trial.suggest_categorical("data",
                                                  ["aux_codes/improved_detect/data.yaml",
                                                   ""])
    overrides["model"] = trial.suggest_categorical("model_type",
                                                   ['yolov8m.pt', 'yolo11m.pt',
                                                    'yolov8l.pt', 'yolo11l.pt'])
    overrides["lr0"] = trial.suggest_float("lr0", low=1e-5, high=1e-2, log=True)
    overrides["optimizer"] = trial.suggest_categorical("optimizer", ['Adam', "SGD", "AdamW"])
    overrides["warmup_epochs"] = trial.suggest_int('warmup_epochs', 0, 10, 2)
    overrides["freeze"] = trial.suggest_int("freeze", 0, 10, 3)
    overrides["warmup_momentum"] = trial.suggest_float('warmup_momentum', 0.5, 0.9)
    overrides["momentum"] = trial.suggest_float('momentum', 0.85, 0.95)
    overrides["lrf"] = trial.suggest_float('lrf', 0.05, 0.001)
    overrides["single_cls"] = True

    trainer = CustomTrainer(IterableSimpleNamespace(**train_cfg),
                            overrides=overrides)
    trainer.train()

    return


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(run_detector, n_trials=100)
