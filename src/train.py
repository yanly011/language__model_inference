import hydra
from omegaconf import DictConfig, OmegaConf
from data.make_dataset import load_and_prepare_data
from features.build_features import build_features
from models.train_model import train_and_evaluate
import wandb

@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # 初始化wandb
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    import os

    def get_abs_path(rel_path):
        # 计算项目根目录，假设src/下是你的train.py
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, rel_path)

    train_file = get_abs_path(cfg.train_file)
    val_file = get_abs_path(cfg.val_file)
    test_file = get_abs_path(cfg.test_file)
    import pandas as pd
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    # train_df, val_df, test_df = load_and_prepare_data(cfg)
    train_texts, train_labels = build_features(train_df, cfg)
    val_texts, val_labels = build_features(val_df, cfg)

    model, tokenizer, results = train_and_evaluate(
        train_texts, train_labels, val_texts, val_labels, cfg
    )
    print("评估指标：", results)
    wandb.finish()

if __name__ == "__main__":
    main()

