import pandas as pd

def load_and_prepare_data(cfg):
    train = pd.read_csv(cfg.train_file)
    val = pd.read_csv(cfg.val_file)
    test = pd.read_csv(cfg.test_file)
    return train, val, test
