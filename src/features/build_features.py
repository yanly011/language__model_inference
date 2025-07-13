def build_features(df, cfg):
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels
