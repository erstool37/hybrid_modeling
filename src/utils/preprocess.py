import pandas as pd

dir = "../../dataset/dataset.csv"
save_dir = "../../dataset/statistics.csv"

ds = pd.read_csv(dir)
train_size = int(len(ds) * 0.8)
ds = ds.iloc[:train_size]

stats = pd.DataFrame({'mean': ds.mean(), 'std': ds.std()}).T
stats.to_csv(save_dir, index=False)