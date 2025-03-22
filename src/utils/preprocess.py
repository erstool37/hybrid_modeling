import pandas as pd

dir = "../../dataset/dataset.csv"
save_dir_train = "../../dataset/statistics_train.csv"
save_dir_val = "../../dataset/statistics_val.csv"

ds = pd.read_csv(dir)

# train stats
train_size = int(len(ds) * 0.8)
ds = ds.iloc[:train_size]
stats = pd.DataFrame({'mean': ds.mean(), 'std': ds.std(), 'max': ds.max(), 'min': ds.min()}).T
stats.to_csv(save_dir_train, index=True)

# val stats
val_size = int(len(ds) * 0.95)
ds_inf = ds.iloc[val_size:]
stats = pd.DataFrame({'mean': ds_inf.mean(), 'std': ds_inf.std(), 'max': ds_inf.max(), 'min': ds_inf.min()}).T
stats.to_csv(save_dir_val, index=True)