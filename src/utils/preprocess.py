import pandas as pd

dir = "../../dataset/dataset.csv"
save_dir = "../../dataset/statistics.csv"
save_dir_inf = "../../dataset/statistics_val.csv"

ds = pd.read_csv(dir)
# train_size = int(len(ds) * 0.8)
# ds = ds.iloc[:train_size]
# stats = pd.DataFrame({'mean': ds.mean(), 'std': ds.std()}).T
# stats.to_csv(save_dir, index=False)

val_size= int(len(ds) * 0.95)
ds_inf = ds.iloc[val_size:]

stats = pd.DataFrame({'mean': ds.mean(), 'std': ds.std()}).T
stats.to_csv(save_dir_inf, index=False)