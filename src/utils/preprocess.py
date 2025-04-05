import pandas as pd

dir = "dataset/dataset.csv"
test_dir = "dataset/test.csv"

save_dir_train = "src/utils/stats/train.csv"
save_dir_val = "src/utils/stats/val.csv"
save_dir_total = "src/utils/stats/total.csv"
save_dir_test = "src/utils/stats/test.csv"

ds = pd.read_csv(dir)
ds_test = ds = pd.read_csv(test_dir)

# train stats
train_size = int(len(ds) * 0.8)
ds_train = ds.iloc[:train_size]
stats = pd.DataFrame({'mean': ds_train.mean(), 'std': ds_train.std(), 'max': ds_train.max(), 'min': ds_train.min()}).T
stats.to_csv(save_dir_train, index=True)

# val stats
val_size = int(len(ds) * 0.8)
ds_inf = ds.iloc[val_size:]
stats = pd.DataFrame({'mean': ds_inf.mean(), 'std': ds_inf.std(), 'max': ds_inf.max(), 'min': ds_inf.min()}).T
stats.to_csv(save_dir_val, index=True)

# total stats
ds_total = ds.iloc[:]
stats = pd.DataFrame({'mean': ds_total.mean(), 'std': ds_total.std(), 'max': ds_total.max(), 'min': ds_total.min()}).T
stats.to_csv(save_dir_total, index=True)

# test stats
ds_test = ds_test.iloc[:]
stats = pd.DataFrame({'mean': ds_test.mean(), 'std': ds_test.std(), 'max': ds_test.max(), 'min': ds_test.min()}).T
stats.to_csv(save_dir_test, index=True)