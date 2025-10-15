import pandas as pd

df = pd.read_csv("skin_data_info_split.csv")

df['set'] = df['tag']
stats = df.groupby(['set', 'category']).size()

train_counts = stats['train'].values
val_counts = stats['val'].values
test_counts = stats['test'].values
# 每個數字欄位寬度（根據最大數決定）
width = max(train_counts.max(), val_counts.max())  # 轉為二進位長度
width = max(3, len(str(max(train_counts.max(), val_counts.max()))))  # 至少寬度3

train_line = "Train:" + "".join(f"{x:>{width}} " for x in train_counts)
val_line   = "Val:  " + "".join(f"{x:>{width}} " for x in val_counts)
test_line   = "Test:  " + "".join(f"{x:>{width}} " for x in test_counts)

print(train_line)
print(val_line)
print(test_line)

# 總結
print(f"\nTotal train cases: {train_counts.sum()}")
print(f"Total val cases:   {val_counts.sum()}")
print(f"Total test cases:   {test_counts.sum()}")
print(f"Total all cases:   {train_counts.sum() + val_counts.sum() + test_counts.sum()}")
