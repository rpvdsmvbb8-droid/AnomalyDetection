import pandas as pd

# 这里对应报错里的 data_files[3]，也就是第4个文件
filename = "data/02-20-2018.csv"

print(f"正在检查文件: {filename} ...")

# 只读取第一行，看看有多少列
try:
    df_temp = pd.read_csv(filename, nrows=0)  # nrows=0 表示只读表头，不读数据，速度极快
    num_cols = len(df_temp.columns)
    print(f"✅ 这个文件一共有 {num_cols} 列。")
    print("最后几列的索引是:", list(range(num_cols)))
except Exception as e:
    print(f"❌ 读取出错: {e}")
