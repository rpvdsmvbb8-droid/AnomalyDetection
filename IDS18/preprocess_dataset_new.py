import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from collections import Counter

def reduce_mem_usage(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem),flush=True)
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem),flush=True)
    return df


def main():
    st = time.time()
    PID = str(os.getppid())

    data_files = []
    for dirname, _, filenames in os.walk('data'):
         for filename in filenames:
             data_files.append(os.path.join(dirname, filename))

    df = reduce_mem_usage(pd.read_csv(data_files[3],low_memory=False,index_col=False,usecols=[i for i in range(4,74)]))
    print(df.dtypes)
    dtypes = df.dtypes.to_dict()
    columns = df.columns.tolist()
    print("======Starting Concat========",flush=True)

    for filename in data_files:
        if filename =="data/02-20-2018.csv":
            continue
        elif filename=="data/02-16-2018.csv" or filename=="data/02-28-2018.csv" or filename=="data/03-01-2018.csv":
            print(f"======={filename}======",flush=True)
            print(f"file size: {os.path.getsize(filename) / (1024**2)} MB",flush=True)
            df = pd.concat([df, reduce_mem_usage(pd.read_csv(filename, low_memory=False, index_col=False,names=columns[:70]))],ignore_index=True)
            print(f"Concatenated dataFrame mem usage: {df.memory_usage().sum()/1024**2} MB",flush=True)
            print("free -mh output is : ",flush=True)
            os.system("free -mh")
        else:
            print(f"======={filename}======",flush=True)
            print(f"file size: {os.path.getsize(filename) / (1024**2)} MB",flush=True)
            df = pd.concat([df, reduce_mem_usage(pd.read_csv(
                filename, low_memory=False, index_col=False,dtype={k: v for k, v in dtypes.items() if k in df.columns}))],ignore_index=True)
            print(f"Concatenated dataFrame mem usage: {df.memory_usage().sum()/1024**2} MB",flush=True)
            print("free -mh output is : ",flush=True)
            os.system("free -mh")
    print(time.time()-st) 

    df.drop(['Bwd PSH Flags'], axis=1, inplace=True)
    df.drop(['Bwd URG Flags'], axis=1, inplace=True)
    df.drop(['Fwd Byts/b Avg'], axis=1, inplace=True)
    df.drop(['Fwd Pkts/b Avg'], axis=1, inplace=True)
    df.drop(['Fwd Blk Rate Avg'], axis=1, inplace=True)
    df.drop(['Bwd Byts/b Avg'], axis=1, inplace=True)
    df.drop(['Bwd Pkts/b Avg'], axis=1, inplace=True)
    df.drop(['Bwd Blk Rate Avg'], axis=1, inplace=True)
    df.drop(['Timestamp'], axis=1, inplace=True)
    df["Flow Byts/s"].replace([np.inf, -np.inf, -np.nan, np.nan], 0, inplace=True)
    df["Flow Pkts/s"].replace([np.inf, -np.inf, -np.nan, np.nan], 0, inplace=True)

    # Encodes target label
    encoder = preprocessing.LabelEncoder()
    for c in df.columns:
        if str(df[c].dtype) == 'object': 
            df[c] = encoder.fit_transform(df[c])

    df["Label"] = (df["Label"].apply(lambda x: np.random.randint(15, 20) if x == 0 else x))
    unique_label=df['Label'].unique()
    print("labels are:",unique_label)
    print("Before removing duplicates",Counter(df['Label']))
    print("Total no of rows before removing duplicates",df.shape[0])
    df.drop_duplicates(inplace=True)
    print("After removing duplicates",Counter(df['Label']))
    print("Total no of rows after removing duplicates",df.shape[0])
    y = df.pop(df.columns[-1]).to_frame()
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min() + 1e-5)

    multilabel_idx=~df.duplicated(keep='first')
   # print("Number of multilabel rows",Counter(multilabel_idx))
   
    df=df[multilabel_idx]
    y=y[multilabel_idx]
    print("Total number of rows after removing multilabel rows",df.shape[0])
    print("After removing multi_label duplicates",Counter(y['Label']))
    df=pd.concat([df,y],axis=1)    
#     for label in unique_label:
#         sub_dataset=df[df['Label']==label]
#         np.save("data/"+str(label)+".npy",sub_dataset)

    
    #df.to_csv("./data/ids18.csv")
    with open('./data/ids_18.pth','wb') as f:
        pickle.dump(df,f) 
    
if __name__ == "__main__":
    main()
