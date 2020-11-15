import shutil
from pathlib import Path
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def concat_dataframe():
    def unwrap_columns(df, key):
        parsed = df[key].map(str).apply(json.loads).apply(pd.Series)
        df[list(map(lambda c: '{}_{}'.format(key, c), [0, 1, 2, 3]))] = parsed
        df['{}_area'.format(key)] = (df['{}_3'.format(key)] - df['{}_1'.format(key)]) * (df['{}_2'.format(key)] - df['{}_0'.format(key)])
        df['{}_0'.format(key)] = df['{}_0'.format(key)] / df['width']
        df['{}_1'.format(key)] = df['{}_1'.format(key)] / df['height']
        df['{}_2'.format(key)] = df['{}_2'.format(key)] / df['width']
        df['{}_3'.format(key)] = df['{}_3'.format(key)] / df['height']
        return df

    dfs = []
    for df in os.listdir('./train_data/redaction_labels_nn/'):
        df = pd.read_csv('./train_data/redaction_labels_nn/' + df, index_col=0)
        dfs.append(df)
    full_df = pd.concat(dfs)
    full_df.drop_duplicates(subset=['file'], keep='last', inplace=True)
    redaction = full_df['redaction'].map(str).apply(json.loads).tolist()
    redaction = list(map(lambda r: list(map(lambda a: (a[3] - a[1]) * (a[2] - a[0]), r)), redaction))
    redaction = list(map(sum, redaction))
    full_df['redaction_area'] = redaction
    full_df = unwrap_columns(full_df, 'title')
    full_df = unwrap_columns(full_df, 'abstract')
    full_df['redaction_area'].hist(bins=1000)
    print(len(full_df))
    plt.show()
    full_df.to_csv('./train_data/full_df.csv')
    print(full_df.head(10))


def filter_dataframe():
    df = pd.read_csv('./train_data/full_df.csv', index_col=0)
    print(len(df))
    df.drop(df[df['abstract_area'] < 4000].index, inplace=True)
    df.drop(df[df['redaction_area'] < 40000].index, inplace=True)
    df.drop(df[df['redaction_area'] > 350000].index, inplace=True)
    # plt.figure()
    # df['abstract_area'].hist(bins=100)
    # plt.show()
    plt.figure()
    df['redaction_area'].hist(bins=1000)
    plt.show()
    plt.figure()
    plt.scatter(df['redaction_area'], df['title_area'])
    plt.axes().set_aspect('equal', adjustable='box')
    plt.show()
    plt.figure()
    plt.scatter(df['redaction_area'], df['abstract_area'])
    plt.axes().set_aspect('equal', adjustable='box')
    plt.show()
    print(len(df))
    return df


concat_dataframe()
filter_dataframe()

train_dir = Path("./train_data/to_train")
train_dir.mkdir(parents=True, exist_ok=True)
shutil.rmtree('./train_data/to_train')
train_dir = Path("./train_data/to_train")
train_dir.mkdir(parents=True, exist_ok=True)
area_dir = Path('./train_data/area')
area_dir.mkdir(parents=True, exist_ok=True)


def save_as_image(key, abstract_0, abstract_1, abstract_2, abstract_3):
    img = np.round(np.array([abstract_0, abstract_1, abstract_2, abstract_3]) * 255, decimals=0)
    img = Image.fromarray(img).convert('L')
    img.save('{}/{}'.format(area_dir, key))


df = pd.read_csv('./train_data/filtered.csv', index_col=0)
df.apply(lambda r: save_as_image(r['file'], r['abstract_0'], r['abstract_1'], r['abstract_2'], r['abstract_3']), axis=1)

for idx, file in enumerate(df['file'].tolist()):
    print(idx, file)
    try:
        shutil.copy('./train_data/redaction_nn/' + file, train_dir)
    except FileNotFoundError:
        pass
