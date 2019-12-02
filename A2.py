import json
import re

import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD


def data_preprocess():
    with open('data/renttherunway_final_data.json') as f:
        content = f.readlines()

    df = pd.DataFrame(columns=['fit', 'user_id', 'item_id', 'bust size', 'cup size', 'weight', 'rating', 'rented for',
                               'body type', 'category', 'height', 'size', 'age'])
    count = 1
    for line in content:
        # print(line)
        obj = json.loads(line)
        if count % 10000 == 0:
            print('data/fit' + str(int(count / 10000)) + '.csv')
            df.to_csv('data/fit' + str(count / 10000) + '.csv', index=False)
            df = pd.DataFrame(
                columns=['fit', 'user_id', 'item_id', 'bust size', 'cup size', 'weight', 'rating', 'rented for',
                         'body type', 'category', 'height', 'size', 'age'])
        if count % 1000 == 0:
            print(count)
        count += 1
        # print('fit' in obj)
        attrs = ['fit', 'user_id', 'item_id', 'bust size', 'weight', 'rating', 'rented for', 'body type', 'category',
                 'height', 'size', 'age']
        contains_all_attrs = True
        for attr in attrs:
            if attr not in obj:
                contains_all_attrs = False
                break
        if not contains_all_attrs:
            continue
        bust_size = obj['bust size'][0:2]
        cup_size = ord(obj['bust size'][2:3]) - ord('a') + 1
        weight = obj['weight'][0:len(obj['weight']) - 3]
        height_str = obj['height']
        height_list = re.findall('\d+', height_str)
        height = float(height_list[0]) * 30.48 + float(height_list[1]) * 2.54

    #     df = df.append({'fit': obj['fit'], 'user_id': obj['user_id'], 'item_id': obj['item_id'],
    #                     'bust size': bust_size, 'cup size': cup_size, 'weight': weight, 'rating': obj['rating'],
    #                     'rented for': obj['rented for'], 'body type': obj['body type'], 'category': obj['category'],
    #                     'height': height, 'size': obj['size'], 'age': obj['age']}, ignore_index=True)
    # df.to_csv('data/fit20.0.csv', index=False)


def data_merge():
    df = pd.DataFrame(columns=['fit', 'user_id', 'item_id', 'bust size', 'cup size', 'weight', 'rating', 'rented for',
                               'body type', 'category', 'height', 'size', 'age'])
    for i in range(1, 21):
        print(i)
        df = df.append(pd.read_csv('data/fit' + str(i) + '.0.csv'), ignore_index=True)
    df.to_csv('data/fit.csv', index=False)


def to_user_item_pairs():
    df = pd.read_csv('data/fit1.csv')
    train_df = df[0: df.shape[0] // 10 * 9]
    train = train_df.drop(['fit', 'bust size', 'cup size', 'weight', 'rented for', 'body type', 'category', 'height', 'size', 'age'], axis=1)
    train.to_csv('data/train_user_item_pairs.csv', index=False)
    test_df = df[df.shape[0] // 10 * 9:]
    test = test_df.drop(['fit', 'bust size', 'cup size', 'weight', 'rented for', 'body type', 'category', 'height', 'size', 'age'], axis=1)
    test.to_csv('data/test_user_item_pairs.csv', index=False)


def rating_predict():
    df = pd.read_csv('data/fit1.csv')
    train = df[0: df.shape[0] // 10 * 9]
    test = df[df.shape[0] // 10 * 9:]

    # SVD
    col_names = ['user_id', 'item_id', 'rating']
    reader = Reader(rating_scale=(2, 10))
    data = Dataset.load_from_df(train[col_names], reader)
    data = data.build_full_train_pairsset()
    algo = SVD()
    algo.fit(data)
    svds = []
    for user_id, book_id in zip(train['user_id'], train['item_id']):
        svds.append(algo.predict(user_id, book_id).est)
    train['SVD'] = svds
    # reg = LinearRegression().fit(X, y)


def remove_nan():
    df = pd.read_csv('data/fit.csv')
    df = df.dropna(axis=0)
    df.to_csv('data/fit1.csv', index=False)


# to_user_item_pairs()
# data_preprocess()
# data_merge()
# remove_nan()
rating_predict()
