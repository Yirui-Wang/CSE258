import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from surprise import Reader, Dataset, SVD

df = pd.read_csv('data/fit.csv')
train = df[0: df.shape[0] // 10 * 8]
validation = df[df.shape[0] // 10 * 8: df.shape[0] // 10 * 9]
test = df[df.shape[0] // 10 * 9:]

# SVD
col_names = ['user_id', 'item_id', 'rating']
reader = Reader(rating_scale=(2, 10))
data = Dataset.load_from_df(train[col_names], reader)
data = data.build_full_trainset()
algo = SVD()
algo.fit(data)
svds = []
for user_id, book_id in zip(train['user_id'], train['item_id']):
    svds.append(algo.predict(user_id, book_id).est)
train = train.assign(SVD=svds)
print(train.shape)

one_hot_rf = pd.get_dummies(train['rented for'])
one_hot_bt = pd.get_dummies(train['body type'])
one_hot_cat = pd.get_dummies(train['category'])

df = df.drop('rented for', axis=1)
df = df.drop('body type', axis=1)
df = df.drop('category', axis=1)

df = df.drop('review date', axis=1)
df = df.drop('fit', axis=1)

ratings = df['rating']
df = df.drop('rating', axis=1)

df = df.join(one_hot_rf)
df = df.join(one_hot_bt)
df = df.join(one_hot_cat)

train = df[0: df.shape[0] // 10 * 8]
validation = df[df.shape[0] // 10 * 8: df.shape[0] // 10 * 9]

reg = LinearRegression().fit(train, ratings)

reg.predict()

labels = test['rating']

reg.predict()
