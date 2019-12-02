from collections import defaultdict
from random import choice

import pandas as pd


def convert():
    data = pd.read_csv('data/pairs_Read.txt')
    test = pd.DataFrame(columns=['userID', 'bookID'])
    for pair in data['userID-bookID']:
        user = pair.split('-')[0]
        book = pair.split('-')[1]
        test = test.append({'userID': user, 'bookID': book}, ignore_index=True)
    test.to_csv('data/test_read.csv', index=False)


def build_validation_set():
    data = pd.read_csv('data/train_Interactions.csv.gz', compression='gzip')
    train = data.iloc[:190000, [0, 1]]
    train['label'] = 1
    validation = data.iloc[190000:, [0, 1]]
    validation['label'] = 1
    users = defaultdict(set)
    books = set()
    for user, book in zip(data['userID'], data['bookID']):
        users[user].add(book)
        books.add(book)
    for user in validation['userID']:
        rand_book = choice(list(books))
        while rand_book in users[user]:
            rand_book = choice(list(books))
        validation = validation.append({'userID': user, 'bookID': rand_book, 'label': 0}, ignore_index=True)
    train.to_csv('data/train_read.csv', index=False)
    validation.to_csv('data/validation_read.csv', index=False)


def predict(train_path, validation_path, score_threshold, popularity, write):
    train = pd.read_csv(train_path)
    user2books = defaultdict(set)
    book2users = defaultdict(set)
    book_count = defaultdict(int)
    total_read = 0
    for user, book in zip(train['userID'], train['bookID']):
        user2books[user].add(book)
        book2users[book].add(user)
        book_count[book] += 1
        total_read += 1
    validation = pd.read_csv(validation_path)
    pairs = pd.DataFrame(columns=['userID-bookID', 'prediction'])
    for user, pred_book in zip(validation['userID'], validation['bookID']):
        have_read_books = user2books[user]
        score = 0
        for book in have_read_books:
            score += jaccard(book2users[book], book2users[pred_book])
        score /= len(have_read_books)
        res = 0
        if score > score_threshold or book_count[pred_book] > total_read / popularity:
            res = 1
        pairs = pairs.append({'userID-bookID': user + '-' + pred_book, 'prediction': res}, ignore_index=True)
    if write:
        pairs.to_csv('data/prediction_read.csv', index=False)
        return 0
    else:
        return MSE(pairs['prediction'], validation['label'])


def jaccard(user_set1, user_set2):
    return len(user_set1 & user_set2) / len(user_set1 | user_set2)


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


def search():
    min_mse = 99999999
    for score_threshold in range(18, 25):
        score_threshold = score_threshold * 0.0001
        for popularity in range(1, 2):  # 1.4 - 2
            mse = predict('data/train_read.csv', 'data/validation_read.csv', score_threshold, popularity, False)
            if mse < min_mse:
                print('score_threshold: ' + str(score_threshold) +
                      ' popularity: ' + str(popularity) +
                      ' MSE: ' + str(mse))
                min_mse = mse


# build_validation_set()
# predict(0.03, 2)
search()
# convert()
# predict('data/.csv', )
# data = pd.read_csv('data/train_Interactions.csv.gz', compression='gzip')
# data.to_csv('data/data_read.csv', index=False)
# predict('data/train_read.csv', 'data/test_read.csv', 0.002, 1, True)
