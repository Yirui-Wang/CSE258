import pandas as pd
from surprise import Reader, Dataset, SVD, SVDpp, NMF, SlopeOne

df = pd.read_csv('data/train_Interactions.csv.gz', compression='gzip')
col_names = ['userID', 'bookID', 'rating']
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['userID', 'bookID', 'rating']], reader)
data = data.build_full_trainset()
algo = SVD(n_factors=16, n_epochs=40, lr_all=0.007, reg_all=0.25)
# algo = SlopeOne()
algo.fit(data)
test = pd.read_csv('data/test_rating.csv')
predictions = pd.DataFrame(columns=['userID-bookID', 'prediction'])
for userID, bookID in zip(test['userID'], test['bookID']):
    predictions = predictions.append({'userID-bookID': userID + '-' + bookID,
                                      'prediction': algo.predict(userID, bookID).est},
                                     ignore_index=True)
predictions.to_csv('data/predictions_rating9.csv', index=False)
# pairs = pd.DataFrame(columns=col_names)
# for row in df:
#     userID = pair.split('-')[0]
#     bookID = pair.split('-')[1]
#     pairs.append({'userID': userID, 'bookID': bookID, 'rating': rating})
# pairs.to_csv('data/train_rating.csv')
