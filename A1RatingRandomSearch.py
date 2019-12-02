import pandas as pd
from surprise import Reader, Dataset, SVD, SVDpp
from surprise.model_selection import GridSearchCV, RandomizedSearchCV


def search():
    df = pd.read_csv('data/train_Interactions.csv.gz', compression='gzip')
    col_names = ['userID', 'bookID', 'rating']
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['userID', 'bookID', 'rating']], reader)
    param_grid = {'n_factors': [15,16,17,18,19,20],
                  'n_epochs': [20,25,30,35,40],
                  'lr_all': [0.002,0.003,0.004,0.005,0.006,0.007],
                  'reg_all': [0.3,0.27,0.25,0.22,0.2,0.15,0.1,0.05]}
    grid_search = RandomizedSearchCV(SVD, param_grid, n_iter=40, measures=['RMSE'], cv=4)
    grid_search.fit(data)
    print(grid_search.best_score)
    print(grid_search.best_params)


search()

# pairs = pd.read_csv('data/pairs_Rating.txt')
# test = pd.DataFrame(columns=['userID', 'bookID'])
# for index, row in pairs.iterrows():
#     userID = row['userID-bookID'].split('-')[0]
#     bookID = row['userID-bookID'].split('-')[1]
#     test = test.append({'userID': userID, 'bookID': bookID}, ignore_index=True)
# test.to_csv('data/test_rating.csv', index=False)
