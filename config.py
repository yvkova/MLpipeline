# Configuration file

# import libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit

# configurations
TEST_SIZE = 0.33
RANDOM_STATE = 10
N_SPLITS = 10
CV = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Must contain at least two models/params

MODELS = [

    [LinearRegression(), {'fit_intercept': [True, False],
                          'normalize': [True, False],
                          'copy_X': [True, False]}],

    [RandomForestRegressor(), {'bootstrap': [True, False],
                               'min_samples_split': [10, 20, 40],
                               'min_samples_leaf': [1, 2, 4],
                               # 'max_features': ['auto', 'sqrt'],
                               'max_depth': [20, 40],
                               'n_estimators': [100, 200]}],

    [DecisionTreeRegressor(), {'min_samples_split': [10, 20, 40],
                               'min_samples_leaf': [20, 40, 100],
                               'max_depth': [2, 6, 8],
                               'max_features': ['auto', 'sqrt'],
                               'max_leaf_nodes': [5, 20, 100]}]

        ]
