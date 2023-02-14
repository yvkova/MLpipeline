# Tuning model hyper-parameters

# Import libraries
from sklearn.model_selection import GridSearchCV


# apply grid search cv to each model passed in the config file
def apply_grid_search_cv(models, cv, X, y):
    results = {}
    for model in models:
        clf = GridSearchCV(model[0], cv=cv, param_grid=model[1], scoring='r2')
        clf.fit(X, y)
        results[str(model[0]).replace("()", "")] = clf.best_score_
    return results
