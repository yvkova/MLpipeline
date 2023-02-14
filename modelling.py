# This script implements data spliting and model fitting

# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns


# train and test each model passed in the config file and plot results
def train_and_test_model(x_train, x_test, y_train, y_test, models):
    n = 0
    results = {}
    fig1, axs1 = plt.subplots(ncols=len(models))
    for model in models:
        model[0].fit(x_train, y_train)
        y_pred = model[0].predict(x_test)
        results[str(model[0]).replace("()", "")] = r2_score(y_test, y_pred)
        get_metrics(model[0], y_test, y_pred)
        sns.regplot(x='y_test',
                    y='y_pred',
                    data=pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}),
                    ax=axs1[n])
        axs1[n].set_title(str(model[0]).replace("()", "") + "\nR2: " + str(r2_score(y_test, y_pred)))
        n += 1
    return results


# get metrics for each model
def get_metrics(model, y_test, y_pred):
    print("\nModel:", model)
    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("EVS:", explained_variance_score(y_test, y_pred))
