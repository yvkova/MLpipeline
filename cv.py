# This script implements the cross-validation process

# import libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from config import CV
import matplotlib.pyplot as plt
import seaborn as sns


# apply cross validation and plot score to boxplot diagram
def get_cross_val_scores(models, X, y):
    fig2, axs2 = plt.subplots(ncols=1)
    results = {}
    for model in models:
        results[str(model[0]).replace("()", "")] = cross_val_score(model[0], X, y, cv=CV, scoring='r2')

    sns.boxplot(x="Model", y="R2 Score", data=pd.melt(pd.DataFrame(results),
                                                      var_name='Model', value_name='R2 Score'), ax=axs2)
    axs2.set_title('Cross Validation Scores')

    return results
