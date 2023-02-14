# The main script calling the other scripts

# import libraries
from preprocessing import X, y, df
from sklearn.model_selection import train_test_split
from modelling import train_and_test_model
from cv import get_cross_val_scores
from results import plot_bar_chart
from tuning import apply_grid_search_cv
from config import MODELS, TEST_SIZE, RANDOM_STATE, CV
import matplotlib.pyplot as plt
# import seaborn as sns

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# sns.pairplot(df)

# dictionary of TTS R2 scores
tts_scores = train_and_test_model(X_train, X_test, y_train, y_test, MODELS)
print("\nTTS", tts_scores)

# dictionary of CV R2 scores
cv_scores = get_cross_val_scores(MODELS, X, y)
print("\nCV", cv_scores)

# dictionary of GCV scores
gcv_scores = apply_grid_search_cv(MODELS, CV, X, y)
print("\nGCV", gcv_scores)

# plot a bar chart comparing tts, cv and gcv scores
plot_bar_chart(tts_scores, cv_scores, gcv_scores)

# display matplotlib/seaborn diagrams
plt.show()
