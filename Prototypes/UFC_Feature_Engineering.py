import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv('/users/n/UFC-Predictions/data/data.csv')


X = df.drop(columns = ['Winner', 'date'])
y = df['Winner']

x_ohe = pd.get_dummies(X)
#dealing with null values, for now we will use a naive mean imputation method, additionally we will standardize before imputing


#two issues so far
#what do we do with:
    #1. categorical variables
    #2. null values

#a few immediate solutions:
    # Naive method: drop na
    # Implement Catboost

#another
    # Impute NA with median vals
    # Implement Catboost

#first iteration of our model we will make two so far, baseline naive model
from catboost import CatBoostClassifier
from sklearn import preprocessing

df_naive = df.dropna()
df_naive.drop(df_naive[df_naive['Winner'] == 'Draw'].index, inplace = True)

y = preprocessing.LabelBinarizer().fit_transform(df_naive['Winner'])


#dropping fights with a draw

X_naive = df_naive.drop(columns = ['Winner', 'date'])


X_train_naive, X_test_naive, y_train_naive, y_test_naive = train_test_split(X_naive, y, test_size = 0.20)

categorical_features_indices = np.where(X.dtypes != float)[0]
model = CatBoostClassifier(cat_features = categorical_features_indices)
model.fit(X_train_naive, y_train_naive)

#basics to show performance on test set
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def metric_scoring(classifier, x_test_data, y_test_data):
  y_true = y_test_data
  y_pred = classifier.predict(x_test_data)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  accuracy = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)

  metric_data = {
      'Precision' : precision,
      'Recall' : recall,
      'Accuracy': accuracy,
      'F1 Score': f1
  }
  return metric_data

metric_scoring(model, X_test_naive, y_test_naive)

df
df_naive['Winner'].value_counts(normalize = True)

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

#red corner is 1
#blue corner is 0
ConfusionMatrixDisplay.from_estimator(model, X_test_naive, y_test_naive)
plt.title("Confusion Matrix")

#gaining an idea of feature importance
import matplotlib.pyplot as plt
feat_importances = pd.Series(model.feature_importances_, index=X_test_naive.columns)
feat_importances.nlargest(15).sort_values(ascending = True).plot(kind='barh')
plt.title("Top 15 important features")
plt.show()
