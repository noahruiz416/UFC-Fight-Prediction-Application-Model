#standard data loading and manipulation libraires
import pandas as pd
import numpy as np

#models, to create pipelines for
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

#pipeline and data processing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from category_encoders.quantile_encoder import QuantileEncoder

#metrics stuff
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

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

#loading in data for training
X_train = pd.read_csv('/users/n/UFC-Predictions/Project_Data/X_processed_train.csv', index_col=0)
y_train = pd.read_csv('/users/n/UFC-Predictions/Project_Data/y_processed_train.csv', index_col=0)

#loading in data for testing models
X_test = pd.read_csv('/users/n/UFC-Predictions/Project_Data/X_processed_test.csv', index_col=0)
y_test = pd.read_csv('/users/n/UFC-Predictions/Project_Data/y_processed_test.csv', index_col=0)

#binarizing our target varaible, 1 indicates red winner, 0 indicates blue winner
y_train = preprocessing.LabelBinarizer().fit_transform(y_train['Winner'])

#binarizing our title bout field in the training data, 1 indicates title, 0 non title fight
X_train['title_bout'] = preprocessing.LabelBinarizer().fit_transform(X_train['title_bout'])

#defining our categorical indices for catboost and numeric values for further processing
categorical_features_indices = ["R_fighter", "B_fighter", "Referee", "location", "weight_class", "B_Stance", "R_Stance"]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


#creating the pipelines for each model, for RV, SVM and Logistc we will use Quantile Encoding
#for catboost, the model inherently handles categorical features
X_train_numeric = X_train.select_dtypes(include = numerics)
X_train_cat = X_train.select_dtypes(include = 'object')

scaler = StandardScaler()
encoder = QuantileEncoder()

#scaling our numeric data
scaled_train = scaler.fit_transform(X_train_numeric.values)
scaled_train_df = pd.DataFrame(scaled_train, index=X_train_numeric.index, columns=X_train_numeric.columns)

#endoing our other data
encoded_cat_data_train = encoder.fit_transform(X_train_cat, y_train)

#joining encoded and scaled data together
X_train = scaled_train_df.join(encoded_cat_data_train)


#clasifiers
logit =  LogisticRegression()
RF = RandomForestClassifier()
SVC = SVC()
Cat = CatBoostClassifier(cat_features = categorical_features_indices)

classifiers_non_cat = [logit, RF, SVC]

#fitting Logistic, RF and SVC
for model in classifiers:
    model.fit(X_train, y_train.ravel())

#fitting Catboost
Cat.fit(X_train, y_train)


#prepping our test data
#binarizing our target varaible, 1 indicates red winner, 0 indicates blue winner
y_test = preprocessing.LabelBinarizer().fit_transform(y_test['Winner'])

#binarizing our title bout field in the training data, 1 indicates title, 0 non title fight
X_test['title_bout'] = preprocessing.LabelBinarizer().fit_transform(X_test['title_bout'])

#processing X_test data
X_test_numeric = X_test.select_dtypes(include = numerics)
X_test_cat = X_test.select_dtypes(include = 'object')

scaler = StandardScaler()
encoder = QuantileEncoder()

#scaling our numeric data
scaled_test = scaler.fit_transform(X_test_numeric.values)
scaled_test_df = pd.DataFrame(scaled_test, index=X_test_numeric.index, columns=X_test_numeric.columns)

#endoing our other data
encoded_cat_data = encoder.fit_transform(X_test_cat, y_test)

#joining encoded and scaled data together
X_test_numerical = scaled_test_df.join(encoded_cat_data)



metric_scoring(classifiers[0], X_test_numerical, y_test)

metric_scoring(classifiers[1], X_test_numerical, y_test)

metric_scoring(classifiers[2], X_test_numerical, y_test)

metric_scoring(Cat, X_test, y_test)

#Checking our baseline models performance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

#red corner is 1
#blue corner is 0
ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test)

RocCurveDisplay.from_estimator(classifier, X_test, y_test)
