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

def metric_scoring(classifier, x_test_data, y_test_data, model_name):
  y_true = y_test_data
  y_pred = classifier.predict(x_test_data)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  accuracy = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)

  metric_data = {
    'Model Name': model_name,
    'Precision' : round(precision,4),
    'Recall' : round(recall,4),
    'Accuracy': round(accuracy,4),
    'F1 Score': round(f1,4)
  }
  return metric_data

#loading in data for training
X_train = pd.read_csv('/users/n/UFC-Predictions/Project_Data/X_processed_train.csv', index_col=0)
y_train = pd.read_csv('/users/n/UFC-Predictions/Project_Data/y_processed_train.csv', index_col=0)

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

#we need to keep the X_train non cat data different from the catboost data since catboost deals with categorical variables implicitly
#LR, RF and SVC do not so we scale and process
X_train_non_cat = scaled_train_df.join(encoded_cat_data_train)

#clasifiers
logit =  LogisticRegression()
RF = RandomForestClassifier()
SVC = SVC()
Cat = CatBoostClassifier(cat_features = categorical_features_indices)

classifiers_non_cat = [logit, RF, SVC]

#fitting Logistic, RF and SVC
for model in classifiers_non_cat:
    model.fit(X_train_non_cat, y_train.ravel())

#fitting Catboost
Cat.fit(X_train, y_train)


#prepping our test data
#in order to help prevent data leakage we load our test data after the models are trained
X_test = pd.read_csv('/users/n/UFC-Predictions/Project_Data/X_processed_test.csv', index_col=0)
y_test = pd.read_csv('/users/n/UFC-Predictions/Project_Data/y_processed_test.csv', index_col=0)

#to enable proper modeling, we binarize our winner column 1 indicates red corner wins, 0 indicates blue corner wins
y_test = preprocessing.LabelBinarizer().fit_transform(y_test['Winner'])

#binarizing our title bout field in the training data, 1 indicates title, 0 non title fight
X_test['title_bout'] = preprocessing.LabelBinarizer().fit_transform(X_test['title_bout'])

#processing X_test data
X_test_numeric = X_test.select_dtypes(include = numerics)
X_test_cat = X_test.select_dtypes(include = 'object')

scaler = StandardScaler()

#scaling our test data
scaled_test = scaler.fit_transform(X_test_numeric.values)
scaled_test_df = pd.DataFrame(scaled_test, index=X_test_numeric.index, columns=X_test_numeric.columns)

#use the previous encoder fitted on the train data this will prevent target leakage
encoded_cat_data = encoder.transform(X_test_cat)

#joining encoded and scaled data together
X_test_numerical = scaled_test_df.join(encoded_cat_data)

#metric scores for baseline models
lr_base_metric = metric_scoring(classifiers_non_cat[0], X_test_numerical, y_test, 'Logistic Regression')

lr_base_metric

rf_base_metric = metric_scoring(classifiers_non_cat[1], X_test_numerical, y_test, 'Random Forest Classifier')

svc_base_metric = metric_scoring(classifiers_non_cat[2], X_test_numerical, y_test, 'Support Vector Classifier')

cat_base_metric = metric_scoring(Cat, X_test, y_test, 'CatBoost Classifier')

metrics = [lr_base_metric,rf_base_metric,svc_base_metric,cat_base_metric]

lr_df = pd.DataFrame.from_dict(metrics[0], orient = 'index').T
rf_df = pd.DataFrame.from_dict(metrics[1], orient = 'index').T
svc_df = pd.DataFrame.from_dict(metrics[2], orient = 'index').T
cat_df = pd.DataFrame.from_dict(metrics[3], orient = 'index').T

baseline_model_results = pd.concat([lr_df, rf_df, svc_df, cat_df]).reset_index().drop(columns = 'index')

baseline_model_results

#Checking our baseline models performance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

#red corner is 1
#blue corner is 0
ConfusionMatrixDisplay.from_estimator(classifiers_non_cat[0], X_test_numerical, y_test)

ConfusionMatrixDisplay.from_estimator(classifiers_non_cat[1], X_test_numerical, y_test)

ConfusionMatrixDisplay.from_estimator(classifiers_non_cat[2], X_test_numerical, y_test)

ConfusionMatrixDisplay.from_estimator(Cat, X_test, y_test)
