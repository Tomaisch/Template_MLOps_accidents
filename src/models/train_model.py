import sklearn
import pandas as pd 
from sklearn import ensemble
from sklearn.preprocessing import OrdinalEncoder
import joblib
import numpy as np

print(joblib.__version__)

X_train = pd.read_csv('data/preprocessed/X_train.csv', low_memory=False)
X_test = pd.read_csv('data/preprocessed/X_test.csv', low_memory=False)
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# 1) Kaputt kodierte Tausendertrennzeichen + Komma-Dezimalzahlen
def clean_numeric(series):
    return (series.astype(str)
                   .str.replace('\xa0', '', regex=False)
                   .str.replace('Â', '', regex=False)
                   .str.replace(' ', '', regex=False)
                   .str.replace(',', '.', regex=False)
                   .replace('nan', np.nan)
                   .astype(float))

for col in ['lartpc', 'larrout']:
    X_train[col] = clean_numeric(X_train[col])
    X_test[col] = clean_numeric(X_test[col])

# 2) Zahlen, die nur als String vorliegen
for col in ['pr', 'pr1']:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# 3) Freitext-/ID-Spalten ohne Vorhersagewert droppen
cols_to_drop = ['id_vehicule', 'voie', 'adr', 'num_veh']
X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

# 4) Echte Kategorien kodieren (auf Train fitten, auf Test nur transformieren)
categorical_cols = ['actp', 'v2']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)
X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

rf_classifier = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=200, criterion='entropy')

#--Train the model
rf_classifier.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/trained_model.joblib'
joblib.dump(rf_classifier, model_filename)
print("Model trained and saved successfully.")