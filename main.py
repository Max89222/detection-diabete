import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import RFECV
from sklearn.linear_model import SGDClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
import joblib
import seaborn as sns
from sklearn.metrics import precision_score, recall_score

data = pd.read_csv('exam_final_diabete.csv')

data = data.sample(frac=0.1, random_state=42) # voir learning curve

data = data.dropna(axis=0)


X = data.drop(['diabetes', 'year', 'clinical_notes', 'race:AfricanAmerican',
       'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other', 'location', 'gender', 'smoking_history', 'heart_disease',
       'bmi', 'hypertension'], axis=1)

y = data['diabetes']

X['new_column'] = X['hbA1c_level']*X['blood_glucose_level']
X['new_column_2'] = X['hbA1c_level'] * X['age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TEST DE DEPENDANCE : SELECT FROM MODEL : 
'''
X_train_test_dependance_categorical = pd.DataFrame(OrdinalEncoder().fit_transform(X_copie), columns=['gender', 'location', 'smoking_history'])
X_train_test_dependance_numerical = pd.DataFrame(MinMaxScaler().fit_transform(X_train[['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']]), columns=['age', 'bmi', 'hbA1c_level', 'blood_glucose_level'])

X_train_test_dependance = pd.concat([X_train_test_dependance_categorical, X_train_test_dependance_numerical], axis=1)
selector = SelectFromModel(SGDClassifier(random_state=42), threshold='mean')
X_train_test_dependance = selector.fit_transform(X_train_test_dependance, y_train)
print(selector.estimator_.coef_)
print(X_train_test_dependance_categorical.columns)
print(X_train_test_dependance_numerical.columns)
print(selector.get_support())'''


# CONCLUSIONS : variables les plus utiles : 'age' , 'hbA1c_level', 'blood_glucose_level'
                # variables inutiles (coefficient très faible) : 'gender', 'location', ' smoking_history', 'bmi'
                # Donc toutes les variables catégorielles sont presque inutiles

model = make_pipeline(PolynomialFeatures(), StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1)) 

param_grid = {
  'polynomialfeatures__degree' : [1, 2, 3],
  'randomforestclassifier__n_estimators': [100, 200, 500],
  'randomforestclassifier__max_depth': [5, 10, None]
}

grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

# entraînement modèle

print("début phase d'entraînement")
grid.fit(X_train, y_train)
print("fin phase d'entraînement")

model = grid.best_estimator_

# affichage meilleurs paramètres :

print(grid.best_params_)

# affichage scores : 

print('scores obtenus : ')
print('coefficient de détermination :', model.score(X_test, y_test))
print('précision score :', precision_score(y_test, model.predict(X_test)))
print('recall score :', recall_score(y_test, model.predict(X_test)))

# création d'un fichier "model_file_joblib" dans lequel on enregistre model : 

joblib.dump(model, 'model_file_joblib') 

# affichage learning curve : 

N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.01, 1, 10), cv=5)
plt.plot(N, train_score.mean(axis=1), label="évolution du train_score ")
plt.plot(N, val_score.mean(axis=1), label='évolution du val_score')
plt.xlabel('taille du dataset (en exemples)')
plt.ylabel('scores obtenus sur le train_set et le val_set')
plt.legend()

plt.show()

