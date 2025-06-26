# Création d'un modèle de prédiction du diabète à partir de données cliniques

**Auteur : Max89222**

---

Ce projet vise à créer un modèle capable de prédire le diabète à partir de différentes caractéristiques. 

## 1) Structure et utilité des fichiers

- `exam_final_diabete.csv` : fichier dans lequel chaque ligne représente un individu et chaque colonne une caractéristique (ex : hbA1c_level)
- `main.py` : fichier python contenant le code source de notre modèle
- `model_file_joblib.pkl`: fichier permettant d'enregistrer le modèle déjà entraîné dans le but d'effectuer des prédictions sans avoir à l'entraîner à nouveau

## 2) Dataset

taille du dataset : (100000, 17)

 📊 Description des variables du dataset sur le diabète

| Variable                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `year`                 | Année de la consultation ou de l'enregistrement.                            |
| `gender`               | Sexe du patient (ex. : Male, Female, Other).                                |
| `age`                  | Âge du patient.                                                             |
| `location`             | Lieu géographique (ville, région ou établissement médical).                 |
| `race:AfricanAmerican` | Variable binaire : le patient est Afro-Américain (1) ou non (0).            |
| `race:Asian`           | Variable binaire : le patient est Asiatique (1) ou non (0).                 |
| `race:Caucasian`       | Variable binaire : le patient est Caucasien (1) ou non (0).                 |
| `race:Hispanic`        | Variable binaire : le patient est Hispanique (1) ou non (0).                |
| `race:Other`           | Variable binaire : le patient appartient à une autre origine (1) ou non (0).|
| `hypertension`         | Présence d'hypertension : oui (1), non (0).                                |
| `heart_disease`        | Présence de maladie cardiaque : oui (1), non (0).                          |
| `smoking_history`      | Historique tabagique (ex. : jamais, ancien fumeur, fumeur actuel, etc.).    |
| `bmi`                  | Indice de masse corporelle (Body Mass Index).                              |
| `hbA1c_level`          | Taux d'hémoglobine glyquée (HbA1c), indicateur de glycémie chronique.       |
| `blood_glucose_level`  | Taux de glucose dans le sang mesuré lors d’un test.                        |
| `diabetes`             | Diagnostic de diabète : oui (1), non (0).                                   |
| `clinical_notes`       | Commentaires médicaux ou observations cliniques textuelles.                |


## 3) Technologies utilisées

- `pandas` : manipulation et nettoyage des données  
- `scikit-learn` : création et évaluation des modèles de machine learning  
- `matplotlib` : visualisation des données  
- `numpy` : opérations numériques  
- `joblib` : sauvegarde et chargement des modèles

Le modèle final utilisé est un **RandomForestClassifier**, qui s’est avéré le plus performant dans ce problème de classification binaire.

## 4) Résultats et métriques

voici les scores obtenus sur le test set : 

- coefficient de détermination : 0.971
- précision score : 1.0
- recall score : 0.6441717791411042

## 5) Installation

1. Installer Git (si ce n’est pas déjà fait) :
   
`brew install git`

2. Cloner le dépôt :

`git clone <clé_ssh>`
`cd <nom_du_dossier>`

3. Installer les dépendances :

`pip3 install pandas scikit-learn matplotlib numpy joblib`

4. Entraîner le modèle :

!!! Pour toute les action qui vont suivre, pensez bien à mettre tous les fichiers dans le même dossier sans quoi vous rencontrerez une erreur au lancement du fichier `main.py` !!!

-Pour entraîner le modèle : 
Ouvrir `main.py` (ainsi que les autres fichiers) dans un éditeur de code et l'exécuter (l'entraînement devrait être plutôt rapide en raison du fractionnage du dataset).

-Pour effectuer des prédictions : 
Si vous souhaitez effectuer des prédictions, créez un nouveau fichier python, copiez-coller le code suivant dans un éditeur de code et exécutez le en pensant bien à également passer les autres fichiers ! : 

```
import joblib
import numpy as np

model = joblib.load('model_file_joblib')

def diabete(age, hbA1c_level, blood_glucose_level):
    
    new_col_1 = int(hbA1c_level) * int(blood_glucose_level)
    new_col_2 = int(hbA1c_level) * int(age)
    X_input = np.array([[int(age), int(hbA1c_level), int(blood_glucose_level), int(new_col_1), int(new_col_2)]])
    pred = model.predict(X_input)
    return pred

age = input('votre âge : ')
hbA1c_level = input("Votre taux d'hémoglobine glyquée (HbA1c) : ")
blood_glucose_level = input("Votre taux de glucose dans le sang : ")

print('-------------------------------------------')
print('RESULTAT : ')
if diabete(age, hbA1c_level, blood_glucose_level) == 1:
    print('vous êtes diabétique ! ')
else:
    print("vous n'êtes pas diabétique ! ")
print('------------------------------------------------')

```


## 6) Idées d'amélioration et contributions
De nombreux points restent à améliorer notamment au niveau du recall score comme vous avez pu le voir. N'hésitez pas à contribuer à ce projet ou à proposer des idées d'amélioration !


