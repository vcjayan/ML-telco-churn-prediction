import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("C:/Python/churn (1).csv")
df1 = df.copy(deep = True)
print(df.shape)

print(df1.dtypes)

df1['TotalCharges'] = pd.to_numeric(df1['TotalCharges'], errors = 'coerce')
print(df1.dtypes)


df1['TotalCharges'].fillna(df1['TotalCharges'].mean(), inplace = True)


df1.drop(['customerID'], axis = 1, inplace = True)
print(df1.isnull().sum())

#labelencoding
df1_num = df1.select_dtypes(include = [np.number])
df1_cat = df1.select_dtypes(exclude = [np.number])

encoder = LabelEncoder()

mapping = {}
for i in df1_cat:
    df1[i] = encoder.fit_transform(df1[i])
    le_name_mapping = dict(zip(encoder.classes_,encoder.transform(encoder.classes_)))
    mapping[i] = le_name_mapping
print(mapping)

X = df1.drop(['Churn'], axis =1)
y = df1['Churn']
X.shape

#as the data is imbalanced, we will resample using SMOTENN
sm = SMOTEENN()
X_resample, y_resample = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size=0.2, random_state=100)

#build SVM model
model = SVC(kernel = 'linear')
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))

print('accuracy :', accuracy_score(y_test, pred))
print('precision :', precision_score(y_test, pred))
print('recall :', recall_score(y_test, pred))
print('f1 score :', f1_score(y_test, pred))

#save the model
pickle.dump(model, open('model.pkl', 'wb'))