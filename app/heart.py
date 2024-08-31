import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

heart=pd.read_csv('app/heart.csv')

#X->model data y->output of that particular row
X=heart.iloc[:,:13]
y=heart.iloc[:,-1]

# applied standard scaler to the data
scaler=StandardScaler()
X_scale=scaler.fit_transform(X)

#splittting the train and testing data
X_train,X_test,y_train,y_test = train_test_split(X_scale,
                                                y,
                                                test_size=0.1,
                                                random_state=42)


#random forest classifier
rf=RandomForestClassifier(criterion='entropy',max_depth=1)
data = rf.fit(X_train,y_train)

pickle.dump(data, open('app/model.pkl', 'wb'))
 
# Creating a function for the user

class HeartDiseaseDiagnosis:
    def __init__(self):
        # Load data and initialize variables
        self.heart = pd.read_csv('app/heart.csv')
        self.scaler = StandardScaler()
        self.rf = RandomForestClassifier()
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()
        self.y_pred = None

        # Train the model
        self.rf.fit(self.X_train, self.y_train)

    def preprocess_data(self):
        # Example preprocessing function
        X = self.heart.drop('target', axis=1)
        y = self.heart['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
        return X_train, X_test, y_train, y_test

    def Diagnosis(self,X_test):
        # Scale the input data
        X_test_scaled = self.scaler.fit_transform(X_test)
        # Make predictions
        self.y_pred = self.rf.predict(X_test_scaled)
        # Print the accuracy score
        print('accuracy_score:', accuracy_score(self.y_pred, self.y_test))

    def getData(self):
        # Print the predictions
        print(self.y_pred)

# Example of how to use the class
diagnosis = HeartDiseaseDiagnosis()
diagnosis.Diagnosis(diagnosis.X_test)
diagnosis.getData()