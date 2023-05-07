import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from metaflow import FlowSpec, step

class TrainFlow(FlowSpec):

    @step
    def start(self):
        self.data_url = 'credit_filtered.csv'
        self.next(self.read_data)
    @step
    def read_data(self):
        self.data = pd.read_csv(self.data_url)
        self.next(self.preprocess_data)
    @step
    def preprocess_data(self):
        cat_vars = self.data.select_dtypes(include=['object']).columns
        num_vars = self.data.select_dtypes(exclude=['object']).columns
        encoder = LabelEncoder()
        encoded_vars = self.data[cat_vars].apply(encoder.fit_transform)
        self.data_encoded = pd.concat([self.data[num_vars], encoded_vars], axis=1)

        self.next(self.split_data)
        
    @step
    def split_data(self):
        target = 'isFlaggedFraud'
        X = self.data_encoded.drop(target, axis=1)
        y = self.data_encoded[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model_params = [
                {'penalty': 'l1', 'C': 1},
                {'penalty': 'l2', 'C': 1},
                {'penalty': 'l1', 'C': 0.5},
                {'penalty': 'l2', 'C': 0.5}
        ]
        print('Started modeling, num columns in train = ', self.X_train.shape[1])
        self.next(self.train, foreach='model_params')
    @step
    def train(self):
        self.model = LogisticRegression(penalty=self.input['penalty'], C=self.input['C'], solver='liblinear')
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.next(self.choose_best_model)
    @step
    def choose_best_model(self, inputs):
        print('Choosing best model')
        self.best_accuracy = 0
        self.best_model = None
        for inp in inputs:
            if inp.accuracy>self.best_accuracy:
                self.best_accuracy = inp.accuracy
                self.best_model = inp.model
        print("Best accuracy:", self.best_accuracy)
        self.next(self.end)
    
    @step
    def end(self):
        pass

    def infer(self, input_vector):
        return self.best_model.predict(input_vector)

if __name__ == '__main__':
    TrainFlow()
