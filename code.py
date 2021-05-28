import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
x = data['YearExperience'].values
y = data['Salary'].values
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
model.coef_
model.predict([[6]])
import joblib
joblib.dump(model , 'data1')
joblib.load('data1')
