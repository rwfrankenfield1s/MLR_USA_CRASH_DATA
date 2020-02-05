import pandas as pd
import numpy as np
import statsmodels.api as sm


from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split





data = pd.read_csv('US_Accidents_May19.csv')

states = ['FL','OH','NY']
data = data[data.State.isin(states)]
#data = data.head(10)
data = data.fillna(0)
#print(data.dtypes)
#print(data.shape)
#print(data.info)
##print(data['State'].nunique())

df = pd.DataFrame(data, columns=['State','Severity', 'Start_Lat', 'Start_Lng', 'Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)'
,'Wind_Speed(mph)','Precipitation(in)'])


X = df[['State','Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)'
,'Wind_Speed(mph)','Precipitation(in)']]
Y = df['Severity']

X = pd.get_dummies(X) 
print(X)



X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


lm = LinearRegression() 
lm.fit(X_Train, Y_Train)

pred = lm.predict(X_Test) 

print('y_pred = ', pred)
print("y_test = ", Y_Test)

print("Train : ", lm.score(X_Train, Y_Train)*100," %")
print("Test : ", lm.score(X_Test, Y_Test)*100," %")

#Backwards elimination Technique

X = np.append(arr = np.ones((329113, 1)).astype(int),  
              values = X, axis = 1) 

# choose a Significance level usually 0.05, if p>0.05 
#  for the highest values parameter, remove that value 


X_opt = X[:, [0, 1, 2, 3, 4, 5, 7, 8, 9]] 
ols = sm.OLS(endog = Y, exog = X_opt).fit() 
ols.summary() 
#print(ols.summary())


X_opt = X[:, [0, 1, 2, 3, 4, 5, 8, 9]] 
ols = sm.OLS(endog = Y, exog = X_opt).fit() 
ols.summary() 
#print(ols.summary())


X_opt = X[:, [0, 1, 2,  4, 5, 8, 9]] 
ols = sm.OLS(endog = Y, exog = X_opt).fit() 
ols.summary() 
#print(ols.summary())


X_opt = X[:, [0, 1, 4, 5, 8, 9]] 
ols = sm.OLS(endog = Y, exog = X_opt).fit() 
ols.summary() 
print(ols.summary())



