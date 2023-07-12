import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the Electric Production dataset
data = pd.read_csv('Electric_Production.csv')

# Split the dataset into features (X) and target variable (y)
X = pd.to_datetime(data['DATE'])
y = data['IPG2211A2N']

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train = X.iloc[:len(data) - 12]
X_test = X.iloc[len(data) - 12:]
y_train = y.iloc[:len(data) - 12]
y_test = y.iloc[len(data) - 12:]

# Initialize the SVR model
svr = SVR()
# Train the model
result = svr.fit(X_train.values.reshape(-1, 1), y_train)

# Make predictions on the testing and training sets
y_pred_test = svr.predict(X_test.values.reshape(-1, 1))
y_pred_train = svr.predict(X_train.values.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# Plot the graph
plt.figure(figsize = (10, 10))
plt.plot(X_train, y_train, color = 'blue', label = 'Actual (Train)')
plt.plot(X_test, y_test, color = 'green', label = 'Actual (Test)')
plt.plot(X_train, y_pred_train, color = 'red', label = 'Predicted (Train)')
plt.plot(X_test, y_pred_test, color = 'orange', label = 'Predicted (Test)')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('SVR Demand Forecasting')
plt.legend()
plt.show()

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Make forecast

X_forecast =  pd.date_range(start='1/1/1985', periods = 500, freq='M')

forecast = svr.predict(X_forecast.values.reshape(-1, 1))
  
plt.figure(figsize = (10, 10))
plt.plot(X_forecast, forecast, color = 'blue', label = 'Forecast')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('SVR Demand Forecasting')
plt.legend()
plt.show()

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)