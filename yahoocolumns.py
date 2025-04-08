from datetime import timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import seaborn as sn

# Step 1: Download the data from Yahoo Finance
applstock = 'AAPL'
data = yf.download(applstock, start='2011-09-30', end='2024-11-12', auto_adjust=True)

data['Prev Close'] = data['Close'].shift(1)
data['High-Low'] = data['High'] - data['Low']
data['Open-Close'] = data['Open'] - data['Close']
data['Volume'] = data['Volume']
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

data = data.dropna()

x = ['Prev Close', 'High-Low', 'Open-Close', 'Volume', 'MA10', 'MA50']
y = ['Close']

x_train, x_test, y_train, y_test = train_test_split(data[x], data[y], test_size=0.2, random_state=123)

model = RandomForestRegressor(n_estimators=100,random_state=123)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error:  {mse}')

plt.figure(figsize=(14,7))
plt.scatter(y_test.index, y_test, label='Actual Price', color='blue')
plt.scatter(y_test.index, y_pred, label='Predicted Price', color='red')
plt.legend()
plt.show()

last_date = data.index[-1]
future_dates = [last_date + timedelta(days=1) for i in range(1,6)]
future_data = pd.DataFrame(index=future_dates, columns=x.columns)

last_row = data.iloc[-1]
for date in future_dates:
    future_data.loc[date, 'Prev Close'] = last_row['Close']
    future_data.loc[date, 'High-Low'] = last_row['High-Low']
    future_data.loc[date, 'Open-Close'] = last_row['Open-Close']
    future_data.loc[date, 'Volume'] = last_row['Volume']
    future_data.loc[date, 'MA10'] = data['Close'].rolling(window=10).mean()[-1]
    future_data.loc[date, 'MA50'] = data['Close'].rolling(window=50).mean()[-1]
    
future_price = model.predict(future_data)

print(f'Predicted Price on 5/16: {future_price}')







