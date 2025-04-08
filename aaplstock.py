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

AAPL_data = yf.download('AAPL', start = '2022-09-30', end = '2023-09-10', auto_adjust = True)

AAPL_data['Close'] = AAPL_data['Close']
AAPL_data['Lagged_Close'] = AAPL_data['Close'].shift(1)
AAPL_data.dropna(inplace = True)

columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume', 'Lagged_Close']

scaler = MinMaxScaler()

AAPL_data[columns_to_normalize] = scaler.fit_transform(AAPL_data[columns_to_normalize])

print(AAPL_data.head())

X = AAPL_data[['Lagged_Close']]
y = AAPL_data[['Close']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
 
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, short_window=12, long_window=26):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    return macd

AAPL_data['RSI'] = calculate_rsi(AAPL_data)
AAPL_data['MACD'] = calculate_macd(AAPL_data)

AAPL_data['Price_Movement'] = 'Stay'
AAPL_data.loc[AAPL_data['Close'] > AAPL_data['Close'].shift(1), 'Price_Movement'] = 'Increase'
AAPL_data.loc[AAPL_data['Close'] < AAPL_data['Close'].shift(1), 'Price_Movement'] = 'Decrease'

AAPL_data.dropna(inplace=True)

X = AAPL_data[['RSI', 'MACD']]
y = AAPL_data[['Price_Movement']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

plt.figure(figsize=(10,6))
colors = {'Increase': 'green', 'Decrease': 'red', 'Stay': 'blue'}

for category, color in colors.items():
    plt.scatter(X_test[y_test == category]['RSI'], X_test[y_test == category]['MACD'], c=color, label=category)


plt.xlabel('RSI')
plt.ylabel('MACD')
plt.title('Scatter Plot: RSI vs. MACD (Multi Class Classification)')
plt.legend()
plt.show()






