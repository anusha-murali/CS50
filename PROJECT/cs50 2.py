# Anusha and Helen
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

rcParams['figure.figsize'] = 18,8
scaler = MinMaxScaler(feature_range=(0, 1))
register_matplotlib_converters()

def readFile(data_file):
     df = pd.read_csv(data_file)
      #creating dataframe with date and the target variable
     data = df.sort_index(ascending=True, axis=0)

     #creating dataframe
     stockData = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

     for i in range(0,len(data)):
         stockData['Date'][i] = data['Date'][i]
         stockData['Close'][i] = data['Close'][i]

     stockData['Date'] = pd.to_datetime(stockData.Date,format='%Y-%m-%d')
     stockData.index = stockData['Date']
     return stockData

# Plot historical data as well as the predictions
# We assume both train and valid has columns, 'Close' and 'Predictions'
#
def plotStock(trainSet, testSet):
    plt.plot(trainSet)
    plt.plot(testSet)
    plt.show()
    
def linearRegression():
    print("Linear Regression")
    # LINEAR REGRESSION

    stockData = readFile("IBM.csv")

    # We use fastai add_datepart to generate the date parts
    #
    from fastai.structured import  add_datepart
    add_datepart(stockData, 'Date')
    stockData.drop('Elapsed', axis=1, inplace=True)  

    stockData['mon_fri'] = 0
    
    for i in range(0,len(stockData)):
        if (stockData['Dayofweek'][i] == 0 or stockData['Dayofweek'][i] == 4):
            stockData['mon_fri'][i] = 1
        else:
            stockData['mon_fri'][i] = 0

    # Use 80% of the dataset as the train set
    # and the remaining 20% as the test set
    N = round(0.8*len(stockData))

    # Now split the data into trainSet and testSet
    #
    trainSet = stockData[:N]
    testSet = stockData[N:]

    x_trainSet = trainSet.drop('Close', axis=1)
    y_trainSet = trainSet['Close']
    x_testSet = testSet.drop('Close', axis=1)
    y_testSet = testSet['Close']

    #implement linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_trainSet,y_trainSet)

    testSet['Predictions'] = model.predict(x_testSet)

    testSet.index = stockData[N:].index
    trainSet.index = stockData[:N].index

    plotStock(trainSet['Close'], testSet[['Close', 'Predictions']])
    

def fbProphet():
     print("Facebook Prophet")
     # FACEBOOK PROPHET
     #
     #importing prophet
     from fbprophet import Prophet

     stockData = readFile('IBM.csv')

     #preparing data
     stockData.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

     # Use 80% of the dataset as the train set
     # and the remaining 20% as the test set
     N = round(0.8*len(stockData))
     
     #train and validation
     train = stockData[:N]
     valid = stockData[N:]

     #fit the model
     model = Prophet()
     model.fit(train)

     #predictions
     close_prices = model.make_future_dataframe(periods=len(valid))
     forecast = model.predict(close_prices)
     forecast_valid = forecast['yhat'][N:]
          
     valid['Predictions'] = forecast_valid.values

     plotStock(train['y'], valid[['y', 'Predictions']])


def rnn():
     print("RNN")

def lstm():
     print("LSTM")
     # LSTM

     stockData = readFile('IBM.csv')
     
     stockData.drop('Date', axis=1, inplace=True)

     #creating train and test sets
     dataset = stockData.values

     # Use 80% of the dataset as the train set
     # and the remaining 20% as the test set
     N = round(0.8*len(dataset))

     trainSet = dataset[0:N,:]
     testSet = dataset[N:,:]

     #converting dataset into x_trainSet and y_trainSet
     scaler = MinMaxScaler(feature_range=(0, 1))
     scaled_data = scaler.fit_transform(dataset)

     x_trainSet, y_trainSet = [], []
     for i in range(60,len(trainSet)):
         x_trainSet.append(scaled_data[i-60:i,0])
         y_trainSet.append(scaled_data[i,0])
     x_trainSet, y_trainSet = np.array(x_trainSet), np.array(y_trainSet)

     x_trainSet = np.reshape(x_trainSet, (x_trainSet.shape[0],x_trainSet.shape[1],1))

     # create and fit the LSTM network
     model = Sequential()
     model.add(LSTM(units=50, return_sequences=True, input_shape=(x_trainSet.shape[1],1)))
     model.add(LSTM(units=50))
     model.add(Dense(1))

     model.compile(loss='mean_squared_error', optimizer='adam')
     model.fit(x_trainSet, y_trainSet, epochs=1, batch_size=1, verbose=2)

     #predicting 246 values, using past 60 from the train data
     inputs = stockData[len(stockData) - len(testSet) - 60:].values
     inputs = inputs.reshape(-1,1)
     inputs  = scaler.transform(inputs)

     X_test = []
     for i in range(60,inputs.shape[0]):
         X_test.append(inputs[i-60:i,0])
     X_test = np.array(X_test)

     X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
     closing_price = model.predict(X_test)
     closing_price = scaler.inverse_transform(closing_price)

     #for plotting
     trainSet = stockData[:N]
     testSet = stockData[N:]
     testSet['Predictions'] = closing_price

     plotStock(trainSet['Close'], testSet[['Close', 'Predictions']])


def report():
     print("Comparison Report")

def menu():
    print("************ CS50 Stock Forecast **************")
    print()

    choice = input("""
                      A: Linear Regression
                      B: Facebook Prophet
                      C: RNN
                      D: LSTM
                      Q: Comparision Report

                      Please enter your choice: """)

    if choice == "A" or choice =="a":
        linearRegression()
    elif choice == "B" or choice =="b":
        fbProphet()
    elif choice == "C" or choice =="c":
        rnn()
    elif choice=="D" or choice=="d":
        lstm()
    elif choice == "E" or choice == "e":
         report()
    elif choice=="Q" or choice=="q":
        sys.exit
    else:
        print("Invalid choice")
        menu()


menu()
