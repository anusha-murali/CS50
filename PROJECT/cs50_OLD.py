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
     new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

     for i in range(0,len(data)):
         new_data['Date'][i] = data['Date'][i]
         new_data['Close'][i] = data['Close'][i]

     new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
     new_data.index = new_data['Date']
     return new_data

# Plot historical data as well as the predictions
# We assume both train and valid has columns, 'Close' and 'Predictions'
#
def plotStock(train, valid):
    plt.plot(train)
    plt.plot(valid)
    plt.show()
    
def linearRegression():
    print("Linear Regression")
    # LINEAR REGRESSION

    new_data = readFile("IBM.csv")

    #create features
    from fastai.structured import  add_datepart
    add_datepart(new_data, 'Date')
    new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

    new_data['mon_fri'] = 0
    for i in range(0,len(new_data)):
        if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
            new_data['mon_fri'][i] = 1
        else:
            new_data['mon_fri'][i] = 0

    #split into train and validation
    train = new_data[:200]
    valid = new_data[200:]

    x_train = train.drop('Close', axis=1)
    y_train = train['Close']
    x_valid = valid.drop('Close', axis=1)
    y_valid = valid['Close']

    #implement linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train,y_train)

    valid['Predictions'] = model.predict(x_valid)

    valid.index = new_data[200:].index
    train.index = new_data[:200].index

    plotStock(train['Close'], valid[['Close', 'Predictions']])
    

def fbProphet():
     print("Facebook Prophet")
     # FACEBOOK PROPHET
     #
     #importing prophet
     from fbprophet import Prophet

     new_data = readFile('IBM.csv')

     #preparing data
     new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

     #train and validation
     train = new_data[:200]
     valid = new_data[200:]

     #fit the model
     model = Prophet()
     model.fit(train)

     #predictions
     close_prices = model.make_future_dataframe(periods=len(valid))
     forecast = model.predict(close_prices)
     forecast_valid = forecast['yhat'][200:]
          
     valid['Predictions'] = forecast_valid.values

     plotStock(train['y'], valid[['y', 'Predictions']])


def rnn():
     print("RNN")

def lstm():
     print("LSTM")
     # LSTM

     new_data = readFile('IBM.csv')
     
     new_data.drop('Date', axis=1, inplace=True)

     #creating train and test sets
     dataset = new_data.values

     train = dataset[0:200,:]
     valid = dataset[200:,:]

     #converting dataset into x_train and y_train
     scaler = MinMaxScaler(feature_range=(0, 1))
     scaled_data = scaler.fit_transform(dataset)

     x_train, y_train = [], []
     for i in range(60,len(train)):
         x_train.append(scaled_data[i-60:i,0])
         y_train.append(scaled_data[i,0])
     x_train, y_train = np.array(x_train), np.array(y_train)

     x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

     # create and fit the LSTM network
     model = Sequential()
     model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
     model.add(LSTM(units=50))
     model.add(Dense(1))

     model.compile(loss='mean_squared_error', optimizer='adam')
     model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2)

     #predicting 246 values, using past 60 from the train data
     inputs = new_data[len(new_data) - len(valid) - 60:].values
     inputs = inputs.reshape(-1,1)
     inputs  = scaler.transform(inputs)

     X_test = []
     for i in range(60,inputs.shape[0]):
         X_test.append(inputs[i-60:i,0])
     X_test = np.array(X_test)

     X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
     closing_price = model.predict(X_test)
     closing_price = scaler.inverse_transform(closing_price)

     rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
     rms

     #for plotting
     train = new_data[:200]
     valid = new_data[200:]
     valid['Predictions'] = closing_price

     plotStock(train['Close'], valid[['Close', 'Predictions']])


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
