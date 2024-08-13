
#Importing the necessary libraries 

import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns


os.environ["KERAS_BACKEND"] = "tensorflow"


##Helper Class##
#--stockdf: Loads in the historical stock data from the yfinance library
#---Uses a framework to focus on price across various stages of the trading day: i.e. close, open, high, low, etc.
#---Additional framework placeholders for future development

class finHelp:

    def __init__(self,tkr,type,sdate,edate,intv):
        self.tkr = tkr
        self.type = type
        self.sdate = sdate
        self.edate = edate
        self.intv = intv

    def stockdf(self,price):
        tkr_fetch = yf.Ticker(self.tkr)

        if self.type == 'historical_price':
            hstprice = tkr_fetch.history(start=self.sdate,end=self.edate,interval=self.intv)

            if price == 'close':
                hstprice = hstprice['Close']
                return hstprice
            elif price == 'open':
                hstprice = hstprice['Open']
                return hstprice
            elif price == 'high':
                hstprice = hstprice['High']
                return hstprice
            elif price == 'low':
                hstprice = hstprice['Low']
                return hstprice
            elif price == 'all':
                hstprice = hstprice[['Open','Close','High','Low','Volume']].reset_index()
                hstprice['Mean'] = (hstprice['High']+hstprice['Low'])/2
                hstprice['gaps'] = hstprice['Close'].shift(1)
                hstprice['mean_shift'] = hstprice['Mean'].shift(1)
                hstprice['open_day_change'] = (hstprice['Open']/hstprice['gaps'])-1
                hstprice['mean_change'] = (hstprice['Mean']/hstprice['mean_shift'])-1
                hstprice['prev_open'] = hstprice['Open'].shift(1)
                hstprice['prev_close'] = hstprice['Close'].shift(1)
                hstprice['Datetime'] = pd.to_datetime(hstprice['Datetime'])
                hstprice['date'] = hstprice['Datetime'].dt.date
                hstprice['time'] = hstprice['Datetime'].dt.time
                hstprice['open_close_hr'] = hstprice['time'].astype(str).apply(lambda x: 1 if x == '09:30:00' else (2 if x == '15:30:00' else 0))
                hstprice['mov_avg'] = hstprice['Mean'].rolling(window=50).mean()
                hstprice['mov_avg_diff'] = (hstprice['mov_avg']/hstprice['Mean'])-1
                hstprice['inside_days'] = hstprice.apply(lambda x: 1 if (x['Open'] < x['prev_open']) and (x['Close'] < x['prev_close']) else 0, axis=1)
                hstprice['vol_var'] = (hstprice['Volume']/(hstprice['Volume'].median()))-1
                hstprice['abs_mnChng'] = abs(hstprice['mean_change'])
                return hstprice
            elif price == 'mean':
                mnprice = (hstprice['High']+hstprice['Low'])/2
                hstprice['Mean'] = mnprice
                return hstprice
            else:
                return ValueError('Must pass "close", "open", "high", "low", "mean"')

        elif self.type == 'volume':
            hstvol = tkr_fetch.history(start=self.sdate,end=self.edate,interval=self.intv)
            hstvol = hstvol['Volume']
            return hstvol

        else:
            return ValueError('Must pass "historical_price" or "volume"')


##Feature Engineering Class##
#--Leverages previous arguments made during the data sourcing for additional preprocessing
#--rsi: Calculates the relative strength index
#--targetvar: calculates a standard deviation for the target variable

class deriveVar(finHelp):

    def __init__(self,tkr,type,sdate,edate,intv):
        super().__init__(tkr,type,sdate,edate,intv)

    def rsi(self,period):
        closeprc = self.stockdf(price='close')
        delta = closeprc.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rels = avg_gain/avg_loss

        rsi = 100 - (100/(1+rels))

        rsi_df = pd.DataFrame({
            'Close': closeprc,
            'RSI': rsi
        })

        rsi_df = rsi_df.reset_index()

        return rsi_df

    def targetvar(self,df,col):
        std_dev = df[col].std()
        return std_dev


##Neural Network Preprocessing Class##
#--Processes the data so the model can leverage it for computation
#--preprocess_time: creates a static method to process the cyclical nature of timestamp data before scaling
#--tt_split: splits the data into training/testing datasets of features and targets, processes timestamp data, and performs a scaling of the features
#--scaler: performs preprocessing to run the neural network on production data
class ml_preprocess:

    def __init__(self,df):
        self.df = df

    @staticmethod
    def preprocess_time(df, time_col):
        df = df.copy(deep=True)
        df[time_col] = pd.to_datetime(df[time_col], format='%H:%M:%S').dt.time
        df['hour'] = df[time_col].apply(lambda x: x.hour + x.minute / 60)
        df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df.drop(columns=[time_col, 'hour'], inplace=True)
        return df

    def tt_split(self,target_col,time_col=None):
        df = self.df.copy(deep=True)
        features = df.drop(columns=[target_col])
        target = df[target_col]
        if time_col is not None:
            features = ml_preprocess.preprocess_time(features, time_col)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2024)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        return X_train, X_test, y_train, y_test

    def scaler(self,target_col,time_col=None):
        df = self.df.copy(deep=True)
        features = df.drop(columns=[target_col])
        target = df[target_col]
        if time_col is not None:
            features = ml_preprocess.preprocess_time(features, time_col)
        numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
        return features, target



##Neural Network Preprocessing Class##
#--Constructs the model itself and the various accuracy measures
#--feedforward_construct: constructs the MLP feedforward neural network
#--gen_accuracy: computes the general accuracy for training and testing sets 
#--density_acc: creates a density plot to assist in visualizing the accuracy of the binary classification
#--tgt_acc: plots the target and predicted values against production level data to assess classification accuracy

class nn_model:

    def __init__(self,learning_rate=0.01,epochs=15,batch_size=32,pltdf=None):
        learning_rate.self = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.pltdf = pltdf

    def feedforward_construct(self):
        model = Sequential()
        model.add(layers.Dense(input_dim=len(X_train.columns),units=10,activation='relu'))
        model.add(layers.Dense(units=8,activation='relu'))
        model.add(layers.Dense(units=5, activation='relu'))
        model.add(layers.Dense(units=1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=self.learning_rate),metrics=['accuracy'])
        fitted_model = model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        return fitted_model, model

    def gen_accuracy(self,model):
        train_acc = model.evaluate(X_train, y_train)[1]
        test_acc = model.evaluate(X_test, y_test)[1]
        return train_acc, test_acc

    def density_acc(self):
        fp0 = full_predictions_df[full_predictions_df['True Label'] == 0]
        fp1 = full_predictions_df[full_predictions_df['True Label'] == 1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sns.kdeplot(x=fp0['index'], y=fp0['Predicted'], cmap='coolwarm', fill=True, alpha=0.5, ax=ax1)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Prediction Value')
        ax1.set_title('Predictions of True Classification 0')
        sns.kdeplot(x=fp1['index'], y=fp1['Predicted'], cmap='coolwarm', fill=True, alpha=0.5, ax=ax2)
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Prediction Value')
        ax2.set_title('Predictions of True Classification 1')
        plt.tight_layout()
        plt.show()

    def tgt_acc(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
        ax1.plot(self.pltdf.index, self.pltdf['Mean'], 'b-', label='Mean')
        ax1.set_ylabel('Mean')
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(self.pltdf.index, self.pltdf['Predicted'], 'r-', alpha=0.5, label='Predicted')
        ax1_twin.set_ylabel('Predicted Classification')
        ax1_twin.tick_params(axis='y', labelcolor='black')
        ax2.plot(self.pltdf.index, self.pltdf['Mean'], 'b-', label='Mean')
        ax2.set_ylabel('Mean')
        ax2.set_xlabel('Index')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.tick_params(axis='y', labelcolor='black')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.pltdf.index, self.pltdf['True Label'], 'g-', alpha=0.5, label='True Label')
        ax2_twin.set_ylabel('Target Variable')
        ax2_twin.tick_params(axis='y', labelcolor='black')
        fig.suptitle('Predicted Classification vs. Target Variable')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        plt.tight_layout()
        plt.show()




#--------------------------------
###Data Preprocessing###
#--1) Selecting a specific ticker symbol and date range for the training. Post COVID dates were specifically chose to reduce bias. 
#--2) Class objects initiated for sourcing and preprocessing.
#--3) Preprocessed data is finally joined, scaled and split into the training/testing sets.          

ticker = 'SPY'
start_date = '2023-01-01'
end_date = '2024-02-01'
sk = finHelp(tkr='SPY',type='historical_price',sdate=start_date,edate=end_date,intv='1h')
dv = deriveVar(tkr='SPY', type='historical_price', sdate=start_date, edate=end_date, intv='1h')
stk3 = sk.stockdf(price='all')

rsi_df = dv.rsi(period=13).drop(columns='Close')

sDf2 = stk3.merge(rsi_df,on='Datetime',how='left')
target_level = dv.targetvar(sDf2,'mean_change')
sDf2['target'] = sDf2['abs_mnChng'].apply(lambda x: 1 if x >= target_level else 0)
sDf2['adjusted_target'] = sDf2['target'].shift(-10)
sDf2['channel_ptrn'] = sDf2['abs_mnChng'].rolling(window=12).mean()
sDf2 = sDf2.drop(columns=['Open','Close','High','Low','Mean','gaps','mean_shift','mean_change','prev_open','prev_close','Datetime','date','mov_avg','abs_mnChng','target'])
sDf2 = sDf2.iloc[50:-10]


ml = ml_preprocess(df=sDf2)


#X_train: 80% training features
#X_test: 20% testing features
#y_train: 80% training targets
#y_test: 20% testing targets
X_train, X_test, y_train, y_test = ml.tt_split(target_col='adjusted_target',time_col='time')
print(X_train)


#--------------------------------
###Modeling Training and Accuracy Test###
#--1) A MLP feedforward model object is created using the standard set parameters
#--2) The general accuracy is then computed through the gen_accuracy method

mlp = nn_model(learning_rate=0.01,epochs=15,batch_size=32)

fitted, model = mlp.feedforward_construct()

train_acc, test_acc = mlp.gen_accuracy(model=model)

print('Training accuracy: %s' % train_acc)
print('Testing accuracy: %s' % test_acc)


#--------------------------------
###Assessing True vs Predicted Probabilities###
#--1) The train and test set predictions are converted into a dataframe indexed on their original index from the 
#---training and testing sets for sequential order and compared to the true prediction

y_train_pred_proba = model.predict(X_train)
y_test_pred_proba = model.predict(X_test)


train_preds_df = pd.DataFrame(y_train_pred_proba, columns=['Predicted'], index=X_train.index)
train_preds_df['True Label'] = y_train

test_preds_df = pd.DataFrame(y_test_pred_proba, columns=['Predicted'], index=X_test.index)
test_preds_df['True Label'] = y_test

full_predictions_df = pd.concat([train_preds_df, test_preds_df])

full_predictions_df = full_predictions_df.sort_index().reset_index()

full_predictions_df



#--------------------------------
###Assessing Accuracy Density###
#--1) Creating a new nn_model object and calling the density_acc method

dens = nn_model(pltdf=full_predictions_df)

dens.density_acc()



#--------------------------------
###Testing True Target Accuracy for Future Unseen Data###
#--1) Following the same steps as the "data preprocessing" stage previous except on a future set of unseen data
#--2) The former model is the used to predict the binary classification 
#--3) The predictions are formatted into a similar dataframe as before and filtered to compute the true positive accuracy

ticker = 'SPY'
start_date = '2024-05-01'
end_date = '2024-08-01'
sk = finHelp(tkr='SPY',type='historical_price',sdate=start_date,edate=end_date,intv='1h')
dv = deriveVar(tkr='SPY', type='historical_price', sdate=start_date, edate=end_date, intv='1h')
stk4 = sk.stockdf(price='all')

rsi_df2 = dv.rsi(period=13).drop(columns='Close')

sDf3 = stk4.merge(rsi_df2,on='Datetime',how='left')
target_level = dv.targetvar(sDf3,'mean_change')
sDf3['target'] = sDf3['abs_mnChng'].apply(lambda x: 1 if x >= target_level else 0)
sDf3['adjusted_target'] = sDf3['target'].shift(-10)
sDf3['channel_ptrn'] = sDf3['abs_mnChng'].rolling(window=12).mean()
sDf3 = sDf3.iloc[50:-10]
base_future = sDf3.copy(deep=True)
sDf3 = sDf3.drop(columns=['Open','Close','High','Low','Mean','gaps','mean_shift','mean_change','prev_open','prev_close','Datetime','date','mov_avg','abs_mnChng','target'])


ml = ml_preprocess(df=sDf3)

new_features, new_target = ml.scaler(target_col='adjusted_target',time_col='time')

y_new_pred_proba = model.predict(new_features)
y_new_pred_proba = (y_new_pred_proba > 0.5).astype(int)

new_preds_df = pd.DataFrame(y_new_pred_proba, columns=['Predicted'], index=new_features.index)
new_preds_df['True Label'] = new_target

full_dataset = sDf3.copy()
full_dataset = full_dataset.rename_axis('Index').reset_index()
price_df = base_future.rename_axis('Index').reset_index()
price_df = price_df[['Index','Datetime','Mean']]
merged_df = price_df.merge(new_preds_df, left_on='Index', right_index=True, how='left')

accuracy1 = merged_df[merged_df['Predicted'] == 1]
accuracy1['True Label'] = accuracy1['True Label'].astype(int)
tot = accuracy1['Predicted'].count()
truepred = accuracy1['True Label'].sum()
print(f'Accuracy:{truepred/tot}')


#--------------------------------
###Target Predictions Viz###
#--1) Creating a new nn_model object and calling the tgt_acc method

tgt = nn_model(pltdf=merged_df)

tgt.tgt_acc()



