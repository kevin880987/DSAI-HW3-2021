
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import inspect

from datetime import datetime, timedelta

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# self=Model()
class Model():
    def __init__(self):
        pass
                    
    # X_train, Y_train=X.copy(), Y.copy()
    def build_model(self, X_train, Y_train): # , X_val, Y_val
        layers = 2 # 4 # 
        # units = 24 # 4 # 
        dropout = 0.05
        
        loss = 'mse'
        optimizer = 'adam'
        
        epochs = 2 # 10000 # 
        batch_size = 128
        patience = 100
        
        starttime = datetime.now()
        print()
        print()
        print(inspect.currentframe().f_code.co_name)
        print('\tstart time:', starttime)
        print()

        nn_model = Sequential()
        layer_sizes = np.linspace(X_train.shape[1], Y_train.shape[1], layers)
        for i, u in zip(range(layers), layer_sizes):
            if i<layers-1:
                return_sequences = True
            else:
                return_sequences = False
                
            nn_model.add(GRU(units=int(np.round(max(1, u))), input_shape=(X_train.shape[1], 1), 
                         return_sequences=return_sequences))
            nn_model.add(Dropout(dropout))
        
        nn_model.add(Dense(Y_train.shape[1])) # nn_model.add(TimeDistributed(Dense(1)))
        nn_model.compile(loss=loss, optimizer=optimizer)
        nn_model.summary()
        
        X_train_re = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        Y_train_re = Y_train[:, :, np.newaxis]
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience) # 
        # mc = ModelCheckpoint('best_model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        history = nn_model.fit(X_train_re, Y_train_re, 
                               validation_split=0.2, 
                               epochs=epochs, batch_size=batch_size, 
                               callbacks=[], verbose=True) # , validation_data=(X_val, Y_val) # es, mc

        # Visualize
        plt.plot(history.history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Function Value')
        plt.legend(['Loss'])
        plt.show()

        endtime = datetime.now()
        print()
        print(inspect.currentframe().f_code.co_name)
        print('\tend time:', endtime)
        print('\ttime consumption:', endtime-starttime)
        print()
        print()
        
        return nn_model
        
# a=self
# self=a
# training_df=df_training.copy()
    def train(self, X_train, Y_train):
        # Scale Y
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler = scaler.fit(Y_train)
        Y_train = scaler.transform(Y_train)
        self.scaler = scaler
        
        # # Load nn_model
        # nn_model = load_model('nn_model.h5')
        
        # Build and fit nn_model
        nn_model = self.build_model(X_train, Y_train)
        
        # Save nn_model
        nn_model.save('nn_model.h5')
        self.nn_model = nn_model

        # Predict train and visualize
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        plt.plot(Y_train, ax=ax, label=True)
        for _, x in X_train.iterrows():
            # Predict
            x = x.to_frame().transpose()
            Y_pred = self.predict(x)
            ax.plot(Y_pred, label=False)

        plt.title('Training')
        plt.xlabel('Time')
        plt.ylabel('Target Value')
        plt.legend(['Real Y'])
        plt.show()

    def predict(self, x=None, display=False):
        # # x is dataframe
        
        # if x is None:
        #     display = True
        #     x = self.X_test.iloc[[-1]]
        
        try:
            nn_model = self.nn_model
        except:
            nn_model = load_model('nn_model.h5')
            
        # testing = x.values
        
        # Predict
        # Y_pred = nn_model.predict(self.X_train_re[[x], :, :])
        Y_pred = nn_model.predict(np.reshape(x, (x.shape[0], x.shape[1], 1)))
        # Y_pred = pd.DataFrame(Y_pred, index=x.index[[-1]], 
        #                       columns=self.Y_train.columns)
        
        # Inverse
        Y_pred = self.scaler.inverse(Y_pred)
        
        return Y_pred


