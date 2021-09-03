from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras import backend as K
from functools import partial

import numpy as np

class Time2Vec(keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        self.bb = self.add_weight(name='bb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        self.ba = self.add_weight(name='ba',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))

class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1) 

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs + x)
        return x

class ModelTrunk(keras.Model):
    def __init__(self, name='ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]

        
    def call(self, inputs):
        time_embedding = keras.layers.TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return K.reshape(x, (-1, x.shape[1] * x.shape[2])) # flat vector of features out


class Transformer():
    """
        X_train: X_train for lstm.
        y_train: y_train for lstm.
        X_test: X_test for lstm.
        y_test: y_test for lstm.

        n_variables: N variables to be predicted, equals to y_test.shape[1]
        
        train_test_split: division in train and test for X and y in lstm training and test.
    
        options_ACO: parametrization for ACO algorithm. EG:
            {'antNumber':2, 'antTours':1, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2}
    """
    def __init__(self, X_train, y_train, X_test, y_test, n_variables=1):
        self._X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_variables))
        self._y_train = y_train
        self._X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_variables))
        self._y_test = y_test
    
    def lr_scheduler(self, epoch, lr, warmup_epochs=15, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
        if epoch <= warmup_epochs:
            pct = epoch / warmup_epochs
            return ((base_lr - initial_lr) * pct) + initial_lr

        if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
            pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
            return ((base_lr - min_lr) * pct) + min_lr

        return min_lr

    def getModel(self, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0):
        model = None
        K.clear_session()
        
        model = ModelTrunk()
        
        model.compile(optimizer='adam', loss='mae', metrics=['mse'])
            
        return model

    def fitModel(self, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0, epochs=300, early_stop=True, verbose=True):
        
        Model = self.getModel(time2vec_dim, num_heads, head_size, ff_dim, num_layers, dropout)

        for i in range(self._X_train.shape[1]):
            X_train_col = self._X_train[i]
            if np.isnan(np.sum(X_train_col)):
                raise("X train has nan in column {0}".format(i))

        if np.isnan(np.sum(self._y_train)):
            raise("y train has nan")

        # simple early stopping
        if early_stop:
            es = EarlyStopping(monitor='loss', mode='auto', patience=5, verbose=1)
        else:
            es = EarlyStopping(monitor='loss', mode='auto', patience=100, verbose=1)

        my_callbacks = [keras.callbacks.LearningRateScheduler(partial(self.lr_scheduler), verbose=0), es]

        Model.fit(self._X_train, self._y_train, epochs=epochs, shuffle=False, use_multiprocessing=True, callbacks=my_callbacks, verbose=verbose)

        
        return Model