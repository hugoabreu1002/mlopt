from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import backend as K
from functools import partial
import random
import numpy as np
from tqdm import tqdm

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
    def __init__(self, name='ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128, denseSize=24, outputSize=1, ff_dim=None, num_layers=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)] + [Flatten(), Dense(denseSize), Dense(outputSize)]
        
    def call(self, inputs):
        time_embedding = keras.layers.TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return x # flat vector of features out K.reshape(x, (-1, x.shape[1] * x.shape[2]))

class TransformerKeras():
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
    
    def lr_scheduler(self, epoch, **lr_scheduler_kwargs):
        """
            base_lr must be greater than min_lr and initial_lr

            warmup_epochs must be lesser than epoch
            decay_epochs must be greater than wamup_epochs and lesser than epoch
        """
        if isinstance(lr_scheduler_kwargs, dict):
            warmup_epochs=lr_scheduler_kwargs.get('warmup_epochs')
            decay_epochs=lr_scheduler_kwargs.get('decay_epochs') 
            initial_lr=lr_scheduler_kwargs.get('initial_lr')
            base_lr=lr_scheduler_kwargs.get('base_lr')
            min_lr=lr_scheduler_kwargs.get('min_lr')
        else:
            warmup_epochs=20
            decay_epochs=150 
            initial_lr=1e-6 
            base_lr=1e-3
            min_lr=5e-5
        
        if epoch <= warmup_epochs:
            pct = epoch / warmup_epochs
            return ((base_lr - initial_lr) * pct) + initial_lr

        if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
            pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
            return ((base_lr - min_lr) * pct) + min_lr

        return min_lr

    def fitModel(self, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None,
                 num_layers=1, dropout=0, epochs=300,
                 early_stop=True, verbose=True, lr_warmnup=True, **lr_scheduler_kwargs):

        Model = None
        K.clear_session()
        
        Model = ModelTrunk(time2vec_dim=time2vec_dim, num_heads=num_heads, head_size=head_size,
                           ff_dim=ff_dim, num_layers=num_layers,
                           denseSize=self._X_train.shape[1],
                           outputSize=self._y_train.shape[1],
                           dropout=dropout)

        Model.compile(optimizer='adam', loss='mae', metrics=['mse'])
        Model.build(self._X_train.shape)
        Model.summary()

        for i in range(self._X_train.shape[1]):
            X_train_col = self._X_train[i]
            if np.isnan(np.sum(X_train_col)):
                raise("X train has nan in column {0}".format(i))

        if np.isnan(np.sum(self._y_train)):
            raise("y train has nan")

        if early_stop:
            patience = 10
        else:
            patience=epochs

        es = EarlyStopping(monitor='loss', mode='auto', patience=patience, verbose=1)

        my_callbacks = [es]
        if lr_warmnup:
            my_callbacks += [keras.callbacks.LearningRateScheduler(self.lr_scheduler(epoch=epochs, **lr_scheduler_kwargs), verbose=0)]

        fithistory = Model.fit(self._X_train, self._y_train, epochs=epochs, shuffle=False, use_multiprocessing=True, callbacks=my_callbacks, verbose=verbose)

        return Model, fithistory

class AGTransformerKeras():
    def __init__(self,X_train, y_train, X_test, y_test, num_generations, size_population, prob_mut, epochs_per_individual=200):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._epochs_per_individual = epochs_per_individual
        self._num_generations = num_generations
        self._size_population = size_population
        self._prob_mut = prob_mut
        self._fitness_array = np.array([])
        self._best_of_all = None
    
    def gen_population(self):
        """
            Generates the population, which is a list of lists.
            Every individual (list):

            individual_template = [head_size, num_heads, num_layers, warmup_epochs, decay_epochs, Transformer_Object, Fitness]
        """
        sizepop=self._size_population
        population = [['']]*sizepop

        for i in range(0, sizepop):
            population[i] = [random.randint(100,400), random.randint(1, 10),
             random.randint(1, 5), 
             random.randint(1, 30),
             random.randint(100, 150),
             'Transformer-object', 10]

        return population

    def set_fitness_and_sort(self, population, start_set_fit):
        for i in range(start_set_fit, len(population)):
            
            lr_scheduler_kwargs = {'warmup_epochs':population[i][3],'decay_epochs':population[i][4],
             'initial_lr':1e-6,
              'base_lr':1e-3,
              'min_lr':5e-5}

            transformer_fitted, fitHistory = TransformerKeras(self._X_train, self._y_train, self._X_test, self._y_test).fitModel(head_size=population[i][0],
            epochs=400,
            num_heads=population[i][1],
            num_layers=population[i][2],
            early_stop=True,
            **lr_scheduler_kwargs)

            population[i][-1] = fitHistory.history['val_loss'][-1]
            population[i][-2] = transformer_fitted
        
        population.sort(key = lambda x: x[:][-1])
        
        return population

    def cruzamento(self, population):
        qt_cross = len(population[0])
        pop_ori = population
        for p in range(1, len(pop_ori)):
            if np.random.rand() > (1 - self._prob_mut):
                population[p][0:int(qt_cross/2)] = pop_ori[int(p/2)][0:int(qt_cross/2)]
                population[p][int(qt_cross/2):qt_cross] = pop_ori[int(p/2)][int(qt_cross/2):qt_cross]

        return population

    def mutation(self, population):
        """
            Also has the constraints.
        """
        for p in range(1, len(population)):
            if np.random.rand() > (1 - self._prob_mut):
                population[p][0] = population[p][0] + np.random.randint(-20,20)
                if population[p][0] < 64:
                    population[p][0] =  64

                population[p][1] = population[p][1] + np.random.randint(-1,3)
                if population[p][1] < 2:
                    population[p][1] = 2
                
                population[p][2] = population[p][2] + np.random.randint(-1,2)
                
                if population[p][2] < 1:
                    population[p][2] = 1
                
                population[p][3] = population[p][3] + np.random.randint(-5,20)

                if population[p][3] < 10:
                    population[p][3] = 10

                population[p][4] = population[p][4] + np.random.randint(-10,10)

                if population[p][4] < 50:
                    population[p][4] = 50

        return population
    
    def new_gen(self, population, num_gen):
        population = self.cruzamento(population)
        population = self.mutation(population)
        start_set_fit = int(self._size_population*num_gen/(2*self._num_generations))
        population = self.set_fitness_and_sort(population, start_set_fit)
        return population
    
    def search_best_individual(self):
        """
            Returns best fitted transformer network
        """
        population = self.gen_population()
        population = self.set_fitness_and_sort(population, 0)

        self._fitness_array= np.append(self._fitness_array, population[0][-1])
        self._best_of_all = population[0][-2]

        for ng in tqdm(range(0, self._num_generations)):
            population = self.new_gen(population, ng)
            
            if population[0][-1] < min(self._fitness_array):
                self._best_of_all = population[0][-2]

        self._final_trained_mlps = [p[-2] for p in population]
        
        return self._best_of_all 

