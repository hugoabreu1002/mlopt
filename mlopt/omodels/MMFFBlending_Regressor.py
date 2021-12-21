from numpy import hstack
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLP
from .EnsembleSearch import EnsembleSearch

class MMFFBlending:

    def __init__(self,X_train, y_train, X_test, y_test,
                 models=[('mlp', MLP()), ('rfr', RFR()), ('gbr', GBR()), ('svm', SVR())],
                 blender = DecisionTreeRegressor()):
        
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._models = models
        self._blender = blender

    def set_models(self, models=[]):
        self._models = models
    
    # get a list of base models
    def get_models(self):
        return self._models
    
    # fit the blending ensemble
    def fit_ensemble(self, models):
        # fit all models on the training set and predict on hold out set
        meta_X = list()
        for _, model in models:
            # fit in training set
            model.fit(self._X_train, self._y_train)
            # predict on hold out set
            yhat = model.predict(self._X_train)
            # reshape predictions into a matrix with one column
            yhat = yhat.reshape(len(yhat), 1)
            # store predictions as input for blending
            meta_X.append(yhat)
        # create 2d array from predictions, each set is an input feature
        meta_X = hstack(meta_X)
        # define blending model
        # fit on predictions from base models
        self._blender.fit(meta_X, self._y_train)
        return self._blender
    
    # make a prediction with the blending ensemble
    def predict_ensemble(self, models, blender, X_test):
        # make predictions with base models
        meta_X = list()
        for _, model in models:
            # predict with base model
            yhat = model.predict(X_test)
            # reshape predictions into a matrix with one column
            yhat = yhat.reshape(len(yhat), 1)
            # store prediction
            meta_X.append(yhat)               
        # create 2d array from predictions, each set is an input feature
        meta_X = hstack(meta_X)
        # predict
        return blender.predict(meta_X)

    def train(self):
        # summarize data split
        print('Train: %s, Test: %s' % (self._X_train.shape, self._X_test.shape))
        # train the blending ensemble
        self._blender = self.fit_ensemble(self._models)

        return self._blender

    def predict(self, X, models=None, blender=None):
        # make a prediction on a new row of data
        if blender == None:
            blender = self._blender
        if models == None:
            models = self._models
                
        yhat = self.predict_ensemble(models, blender, X)

        return yhat


class AGMMFFBlending(MMFFBlending):

    def __init__(self, X_train, y_train, X_test, y_test,
                 blender = DecisionTreeRegressor(),
                 epochs=5, size_pop=40, verbose=True):
        
        self._epochs = epochs
        self._size_pop = size_pop
        self._verbose = verbose
        super().__init__(X_train, y_train, X_test, y_test, blender=blender)

    def train(self):
        self._ensembleSearch = EnsembleSearch(self._X_train, self._y_train, self._X_test,
                                              self._y_test, epochs=self._epochs,
                                              size_pop=self._size_pop, verbose=self._verbose)
        bestPoolRegressors = self._ensembleSearch.search_best()._best_of_all
        print("Regressors chosen:")
        print(bestPoolRegressors.named_estimators_.items())
        models = bestPoolRegressors.named_estimators_.items()
        self.set_models(models)
        # summarize data split
        print('Train: %s, Test: %s' % (self._X_train.shape, self._X_test.shape))
        # train the blending ensemble
        self._blender = self.fit_ensemble(self._models)
        return models, self._blender
        
    