from numpy import hstack
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLP

class MMFFBleding:

    def __init__(self,X_train, y_train, X_test, y_test):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._models = None
        self._blender = None
    
    # get a list of base models
    def get_models(self):
        models = list()
        models.append(('mlp', MLP()))
        models.append(('rfr', RFR()))
        models.append(('gbr', GBR()))
        models.append(('svm', SVR()))
        return models
    
    # fit the blending ensemble
    def fit_ensemble(self, models, X_train, X_val, y_train, y_val):
        # fit all models on the training set and predict on hold out set
        meta_X = list()
        for _, model in models:
            # fit in training set
            model.fit(X_train, y_train)
            # predict on hold out set
            yhat = model.predict(X_val)
            # reshape predictions into a matrix with one column
            yhat = yhat.reshape(len(yhat), 1)
            # store predictions as input for blending
            meta_X.append(yhat)
        # create 2d array from predictions, each set is an input feature
        meta_X = hstack(meta_X)
        # define blending model
        blender = LinearRegression()
        # fit on predictions from base models
        blender.fit(meta_X, y_val)
        return blender
    
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
        # create the base models
        self._models = self.get_models()
        # train the blending ensemble
        self._blender = self.fit_ensemble(self._models, self._X_train, self._X_test, self._y_train, self._y_test)

        return self._blender

    def predict(self, X_test, blender=None):
        # make a prediction on a new row of data
        if blender == None:
            yhat = self.predict_ensemble(self._models, self._blender, X_test)
        else:
            yhat = self.predict_ensemble(self._models, blender, X_test)

        return yhat