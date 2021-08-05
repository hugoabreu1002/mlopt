import pickle
from TimeSeriesUtils import train_test_split_with_Exog, SMAPE
from TimeSeriesUtils import train_test_split as train_test_split_noExog
import tpot
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from hpsklearn import HyperoptEstimator, any_regressor, any_preprocessing
from hyperopt import tpe
import autokeras as ak
import os
import tensorflow as tf
import numpy as np
from ACOLSTM import ACOLSTM, ACOCLSTM
from MMFFBleding_Regressor import AGMMFFBleding
import traceback
import datetime
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd

#TODO Docsctrings
class TimeSeriesTester():
    def __init__(self) -> None:
        pass

    def applyTPOT(self, X_train, y_train, X_test, y_test, SavePath, popSize=20,
                number_Generations=5, kFolders=0, TPOTSingleMinutes=1,
                TPOTFullMinutes = 10, useSavedModels = True):
        if not useSavedModels or not os.path.isfile(SavePath):
            pipeline_optimizer = tpot.TPOTRegressor(generations=number_Generations, #number of iterations to run the training
                                                    population_size=popSize, #number of individuals to train
                                                    cv=kFolders, #number of folds in StratifiedKFold
                                                    max_eval_time_mins=TPOTSingleMinutes, #time in minutes for each trial
                                                    max_time_mins=TPOTFullMinutes, #time in minutes for whole optimization
                                                    scoring="neg_mean_absolute_error") 
            
            pipeline_optimizer.fit(X_train, y_train) #fit the pipeline optimizer - can take a long time
            pipeline_optimizer.export(SavePath)
            
        else:
            print("######### PLACE THE EXPORTED PIPELINE CODE HERE ########")
            
        print("TPOT - Score: {0}".format(-pipeline_optimizer.score(X_test, y_test)))
        y_hat = pipeline_optimizer.predict(X_test)
        print("MAE: %.4f" % mean_absolute_error(y_test, y_hat))
            
        return y_hat

    def applyHPSKLEARN(self, X_train, y_train, X_test, y_test, SavePath,
                    max_evals=100, trial_timeout=100, useSavedModels = True):

        if not useSavedModels or not os.path.isfile(SavePath+".pckl"):
            HPSKLEARNModel = HyperoptEstimator(regressor=any_regressor('reg'),
                                    preprocessing=any_preprocessing('pre'),
                                    loss_fn=mean_squared_error,
                                    max_evals=max_evals,
                                    trial_timeout=trial_timeout,
                                    algo=tpe.suggest)
            # perform the search
            HPSKLEARNModel.fit(X_train, y_train)
            pickle.dump(HPSKLEARNModel, open(SavePath+".pckl", 'wb'))
        else:
            HPSKLEARNModel = pickle.load(open(SavePath+".pckl", 'rb'))

        # summarize performance
        score = HPSKLEARNModel.score(X_test, y_test)
        y_hat = HPSKLEARNModel.predict(X_test)
        print("HPSKLEARN - Score: ")
        print("MAE: %.4f" % score)
        # summarize the best model
        print(HPSKLEARNModel.best_model())
        
        return y_hat
        
    def applyAutoKeras(self, X_train, y_train, X_test, y_test, SavePath,
                    max_trials=100, epochs=300, useSavedModels = True):

        if not useSavedModels or not os.path.isdir(SavePath+"/keras_auto_model/best_model/"):
            input_node = ak.StructuredDataInput()
            output_node = ak.DenseBlock()(input_node)
            #output_node = ak.ConvBlock()(output_node)
            output_node = ak.RegressionHead()(output_node)
            AKRegressor = ak.AutoModel(
                inputs=input_node,
                outputs=output_node,
                max_trials=max_trials,
                overwrite=True,
                tuner="bayesian",
                project_name=SavePath+"/keras_auto_model"
            )
            print(" X_train shape: {0}\n y_train shape: {1}\n X_test shape: {2}\n y_test shape: {3}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
            AKRegressor.fit(x=X_train, y=y_train[:,0],epochs=epochs,verbose=1, batch_size=int(X_train.shape[0]/10), shuffle=False, use_multiprocessing=True)
            AKRegressor.export_model()
        else:
            AKRegressor = tf.keras.models.load_model(SavePath+"/keras_auto_model/best_model/")
            
        y_hat = AKRegressor.predict(X_test)
        print("AUTOKERAS - Score: ")
        print("MAE: %.4f" % mean_absolute_error(y_test[:,0], y_hat))
            
        return y_hat

    def applyACOLSTM(self, X_train, y_train, X_test, y_test, SavePath,
                    Layers_Qtd=[[40, 50, 60, 70], [20, 25, 30], [5, 10, 15]],
                    epochs=[100,200,300],
                    options_ACO={'antNumber':5, 'antTours':5, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1},
                    useSavedModels = True):

        if not useSavedModels or not os.path.isdir(SavePath):
            lstmOptimizer = ACOLSTM(X_train, y_train, X_test, y_test, n_variables=1 ,options_ACO=options_ACO, verbose=True)
            final_model, y_hat = lstmOptimizer.optimize(Layers_Qtd = Layers_Qtd, epochs=epochs)
            final_model.save(SavePath)
            del lstmOptimizer
        else:
            print(SavePath)
            final_model = tf.keras.models.load_model(SavePath)
            y_hat = final_model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))


        if len(y_hat.shape) > 1:
            y_hat = y_hat[:,0]
            
        print("ACOLSTM - Score: ")
        print("MAE: %.4f" % mean_absolute_error(y_test, y_hat))

        return y_hat

    def applyACOCLSTM(self, X_train, y_train, X_test, y_test, SavePath,
                    Layers_Qtd=[[50, 30, 20, 10], [20, 15, 10], [10, 20], [5, 10], [2, 4]],
                    ConvKernels=[[8, 12], [4, 6]],
                    epochs=[10],
                    options_ACO={'antNumber':5, 'antTours':5, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1},
                    useSavedModels = True):

        if not useSavedModels or not os.path.isdir(SavePath):
            clstmOptimizer = ACOCLSTM(X_train, y_train, X_test, y_test, 1 ,options_ACO=options_ACO, verbose=True)
            final_model, y_hat = clstmOptimizer.optimize(Layers_Qtd = Layers_Qtd, ConvKernels = ConvKernels, epochs=epochs)
            final_model.save(SavePath)
            del clstmOptimizer
        else:
            print(SavePath)
            final_model = tf.keras.models.load_model(SavePath)
            y_hat = final_model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
            
        if len(y_hat.shape) > 1:
            y_hat = y_hat[:,0]
            
        print("ACOCLSTM - Score: ")
        print("MAE: %.4f" % mean_absolute_error(y_test, y_hat))

        return y_hat

    def applyGAMMFF(self, X_train, y_train, X_test, y_test, SavePath,
                    epochs=5, size_pop=40, useSavedModels = True):

        agMMGGBlending = AGMMFFBleding(X_train, y_train, X_test, y_test, epochs=epochs, size_pop=size_pop)
        if not useSavedModels or not os.path.isfile(SavePath+".pckl"):
            final_models, final_blender = agMMGGBlending.train()
            y_hat = agMMGGBlending.predict(X_test=X_test, blender=final_blender, models=final_models)
            pickle.dump(final_blender, open(SavePath+"_blender.pckl", 'wb'))
            pickle.dump(final_blender, open(SavePath+"_models.pckl", 'wb'))
        else:
            final_blender = pickle.load(open(SavePath+"_blender.pckl", 'rb'))
            final_models = pickle.load(open(SavePath+"_models.pckl", 'rb'))
            y_hat = agMMGGBlending.predict(X_test=X_test, blender=final_blender, models=final_models)
            

        print("AGMMFF - Score: ")
        print("MAE: %.4f" % mean_absolute_error(y_test, y_hat))

        return y_hat

    def applyETS(self, X_train, y_train, X_test, y_test, SavePath):
        y_pos = np.concatenate([y_train, y_test]) + 0.01
        fit1 = ExponentialSmoothing(y_pos, seasonal_periods=24, trend="add", seasonal="add",use_boxcox=False,initialization_method="estimated").fit()
        fit2 = ExponentialSmoothing(y_pos,seasonal_periods=24,trend="add",seasonal="mul",use_boxcox=False,initialization_method="estimated").fit()
        fit3 = ExponentialSmoothing(y_pos,seasonal_periods=24,trend="add",seasonal="add",damped_trend=True,use_boxcox=False,initialization_method="estimated",).fit()
        fit4 = ExponentialSmoothing(y_pos,seasonal_periods=24,trend="add",seasonal="mul",damped_trend=True,use_boxcox=False,initialization_method="estimated").fit()
        results = pd.DataFrame(index=[r"$\alpha$", r"$\beta$", r"$\phi$", r"$\gamma$", r"$l_0$", "$b_0$", "SSE (SUM OF SQUARED ERRORS)"])
        params = ["smoothing_level","smoothing_trend","damping_trend","smoothing_seasonal","initial_level","initial_trend"]
        results["Additive"] = [fit1.params[p] for p in params] + [fit1.sse]
        results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]
        results["Additive Dam"] = [fit3.params[p] for p in params] + [fit3.sse]
        results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse]
        best = results.columns[results["SSE (SUM OF SQUARED ERRORS)"].argmin()]


    def saveResults(self, y_test, y_hats, labels, save_path, timestap_now):
        logResults = ""
        logResults += "Scores" + "\n"
        print("Scores")

        for y_hat, plotlabel in zip(y_hats, labels):
            print("ploting... " + plotlabel)
            logResults += "{0} ".format(plotlabel) + "- MAE: %.4f" % mean_absolute_error(y_test, y_hat) + "\n"
            logResults += "{0} ".format(plotlabel) + "- MAPE: %.4f" % MAPE(y_test, y_hat) + "\n"
            logResults += "{0} ".format(plotlabel) + "- SMAPE: %.4f" % SMAPE(y_test, y_hat) + "\n"
            logResults += "{0} ".format(plotlabel) + "- MSE: %.4f" % mean_squared_error(y_test, y_hat) + "\n"

        with open(save_path+"/results_{0}.txt".format(timestap_now), "w") as text_file:
            text_file.write(logResults)

        print(logResults)

    def executeTests(self, y_data, exog_data, autoMlsToExecute="All", train_test=[80,20],
                     lags=24, useSavedModels=True, useSavedArrays=True,
                     save_path="./TimeSeriesTester/"):
        """
            autoMlsToExecute="All"

            or insert the automls in a list like autoMlsToExecute=["tpot", "hpsklearn", "autokeras", "agmmff", "acolstm", "acoclstm"]
        """
        X_train, y_train, X_test, y_test  = train_test_split_with_Exog(y_data, exog_data, lags, train_test)

        timestamp_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        y_hats = []
        labels = []

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.savetxt(save_path+"./y_test_{0}".format(timestamp_now), y_test.reshape(-1, 1), delimiter=';')

        if "tpot" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("TPOT Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_TPOT"):
                    y_hat_tpot = self.applyTPOT(X_train, y_train, X_test, y_test, save_path+"/tpotModel_{0}".format(timestamp_now),
                                        popSize=10, number_Generations=10,
                                        useSavedModels = useSavedModels)
                    np.savetxt(save_path+"/y_hat_TPOT", y_hat_tpot, delimiter=';')
                else:
                    y_hat_tpot = np.loadtxt(save_path+"/y_hat_TPOT", delimiter=';')
                    
                y_hats.append(y_hat_tpot)
                labels.append("TPOT")
            except Exception:
                traceback.print_exc()
                pass

        if "hpsklearn" in autoMlsToExecute or autoMlsToExecute=="All":    
            try:
                print("HPSKLEARN Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_HPSKLEARN"):
                    y_hat_HPSKLEARN = self.applyHPSKLEARN(X_train, y_train, X_test, y_test, save_path+"/HPSKLEARNModel_{0}".format(timestamp_now),
                                                max_evals=100, useSavedModels = useSavedModels)
                    np.savetxt(save_path+"/y_hat_HPSKLEARN", y_hat_HPSKLEARN, delimiter=';')
                else:
                    y_hat_HPSKLEARN = np.loadtxt(save_path+"/y_hat_HPSKLEARN", delimiter=';')

                y_hats.append(y_hat_HPSKLEARN)
                labels.append("HPSKLEARN")
            except Exception:
                traceback.print_exc()
                pass

        if "autokeras" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("AUTOKERAS Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_AUTOKERAS"):
                    y_hat_autokeras = self.applyAutoKeras(X_train, y_train, X_test, y_test, save_path+"/autokerastModel_{0}".format(timestamp_now),
                                                max_trials=10, epochs=300, useSavedModels = useSavedModels)
                    np.savetxt(save_path+"/y_hat_AUTOKERAS", y_hat_autokeras, delimiter=';')
                else:
                    y_hat_autokeras = np.loadtxt(save_path+"/y_hat_AUTOKERAS", delimiter=';')
                    
                y_hats.append(y_hat_autokeras)
                labels.append("AUTOKERAS")
            except Exception:
                traceback.print_exc()
                pass

        if "agmmff" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("AGMMFF Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_AGMMFF"):
                    y_hat_agmmff= self.applyGAMMFF( X_train, y_train, X_test, y_test,
                                            save_path+"/mmffModel_{0}".format(timestamp_now),
                                            epochs=3, size_pop=20, useSavedModels = useSavedModels)
                    np.savetxt(save_path+"/y_hat_AGMMFF", y_hat_agmmff, delimiter=';')
                else:
                    y_hat_agmmff = np.loadtxt(save_path+"/y_hat_AGMMFF", delimiter=';')

                print("SHAPE HAT {0}".format(y_hat_agmmff.shape))
                y_hats.append(y_hat_agmmff)
                labels.append("AGMMFF")
            except Exception:
                traceback.print_exc()
                pass

        ##################################################################################################
        #################################### NO EXOG MODELS ##############################################    
        ##################################################################################################
        X_train_noexog, y_train_noexog, X_test_noexog, y_test_noexog = train_test_split_noExog(y_data, 23,
                                                                            tr_vd_ts_percents = [80, 20],
                                                                            print_shapes = useSavedModels)
        options_ACO={'antNumber':6, 'antTours':5, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1}

        if "acolstm" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("ACOLSTM Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_ACOLSTM"):
                    Layers_Qtd=[[60, 40, 30], [20, 15], [10, 7, 5]]
                    epochs=[300]        
                    y_hat_acolstm = self.applyACOLSTM(X_train_noexog, y_train_noexog, X_test_noexog, y_test_noexog,
                                                save_path+"/acolstmModel_{0}".format(timestamp_now),
                                                Layers_Qtd, epochs, options_ACO, useSavedModels = useSavedModels)
                    np.savetxt(save_path+"/y_hat_ACOLSTM", y_hat_acolstm, delimiter=';')
                else:
                    y_hat_acolstm = np.loadtxt(save_path+"/y_hat_ACOLSTM", delimiter=';')
                y_hats.append(y_hat_acolstm)
                labels.append("ACOLSTM")
            except Exception:
                traceback.print_exc()
                pass

        if "acoclstm" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("ACOCLSTM Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_ACOCLSTM"):
                    Layers_Qtd=[[60, 40], [30, 15], [60, 40, 30], [20, 15], [10, 7, 5]]
                    ConvKernels=[[8, 12], [6, 4]]
                    epochs=[300]
                    y_hat_acoclstm = self.applyACOCLSTM(X_train_noexog, y_train_noexog, X_test_noexog, y_test_noexog,
                                                save_path+"/acoclstmModel_{0}".format(timestamp_now),
                                                Layers_Qtd, ConvKernels, epochs, options_ACO, useSavedModels = useSavedModels)
                        
                    np.savetxt(save_path+"/y_hat_ACOCLSTM", y_hat_acoclstm, delimiter=';')
                else:
                    y_hat_acoclstm = np.loadtxt(save_path+"/y_hat_ACOCLSTM", delimiter=';')

                print("SHAPE HAT {0}".format(y_hat_acoclstm.shape))
                y_hats.append(y_hat_acoclstm)
                labels.append("ACOCLSTM")
            except Exception:
                traceback.print_exc()
                pass

        

        self.saveResults(y_test, y_hats, labels, save_path, timestamp_now)