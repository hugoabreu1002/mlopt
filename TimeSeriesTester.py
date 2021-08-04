import pickle
from mlopt.TimeSeriesUtils import train_test_split_with_Exog, SMAPE
from mlopt.TimeSeriesUtils import train_test_split as train_test_split_noExog
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import argparse
import tpot
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from hpsklearn import HyperoptEstimator, any_regressor, any_preprocessing
from hyperopt import tpe
import autokeras as ak
import os
from matplotlib import pyplot as plt
import warnings
import tensorflow as tf
import numpy as np
from mlopt.ACOLSTM import ACOLSTM, ACOCLSTM
from mlopt.MMFFBleding_Regressor import AGMMFFBleding
import traceback
import datetime

class TimeSeriesTester():
    def __init__(self) -> None:
        pass

    def applyTPOT(X_train, y_train, X_test, y_test, SavePath, popSize=20,
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

    def applyHPSKLEARN(X_train, y_train, X_test, y_test, SavePath,
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
        
    def applyAutoKeras(X_train, y_train, X_test, y_test, SavePath,
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

    def applyACOLSTM(X_train, y_train, X_test, y_test, SavePath,
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

    def applyACOCLSTM(X_train, y_train, X_test, y_test, SavePath,
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

    def applyGAMMFF(X_train, y_train, X_test, y_test, SavePath,
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

    def saveResults(df_inmet, genscaler, y_test, y_hats, labels, city_save_path, showPlot):
        logResults = ""
        logResults += "Scores" + "\n"
        print("Scores")
        
        _, ax = plt.subplots(1,1, figsize=(14,7), dpi=300)
        ticks_X = df_inmet.data.astype('str') + '-' + df_inmet.hora.astype('str')
        len_dt = len(y_test)
        ticks_X = ticks_X[-len_dt:].values
        ax.plot(ticks_X, genscaler.inverse_transform(y_test.reshape(-1, 1)), 'k-o', label='Original', linewidth=2.0)

        for y_hat, plotlabel in zip(y_hats, labels):
            print("ploting... " + plotlabel)
            logResults += "{0} ".format(plotlabel) + "- MAE: %.4f" % mean_absolute_error(y_test, y_hat) + "\n"
            logResults += "{0} ".format(plotlabel) + "- MAPE: %.4f" % MAPE(y_test, y_hat) + "\n"
            logResults += "{0} ".format(plotlabel) + "- SMAPE: %.4f" % SMAPE(y_test, y_hat) + "\n"
            logResults += "{0} ".format(plotlabel) + "- MSE: %.4f" % mean_squared_error(y_test, y_hat) + "\n"
            trueScale_yhat = genscaler.inverse_transform(y_hat[-len_dt:].reshape(-1, 1))
            ax.plot(ticks_X, trueScale_yhat, '--o', label=plotlabel)

        plt.xticks(ticks_X[::3], rotation=45, ha='right', fontsize=12)
        ax.grid(axis='x')
        ax.legend(fontsize=13)
        ax.set_ylabel('W/m2', fontsize=14)
        cidade = city_save_path.split("/")[2]
        if cidade == "joaopessoa":
            cidade = "João Pessoa"
        elif cidade == "saoluis":
            cidade = "São Luis"
        elif cidade == "maceio":
            cidade = "Maceió"
        ax.set_title(cidade.capitalize(), fontsize=14)
        timestamp_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.tight_layout()
        plt.savefig(city_save_path+"/AutoMLS_result_{0}.png".format(timestamp_now), dpi=300)

        if showPlot:
            plt.show()

        with open(city_save_path+"/results_{0}.txt".format(timestamp_now), "w") as text_file:
            text_file.write(logResults)

        print(logResults)

    def executeTests(self, y_data, exog_data, train_test=[80,20], lags=24, autoMlsToExecute, plot=True, useSavedModels=True, useSavedArrays=True):

        X_train, y_train, X_test, y_test  = train_test_split_with_Exog(y_data[:,0], exog_data, lags, train_test)

        y_hats = []
        labels = []

        np.savetxt(city_save_path+"./y_test", y_test.reshape(-1, 1), delimiter=';')

        if "tpot" in autoMlsToExecute:
            try:
                print("TPOT Evaluation...")
                if not useSavedArrays or not os.path.isfile(city_save_path+"/y_hat_TPOT"):
                    y_hat_tpot = applyTPOT(X_train, y_train, X_test, y_test, city_save_path+"/tpotModel_{0}".format(city),
                                        popSize=10, number_Generations=10,
                                        useSavedModels = useSavedModels)
                    np.savetxt(city_save_path+"/y_hat_TPOT", y_hat_tpot, delimiter=';')
                else:
                    y_hat_tpot = np.loadtxt(city_save_path+"/y_hat_TPOT", delimiter=';')
                    
                y_hats.append(y_hat_tpot)
                labels.append("TPOT")
            except Exception:
                traceback.print_exc()
                pass

        if "hpsklearn" in autoMlsToExecute:    
            try:
                print("HPSKLEARN Evaluation...")
                if not useSavedArrays or not os.path.isfile(city_save_path+"/y_hat_HPSKLEARN"):
                    y_hat_HPSKLEARN = applyHPSKLEARN(X_train, y_train, X_test, y_test, city_save_path+"/HPSKLEARNModel_{0}".format(city),
                                                max_evals=100, useSavedModels = useSavedModels)
                    np.savetxt(city_save_path+"/y_hat_HPSKLEARN", y_hat_HPSKLEARN, delimiter=';')
                else:
                    y_hat_HPSKLEARN = np.loadtxt(city_save_path+"/y_hat_HPSKLEARN", delimiter=';')

                y_hats.append(y_hat_HPSKLEARN)
                labels.append("HPSKLEARN")
            except Exception:
                traceback.print_exc()
                pass

        if "autokeras" in autoMlsToExecute:
            try:
                print("AUTOKERAS Evaluation...")
                if not useSavedArrays or not os.path.isfile(city_save_path+"/y_hat_AUTOKERAS"):
                    y_hat_autokeras = applyAutoKeras(X_train, y_train, X_test, y_test, city_save_path+"/autokerastModel_{0}".format(city),
                                                max_trials=10, epochs=300, useSavedModels = useSavedModels)
                    np.savetxt(city_save_path+"/y_hat_AUTOKERAS", y_hat_autokeras, delimiter=';')
                else:
                    y_hat_autokeras = np.loadtxt(city_save_path+"/y_hat_AUTOKERAS", delimiter=';')
                    
                y_hats.append(y_hat_autokeras)
                labels.append("AUTOKERAS")
            except Exception:
                traceback.print_exc()
                pass

        if "agmmff" in autoMlsToExecute:
            try:
                print("AGMMFF Evaluation...")
                if not useSavedArrays or not os.path.isfile(city_save_path+"/y_hat_AGMMFF"):
                    y_hat_agmmff= applyGAMMFF( X_train, y_train, X_test, y_test,
                                            city_save_path+"/mmffModel_{0}".format(city),
                                            epochs=3, size_pop=20, useSavedModels = useSavedModels)
                    np.savetxt(city_save_path+"/y_hat_AGMMFF", y_hat_agmmff, delimiter=';')
                else:
                    y_hat_agmmff = np.loadtxt(city_save_path+"/y_hat_AGMMFF", delimiter=';')

                print("SHAPE HAT {0}".format(y_hat_agmmff.shape))
                y_hats.append(y_hat_agmmff)
                labels.append("AGMMFF")
            except Exception:
                traceback.print_exc()
                pass

        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm = train_test_split_noExog(gen[:,0], 23,
                                                                            tr_vd_ts_percents = [80, 20],
                                                                            print_shapes = useSavedModels)
        options_ACO={'antNumber':6, 'antTours':5, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1}

        if "acolstm" in autoMlsToExecute:
            try:
                print("ACOLSTM Evaluation...")
                if not useSavedArrays or not os.path.isfile(city_save_path+"/y_hat_ACOLSTM"):
                    Layers_Qtd=[[60, 40, 30], [20, 15], [10, 7, 5]]
                    epochs=[300]        
                    y_hat_acolstm = applyACOLSTM(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
                                                city_save_path+"/acolstmModel_{0}".format(city),
                                                Layers_Qtd, epochs, options_ACO, useSavedModels = useSavedModels)
                    np.savetxt(city_save_path+"/y_hat_ACOLSTM", y_hat_acolstm, delimiter=';')
                else:
                    y_hat_acolstm = np.loadtxt(city_save_path+"/y_hat_ACOLSTM", delimiter=';')
                y_hats.append(y_hat_acolstm)
                labels.append("ACOLSTM")
            except Exception:
                traceback.print_exc()
                pass

        if "acoclstm" in autoMlsToExecute:
            try:
                print("ACOCLSTM Evaluation...")
                if not useSavedArrays or not os.path.isfile(city_save_path+"/y_hat_ACOCLSTM"):
                    Layers_Qtd=[[60, 40], [30, 15], [60, 40, 30], [20, 15], [10, 7, 5]]
                    ConvKernels=[[8, 12], [6, 4]]
                    epochs=[300]
                    y_hat_acoclstm = applyACOCLSTM(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
                                                city_save_path+"/acoclstmModel_{0}".format(city),
                                                Layers_Qtd, ConvKernels, epochs, options_ACO, useSavedModels = useSavedModels)
                        
                    np.savetxt(city_save_path+"/y_hat_ACOCLSTM", y_hat_acoclstm, delimiter=';')
                else:
                    y_hat_acoclstm = np.loadtxt(city_save_path+"/y_hat_ACOCLSTM", delimiter=';')

                print("SHAPE HAT {0}".format(y_hat_acoclstm.shape))
                y_hats.append(y_hat_acoclstm)
                labels.append("ACOCLSTM")
            except Exception:
                traceback.print_exc()
                pass

        saveResults(df_inmet, genscaler, y_test, y_hats, labels, city_save_path, showPlot = plot)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def main(self, args):
        citiesFolders = args.listaCidades
        for city in citiesFolders:
            self.executeTests(city, args.cidadesRootFolder,
                        autoMlsToExecute = args.autoMls,
                        plot = args.plot,
                        useSavedModels = args.useSavedModels,
                        useSavedArrays = args.useSavedArrays)
        
        return None