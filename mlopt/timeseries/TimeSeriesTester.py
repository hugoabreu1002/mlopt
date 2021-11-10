from copy import copy

from matplotlib import pyplot as plt
from .AGMLP_Residual import AGMLP_VR_Residual
import pickle
from .TimeSeriesUtils import sarimax_PSO_ACO_search, train_test_split_with_Exog, SMAPE, MAPE
from .TimeSeriesUtils import train_test_split as train_test_split_noExog
from .TimeSeriesUtils import train_test_split_prev
import tpot
from sklearn.metrics import mean_absolute_error, mean_squared_error
from hpsklearn import HyperoptEstimator, any_regressor, any_preprocessing
from hyperopt import tpe
#import autokeras as ak
import os
import tensorflow as tf
import numpy as np
from ..omodels.ACOLSTM import ACOLSTM, ACOCLSTM
from ..omodels.MMFFBlending_Regressor import AGMMFFBlending
import traceback
import datetime
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd

#TODO Docsctrings, show final models of automls
class TimeSeriesTester():
    def __init__(self, verbose=False) -> None:
        self._verbose = verbose
        pass

    def _applyTPOT(self, X_train, y_train, X_test, y_test, SavePath, popSize=20,
                number_Generations=5, kFolders=5, TPOTSingleMinutes=1,
                TPOTFullMinutes = 10, useSavedModels = True):
        if not useSavedModels or not os.path.isfile(SavePath):
            pipeline_optimizer = tpot.TPOTRegressor(generations=number_Generations, #number of iterations to run the training
                                                    population_size=popSize, #number of individuals to train
                                                    cv=kFolders, #number of folds in StratifiedKFold
                                                    max_eval_time_mins=TPOTSingleMinutes, #time in minutes for each trial
                                                    max_time_mins=TPOTFullMinutes, #time in minutes for whole optimization
                                                    scoring="neg_mean_absolute_error",
                                                    verbosity=1) 
            
            pipeline_optimizer.fit(X_train, y_train) #fit the pipeline optimizer - can take a long time
            pipeline_optimizer.export(SavePath)
        else:
            print("######### PLACE THE EXPORTED PIPELINE CODE HERE ########")
            
        print("TPOT - Score: {0}".format(-pipeline_optimizer.score(X_test, y_test)))
        y_hat = pipeline_optimizer.predict(X_test)
        print("MAE: %.4f" % mean_absolute_error(y_test, y_hat))
            
        return y_hat

    def _applyHPSKLEARN(self, X_train, y_train, X_test, y_test, SavePath,
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
        
    # def _applyAutoKeras(self, X_train, y_train, X_test, y_test, SavePath,
    #                 max_trials=100, epochs=300, useSavedModels = True):

    #     if not useSavedModels or not os.path.isdir(SavePath+"/keras_auto_model/best_model/"):
    #         input_node = ak.StructuredDataInput()
    #         output_node = ak.DenseBlock()(input_node)
    #         #output_node = ak.ConvBlock()(output_node)
    #         output_node = ak.RegressionHead()(output_node)
    #         AKRegressor = ak.AutoModel(
    #             inputs=input_node,
    #             outputs=output_node,
    #             max_trials=max_trials,
    #             overwrite=True,
    #             tuner="bayesian",
    #             project_name=SavePath+"/keras_auto_model"
    #         )
    #         print(" X_train shape: {0}\n y_train shape: {1}\n X_test shape: {2}\n y_test shape: {3}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    #         AKRegressor.fit(x=X_train, y=y_train[:,0],epochs=epochs,verbose=1, batch_size=int(X_train.shape[0]/10), shuffle=False, use_multiprocessing=True)
    #         AKRegressor.export_model()
    #     else:
    #         AKRegressor = tf.keras.models.load_model(SavePath+"/keras_auto_model/best_model/")
            
    #     y_hat = AKRegressor.predict(X_test)
    #     print("AUTOKERAS - Score: ")
    #     print("MAE: %.4f" % mean_absolute_error(y_test[:,0], y_hat))
            
    #     return y_hat

    def _applyACOLSTM(self, X_train, y_train, X_test, y_test, SavePath,
                    Layers_Qtd=[[40, 50, 60, 70], [20, 25, 30], [5, 10, 15]],
                    epochs=[100,200,300],
                    options_ACO={'antNumber':5, 'antTours':5, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1},
                    useSavedModels = True):

        if not useSavedModels or not os.path.isdir(SavePath):
            lstmOptimizer = ACOLSTM(X_train, y_train, X_test, y_test, n_variables=1 ,options_ACO=options_ACO, verbose=self._verbose)
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

    def _applyACOCLSTM(self, X_train, y_train, X_test, y_test, SavePath,
                    Layers_Qtd=[[50, 30, 20, 10], [20, 15, 10], [10, 20], [5, 10], [2, 4]],
                    ConvKernels=[[8, 12], [4, 6]],
                    epochs=[10],
                    options_ACO={'antNumber':5, 'antTours':5, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1},
                    useSavedModels = True):

        if not useSavedModels or not os.path.isdir(SavePath):
            clstmOptimizer = ACOCLSTM(X_train, y_train, X_test, y_test, 1 ,options_ACO=options_ACO, verbose=self._verbose)
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

    def _applyGAMMFF(self, X_train, y_train, X_test, y_test, SavePath,
                    epochs=5, size_pop=40, useSavedModels = True):

        agMMGGBlending = AGMMFFBlending(X_train, y_train, X_test, y_test, epochs=epochs, size_pop=size_pop, verbose=self._verbose)
        if not useSavedModels or not (os.path.isfile(SavePath+"blender.pckl") and os.path.isfile(SavePath+"models.pckl")):
            final_models, final_blender = agMMGGBlending.train()
            y_hat = agMMGGBlending.predict(X=X_test, blender=final_blender, models=final_models)
            pickle.dump(final_blender, open(SavePath+"blender.pckl", 'wb'))
            pickle.dump(list(final_models), open(SavePath+"models.pckl", 'wb'))
        else:
            final_blender = pickle.load(open(SavePath+"blender.pckl", 'rb'))
            final_models = pickle.load(open(SavePath+"models.pckl", 'rb'))[0]
            y_hat = agMMGGBlending.predict(X=X_test, blender=final_blender, models=final_models)
            

        print("AGMMFF - Score: ")
        print("MAE: %.4f" % mean_absolute_error(y_test, y_hat))

        return y_hat

    def _applyETS(self, y_train, y_test):
        y_pos = np.concatenate([y_train, y_test]) + 0.01
        all_fits = []
        fit1 = ExponentialSmoothing(y_pos, seasonal_periods=24, trend="add", seasonal="add",use_boxcox=False,initialization_method="estimated").fit()
        fit2 = ExponentialSmoothing(y_pos,seasonal_periods=24,trend="add",seasonal="mul",use_boxcox=False,initialization_method="estimated").fit()
        fit3 = ExponentialSmoothing(y_pos,seasonal_periods=24,trend="add",seasonal="add",damped_trend=True,use_boxcox=False,initialization_method="estimated",).fit()
        fit4 = ExponentialSmoothing(y_pos,seasonal_periods=24,trend="add",seasonal="mul",damped_trend=True,use_boxcox=False,initialization_method="estimated").fit()
        all_fits = [fit1,fit2, fit3, fit4]
        
        results = pd.DataFrame(index=[r"$\alpha$", r"$\beta$", r"$\phi$", r"$\gamma$", r"$l_0$", "$b_0$", "SSE"])
        params = ["smoothing_level","smoothing_trend","damping_trend","smoothing_seasonal","initial_level","initial_trend"]
        results["Additive"] = [fit1.params[p] for p in params] + [fit1.sse]
        results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]
        results["Additive Dam"] = [fit3.params[p] for p in params] + [fit3.sse]
        results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse]
        print(results)
        best = results.loc["SSE"][results.loc["SSE"].iloc[:].values.argmin()]
        print("BEST ETS: " + str(best))

        best_fit = all_fits[results.loc["SSE"].iloc[:].values.argmin()]

        y_hat = best_fit.fittedvalues[-len(y_test):]

        print("ETS - Score:")
        print("MAE: %.4f" % mean_absolute_error(y_test, y_hat))

        return y_hat

    def _applyETSMLPEnsemble(self, endo_var, SavePath, tr_ts_percents=[80,20], popsize=10, numberGenerations=3, useSavedModels = True):
        y_pos = endo_var + 0.001
        all_fits = []
        fit1 = ExponentialSmoothing(y_pos, seasonal_periods=24, trend="add", seasonal="add",use_boxcox=False,initialization_method="estimated").fit()
        fit2 = ExponentialSmoothing(y_pos,seasonal_periods=24,trend="add",seasonal="mul",use_boxcox=False,initialization_method="estimated").fit()
        fit3 = ExponentialSmoothing(y_pos,seasonal_periods=24,trend="add",seasonal="add",damped_trend=True,use_boxcox=False,initialization_method="estimated",).fit()
        fit4 = ExponentialSmoothing(y_pos,seasonal_periods=24,trend="add",seasonal="mul",damped_trend=True,use_boxcox=False,initialization_method="estimated").fit()
        all_fits = [fit1,fit2, fit3, fit4]
        
        results = pd.DataFrame(index=[r"$\alpha$", r"$\beta$", r"$\phi$", r"$\gamma$", r"$l_0$", "$b_0$", "SSE"])
        params = ["smoothing_level","smoothing_trend","damping_trend","smoothing_seasonal","initial_level","initial_trend"]
        results["Additive"] = [fit1.params[p] for p in params] + [fit1.sse]
        results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]
        results["Additive Dam"] = [fit3.params[p] for p in params] + [fit3.sse]
        results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse]
        print(results)
        best = results.loc["SSE"][results.loc["SSE"].iloc[:].values.argmin()]
        print("BEST ETS: " + str(best))

        best_ets = all_fits[results.loc["SSE"].iloc[:].values.argmin()]

        y_ets = best_ets.fittedvalues
        y_pos = y_pos - 0.0001

        pickle.dump(best_ets, open(SavePath+"best_ets", 'wb'))
        
        if not useSavedModels or not os.path.isfile(SavePath+"ets_vr_residual.pckl"):
            ag_mlp_vr_residual = AGMLP_VR_Residual(y_pos, y_ets,
                                                   num_epochs = numberGenerations,
                                                   size_pop = popsize, prob_mut=0.2,
                                                   tr_ts_percents=tr_ts_percents).search_best_model()
            best_mlp_vr_residual = ag_mlp_vr_residual._best_of_all
            pickle.dump(best_mlp_vr_residual, open(SavePath+"ets_vr_residual.pckl", 'wb'))
        else:
            best_mlp_vr_residual = pickle.load(open(SavePath+"ets_vr_residual.pckl", 'rb'))

        best = best_mlp_vr_residual
        erro = y_pos - y_ets
        erro_train_entrada, _, erro_test_entrada, _ = train_test_split_noExog(erro, best[0], tr_ts_percents)
        erro_estimado = np.concatenate((best[-3].VR_predict(erro_train_entrada), best[-3].VR_predict(erro_test_entrada)))
        _, _, X_ass_1_test_in, _ = train_test_split_noExog(y_ets, best[1], tr_ts_percents)
        _, _, X_ass_2_test_in, _ = train_test_split_prev(erro_estimado, best[2], best[3], tr_ts_percents)
        X_in_test = np.concatenate((X_ass_1_test_in, X_ass_2_test_in), axis=1) 
        y_hat = best[-2].VR_predict(X_in_test)

        return y_hat

    def _applySARIMAXAGMLPEnsemble(self, endo_var, exog_var_matrix, SavePath, tr_ts_percents=[80,20],
     popsize=10, numberGenerations=3, useSavedModels = True):

        if not useSavedModels or not os.path.isfile(SavePath+"y_sarimax_pso_aco"):
            p = [0, 1, 2]
            d = [0, 1]
            q = [0, 1, 2, 3]
            sp = [0, 1, 2]
            sd = [0, 1]
            sq = [0, 1, 2, 3]
            s = [24] #como são dados horarios...
            # search Space, exog possibilities comes in the functions.
            searchSpace = [p, d, q, sp, sd, sq, s]

            options_PSO = {'n_particles':5,'n_iterations':3,'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
            options_ACO = {'antNumber':popsize, 'antTours':numberGenerations, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2}
            results_model, y_sarimax_PSO_ACO = sarimax_PSO_ACO_search(endo_var=endo_var, exog_var_matrix=exog_var_matrix,
                                                        searchSpace=copy(searchSpace), 
                                                        options_PSO=options_PSO,
                                                        options_ACO=options_ACO,
                                                        verbose=self._verbose)
            
            pickle.dump(results_model, open(SavePath+"sarimax_pso_aco", 'wb'))
        else:
            y_sarimax_PSO_ACO = pickle.load(open(SavePath+"sarimax_pso_aco", delimiter=';')).predict()
            
        if not useSavedModels or not os.path.isfile(SavePath+"sarimax_mlp_vr_residual.pckl"):            
            ag_mlp_vr_residual = AGMLP_VR_Residual(endo_var, y_sarimax_PSO_ACO,
                                                   num_epochs = numberGenerations,
                                                   size_pop = popsize, prob_mut=0.2,
                                                   tr_ts_percents=tr_ts_percents).search_best_model()
            best_mlp_vr_residual = ag_mlp_vr_residual._best_of_all
            pickle.dump(best_mlp_vr_residual, open(SavePath+"sarimax_vr_residual.pckl", 'wb'))
        else:
            best_mlp_vr_residual = pickle.load(open(SavePath+"sarimax_vr_residual.pckl", 'rb'))

        mape_pso_aco = MAPE(y_sarimax_PSO_ACO, endo_var)
        print("Mape: {0}".format(mape_pso_aco))
        
        best = best_mlp_vr_residual
        erro = endo_var - y_sarimax_PSO_ACO
        erro_train_entrada, _, erro_test_entrada, _ = train_test_split_noExog(erro, best[0], tr_ts_percents)
        erro_estimado = np.concatenate((best[-3].VR_predict(erro_train_entrada), best[-3].VR_predict(erro_test_entrada)))
        _, _, X_ass_1_test_in, _ = train_test_split_noExog(y_sarimax_PSO_ACO, best[1], tr_ts_percents)
        _, _, X_ass_2_test_in, _ = train_test_split_prev(erro_estimado, best[2], best[3], tr_ts_percents)
        X_in_test = np.concatenate((X_ass_1_test_in, X_ass_2_test_in), axis=1) 
        y_hat = best[-2].VR_predict(X_in_test)

        return y_hat

    def _saveResults(self, y_test, y_hats, labels, save_path, timestap_now, metricsThreshHold, customMetric=None):
        logResults = ""
        logResults += "Scores" + "\n"
        print("Scores")

        for y_hat, plotlabel in zip(y_hats, labels):
            print("ploting... " + plotlabel)
            logResults += "{0} ".format(plotlabel) + "- MAE: %.4f" % mean_absolute_error(y_test, y_hat) + "\n"
            logResults += "{0} ".format(plotlabel) + "- MAPE: %.4f" % MAPE(y_test, y_hat, metricsThreshHold) + "\n"
            logResults += "{0} ".format(plotlabel) + "- SMAPE: %.4f" % SMAPE(y_test, y_hat, metricsThreshHold) + "\n"
            logResults += "{0} ".format(plotlabel) + "- MSE: %.4f" % mean_squared_error(y_test, y_hat) + "\n"
            if customMetric != None:
                metric_name = list(customMetric.keys())[0]
                metric_func = customMetric[metric_name]

                print(y_test.shape)
                print(y_hat.shape)
                print(metric_func(y_test, y_hat))
                
                logResults += "{0} ".format(plotlabel) + "- {0}:".format(metric_name) + " %.4f" % metric_func(y_test, y_hat) + "\n"
            

        with open(save_path+"/results_{0}.txt".format(timestap_now), "w") as text_file:
            text_file.write(logResults)

        print(logResults)

        return None

    @classmethod
    def plotResults(self,save_path="./TimeSeriesTester/", title="Time Series Tester Results", transformation=115000,
                    ticksX=None, ticksScapeFrequency=3,
                    labelsMap={"ACOLSTM":"ACO-LSTM", "etsagmlpensemble":"ETS-MLPs", "sarimaxagmlpensemble":"SARIMAX-MLPs"}):
        """
            to change labels, input a mapping dict. Use the last _ separated string as key, like:
            labelsMap = {"ACOLSTM":"ACO-LSTM", "etsagmlpensemble":"ETS-MLPs", "sarimaxagmlpensemble":"SARIMAX-MLPs"}
        """
        plt.set_loglevel('WARNING')
        
        _, ax = plt.subplots(1,1, figsize=(14,7), dpi=300)
        y_test = np.loadtxt(save_path+"y_test")
        y_hats_files= [x for x in os.listdir(save_path) if "y" in x and "test" not in x]
        y_hats = [np.loadtxt(save_path+x) for x in y_hats_files]
        labels = list(map(lambda x: x.split('_')[-1], y_hats_files))
        
        if isinstance(labelsMap,(dict)):
            labels = [labelsMap[l] if l in labelsMap.keys() else l for l in labels]

        if isinstance(ticksX,(list,pd.core.series.Series,pd.core.indexes.base.Index,np.ndarray)):
            ticksX = ticksX[-y_test.shape[0]:]
        else:
            ticksX = np.arange(y_test.shape[0])
        
        ax.plot(ticksX, y_test*transformation, 'k-o', label='Testa Data', linewidth=2.0)

        for y_hat, plotlabel in zip(y_hats, labels):
            print("ploting... " + plotlabel)
            trueScale_yhat = transformation*y_hat
            ax.plot(ticksX, trueScale_yhat, '--o', label=plotlabel.upper())

        plt.xticks(ticksX[::ticksScapeFrequency], rotation=45, ha='right', fontsize=12)
        ax.grid(axis='x')
        ax.legend(fontsize=12, bbox_to_anchor=(1.01,1), loc="upper left")
        ax.set_ylabel('W/m2', fontsize=14)
        ax.set_title(title, fontsize=14)
        plt.show()
        plt.tight_layout()
        plt.savefig(save_path+"results.png", dpi=300)

        return ax

    def executeTests(self, y_data, exog_data=None, autoMlsToExecute="All", train_test_split=[80,20],
                     lags=24, useSavedModels=True, useSavedArrays=True, popsize=10, numberGenerations=5,
                     metricsThreshHold=0.1,
                     save_path="./TimeSeriesTester/", 
                     customMetric=None):

        """
            autoMlsToExecute="All"

            Or insert the automls in a list like autoMlsToExecute=["tpot", "hpsklearn", "agmmff", "acolstm", 
                "acoclstm", "ets", "SARIMAXAGMLPEnsemble", "ETSAGMLPEnsemble"]

            popsize: is utilize in the evolutiary based algorithms

            numberGenerations: is utilize in the evolutiary based algorithms

            metricsThreshHold: applies to MAPE and SMAPE metrics

            customMetric; inset a dict like {"metric name":metric_function}. metric_function should receive (y_test, y_hat) as arguments
        """
        if np.isnan(np.sum(y_data)):
            raise("Main data still has nan")
        
        if isinstance(exog_data,(list,pd.core.series.Series,np.ndarray)):
            X_train, y_train, X_test, y_test = train_test_split_with_Exog(y_data, exog_data, lags, train_test_split)
            for i in range(exog_data.shape[1]):
                ex_var = exog_data[:,i]
                if np.isnan(np.sum(ex_var)):
                    print("exog still has nan in column {0}".format(i))
                    raise ValueError("Exog has nan in column")
        else:
            X_train, y_train, X_test, y_test = train_test_split_noExog(y_data, lags, train_test_split)
            
        timestamp_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        y_hats = []
        labels = []

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if not os.path.isfile(save_path+"./y_test"):
            np.savetxt(save_path+"./y_test", y_test.reshape(-1, 1), delimiter=';')

        if "tpot" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("TPOT Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_TPOT"):
                    y_hat_tpot = self._applyTPOT(X_train, y_train, X_test, y_test, save_path+"/tpotModel_{0}".format(timestamp_now),
                                        popSize=popsize, number_Generations=numberGenerations,
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
                    y_hat_HPSKLEARN = self._applyHPSKLEARN(X_train, y_train, X_test, y_test, save_path+"/HPSKLEARNModel_{0}".format(timestamp_now),
                                                max_evals=100, useSavedModels = useSavedModels)
                    np.savetxt(save_path+"/y_hat_HPSKLEARN", y_hat_HPSKLEARN, delimiter=';')
                else:
                    y_hat_HPSKLEARN = np.loadtxt(save_path+"/y_hat_HPSKLEARN", delimiter=';')

                y_hats.append(y_hat_HPSKLEARN)
                labels.append("HPSKLEARN")
            except Exception:
                traceback.print_exc()
                pass

        # if "autokeras" in autoMlsToExecute or autoMlsToExecute=="All":
        #     try:
        #         print("AUTOKERAS Evaluation...")
        #         if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_AUTOKERAS"):
        #             y_hat_autokeras = self._applyAutoKeras(X_train, y_train, X_test, y_test,
        #                                                   save_path+"/autokerastModel_{0}".format(timestamp_now),
        #                                                   max_trials=10, epochs=300, useSavedModels = useSavedModels)
        #             np.savetxt(save_path+"/y_hat_AUTOKERAS", y_hat_autokeras, delimiter=';')
        #         else:
        #             y_hat_autokeras = np.loadtxt(save_path+"/y_hat_AUTOKERAS", delimiter=';')
                    
        #         y_hats.append(y_hat_autokeras)
        #         labels.append("AUTOKERAS")
        #     except Exception:
        #         traceback.print_exc()
        #         pass

        if "agmmff" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("AGMMFF Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_AGMMFF"):
                    y_hat_agmmff= self._applyGAMMFF(X_train, y_train, X_test, y_test,
                                                   save_path+"/mmffModel_".format(timestamp_now),
                                                   size_pop=popsize, epochs=numberGenerations,
                                                    useSavedModels = useSavedModels)
                    np.savetxt(save_path+"/y_hat_AGMMFF", y_hat_agmmff, delimiter=';')
                else:
                    y_hat_agmmff = np.loadtxt(save_path+"/y_hat_AGMMFF", delimiter=';')

                print("SHAPE HAT {0}".format(y_hat_agmmff.shape))
                y_hats.append(y_hat_agmmff)
                labels.append("AGMMFF")
            except Exception:
                traceback.print_exc()
                pass

        if "SARIMAXAGMLPEnsemble" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("SARIMAXAGMLPEnsemble Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_sarimaxagmlpensemble"):
                    y_hat_sarimaxagmlpensemble = self._applySARIMAXAGMLPEnsemble(y_data, exog_data, SavePath=save_path,
                                                                                tr_ts_percents = train_test_split,
                                                                                popsize = popsize,
                                                                                numberGenerations = numberGenerations,
                                                                                useSavedModels = useSavedModels)
                    np.savetxt(save_path+"/y_hat_sarimaxagmlpensemble", y_hat_sarimaxagmlpensemble, delimiter=';')
                else:
                    y_hat_sarimaxagmlpensemble = np.loadtxt(save_path+"/y_hat_sarimaxagmlpensemble", delimiter=';')

                print("SHAPE HAT {0}".format(y_hat_sarimaxagmlpensemble.shape))
                y_hats.append(y_hat_sarimaxagmlpensemble)
                labels.append("SARIMAXAGMLPEnsemble")
            except Exception:
                traceback.print_exc()
                pass

        ##################################################################################################
        #################################### NO EXOG MODELS ##############################################    
        ##################################################################################################
        X_train_noexog, y_train_noexog, X_test_noexog, y_test_noexog = train_test_split_noExog(y_data, 23,
                                                                            tr_vd_ts_percents = [80, 20],
                                                                            print_shapes = useSavedModels)
        options_ACO={'antNumber':popsize, 'antTours':numberGenerations, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1}

        if "acolstm" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("ACOLSTM Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_ACOLSTM"):
                    Layers_Qtd=[[60, 40, 30], [20, 15], [10, 7, 5]]
                    epochs=[300]        
                    y_hat_acolstm = self._applyACOLSTM(X_train_noexog, y_train_noexog, X_test_noexog, y_test_noexog,
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
                    y_hat_acoclstm = self._applyACOCLSTM(X_train_noexog, y_train_noexog, X_test_noexog, y_test_noexog,
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

        if "ETSAGMLPEnsemble" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("ETSAGMLPEnsemble Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_etsagmlpensemble"):
                    y_hat_etsagmlpensemble = self._applyETSMLPEnsemble(y_data, SavePath = save_path,
                                                                            tr_ts_percents = train_test_split,
                                                                            popsize = popsize,
                                                                            numberGenerations = numberGenerations,
                                                                            useSavedModels = useSavedModels)
                    np.savetxt(save_path+"/y_hat_etsagmlpensemble", y_hat_etsagmlpensemble, delimiter=';')
                else:
                    y_hat_etsagmlpensemble = np.loadtxt(save_path+"/y_hat_etsagmlpensemble", delimiter=';')

                print("SHAPE HAT {0}".format(y_hat_etsagmlpensemble.shape))
                y_hats.append(y_hat_etsagmlpensemble)
                labels.append("ETSAGMLPEnsemble")
            except Exception:
                traceback.print_exc()
                pass

        if "ets" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("ETS Evaluation...")
                if not useSavedArrays or not os.path.isfile(save_path+"/y_hat_ETS"):
                    y_hat_ETS = self._applyETS(y_train_noexog, y_test_noexog)
                    np.savetxt(save_path+"/y_hat_ETS", y_hat_ETS, delimiter=';')
                else:
                    y_hat_ETS = np.loadtxt(save_path+"/y_hat_ETS", delimiter=';')

                print("SHAPE HAT {0}".format(y_hat_ETS.shape))
                y_hats.append(y_hat_ETS)
                labels.append("ETS")
            except Exception:
                traceback.print_exc()
                pass

        self._saveResults(y_test, y_hats, labels, save_path, timestamp_now, metricsThreshHold, customMetric)

    def forecastAheadTest(self, y_data, K=12,  train_test_split=[80,20],
                     lags=24,save_path="./TimeSeriesTester/", autoMlsToExecute="All"):
        """
            In case of forecast ahead, is impossible to make with enable exogenous variables.
        """

        if np.isnan(np.sum(y_data)):
            raise("main data still has nan")
            
        y_hats = {}

        # ##################################################################################################
        # #################################### NO EXOG MODELS ##############################################    
        # ##################################################################################################

        X_train, y_train, X_test, y_test = train_test_split_noExog(y_data, lags, train_test_split)

        # if "SARIMAXAGMLPEnsemble" in autoMlsToExecute or autoMlsToExecute=="All":
        #     try:
        #         print("SARIMAXAGMLPEnsemble Evaluation...")
        #         y_sarimax_PSO_ACO = pickle.load(open(save_path+"sarimax_pso_aco",'rb'))
        #         best_mlp_vr_residual = pickle.load(open(save_path+"sarimax_vr_residual.pckl", 'rb'))
        #         y_hat_sarimaxmlp = AGMLP_VR_Residual(y_data, y_sarimax_PSO_ACO.predict()).forecastAhead(K, y_sarimax_PSO_ACO.forecast(K), bestObject=best_mlp_vr_residual)

        #         y_hats["SARIMAXAGMLPEnsemble"]=y_hat_sarimaxmlp[-K:]
        #     except Exception:
        #         traceback.print_exc()
        #         pass

        if "ETSAGMLPEnsemble" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("ETSAGMLPEnsemble Evaluation...")
                ets_model = pickle.load(open(save_path+"best_ets",'rb'))
                best_mlp_vr_residual = pickle.load(open(save_path+"ets_vr_residual.pckl", 'rb'))
                ets_insample = ets_model.fittedvalues
                ets_outsample = ets_model.forecast(K)
                ets_in_out_samples = np.concatenate((ets_insample, ets_outsample))
                y_hat_etsmlp = AGMLP_VR_Residual(y_data, ets_insample).forecastAhead(K, ets_in_out_samples, bestObject=best_mlp_vr_residual)
                y_hats["ETSAGMLPEnsemble"]=y_hat_etsmlp[-K:]
                y_hats["ETS"]=ets_outsample
            except Exception:
                traceback.print_exc()
                pass
        
        if "agmmff" in autoMlsToExecute or autoMlsToExecute=="All":
            try:
                print("AGMMFF Ahead...")
                final_blender = pickle.load(open(save_path+"mmffModel_blender.pckl", 'rb'))
                final_models = pickle.load(open(save_path+"mmffModel_models.pckl", 'rb'))
                agMMGGBlending = AGMMFFBlending(X_train, y_train, X_test, y_test, verbose=self._verbose)
                y_hat_agmmff_inicial = agMMGGBlending.predict(X=X_test, models=final_models, blender=final_blender)
                y_hat_agmmf = y_hat_agmmff_inicial.copy()
                for k in range(K):
                    X_loop = y_hat_agmmf[-lags:]
                    y_hat_agmmf = np.concatenate((y_hat_agmmf,
                                                  agMMGGBlending.predict(X=X_loop.reshape(1, -1), blender=final_blender, models=final_models)))

                y_hats["AGMMFF"]=y_hat_agmmf[-K:]
            except Exception:
                traceback.print_exc()
                pass
        
        # X_train_noexog, y_train_noexog, X_test_noexog, y_test_noexog = train_test_split_noExog(y_data, 23,
        #                                                                     tr_vd_ts_percents = [80, 20],
        #                                                                     print_shapes = useSavedModels)
        # options_ACO={'antNumber':popsize, 'antTours':numberGenerations, 'alpha':1, 'beta':1, 'rho':0.5, 'Q':1}

        # if "acolstm" in autoMlsToExecute or autoMlsToExecute=="All":
        #     try:
        #         print("ACOLSTM Evaluation...")
        #         Layers_Qtd=[[60, 40, 30], [20, 15], [10, 7, 5]]
        #         epochs=[300]        
        #         y_hat_acolstm = self._applyACOLSTM(X_train_noexog, y_train_noexog, X_test_noexog, y_test_noexog,
        #                                     save_path+"/acolstmModel_{0}".format(timestamp_now),
        #                                     Layers_Qtd, epochs, options_ACO, useSavedModels = useSavedModels)

        #         y_hats.append(y_hat_acolstm)
        #         labels.append("ACOLSTM")
        #     except Exception:
        #         traceback.print_exc()
        #         pass

        # if "ets" in autoMlsToExecute or autoMlsToExecute=="All":
        #     try:
        #         print("ETS Evaluation...")
        #         y_hat_ETS = self._applyETS(y_train_noexog, y_test_noexog)

        #         print("SHAPE HAT {0}".format(y_hat_ETS.shape))
        #         y_hats.append(y_hat_ETS)
        #         labels.append("ETS")
        #     except Exception:
        #         traceback.print_exc()
        #         pass

        return y_hats