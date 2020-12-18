from ML_Algorithms.ACO import ACO
import pyswarms as ps
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings
import itertools

def MAPE(y_pred, y_true): 
    mask = y_true != 0
    return (np.fabs(y_true - y_pred)/y_true)[mask].mean()

def train_test_split(serie, num_lags, tr_vd_ts_percents = [80, 20], print_shapes = False):
    len_serie = len(serie)
    X = np.zeros((len_serie, num_lags))
    y = np.zeros((len_serie,1))
    for i in np.arange(0, len_serie):
        if i-num_lags>0:
            X[i,:] = serie[i-num_lags:i]
            y[i] = serie[i]
    
    len_train = np.floor(len_serie*tr_vd_ts_percents[0]/100).astype('int')
    len_test = np.ceil(len_serie*tr_vd_ts_percents[1]/100).astype('int')
    
    X_train = X[0:len_train]
    y_train = y[0:len_train]
    X_test = X[len_train:len_train+len_test]
    y_test = y[len_train:len_train+len_test]
       
    return X_train, y_train, X_test, y_test

def train_test_split_prev(serie, num_lags_pass, num_lags_fut, tr_vd_ts_percents = [80, 20], print_shapes = False):
    #alterar para deixar com passado e futuro.
    len_serie = len(serie)
    X = np.zeros((len_serie, (num_lags_pass+num_lags_fut)))
    y = np.zeros((len_serie,1))
    for i in np.arange(0, len_serie):
        if (i-num_lags_pass > 0) and ((i+num_lags_fut) <= len_serie):
            X[i,:] = serie[i-num_lags_pass:i+num_lags_fut]
            y[i] = serie[i]
        elif (i-num_lags_pass > 0) and ((i+num_lags_fut) > len_serie):
            X[i,-num_lags_pass:] = serie[i-num_lags_pass:i]
            y[i] = serie[i]
    
    len_train = np.floor(len_serie*tr_vd_ts_percents[0]/100).astype('int')
    len_test = np.ceil(len_serie*tr_vd_ts_percents[1]/100).astype('int')
    
    X_train = X[0:len_train]
    y_train = serie[0:len_train]
    X_test = X[len_train:len_train+len_test]
    y_test = y[len_train:len_train+len_test]
    
    return X_train, y_train, X_test, y_test

def convertInt2BinaryList(number):
    return [int(x) for x in bin(number)[2:]]

def sarimax_serial_search(endo, exog_var_matrix, search=False, search_exog=False, pdq_ranges=[0,1,2], s_possibilities=[6,12,24,48],
                          param_default = (0, 1, 1), param_seasonal_default=(0,0,0,12)):
    
    if search:
        p = d = q = pdq_ranges
        s = s_possibilities
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        
        if search_exog:
            exogs_possibilites = list(map(lambda L: convertInt2BinaryList(2**L), range(exog_var_matrix.shape[1])))
        else:
            exogs_possibilites = np.ones(exog_var_matrix.shape[1], 1)
            
        print(exogs_possibilites)

        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]

        warnings.filterwarnings("ignore") # specify to ignore warning messages

        best_model = None
        best_AIC = np.inf 

        for param in pdq:
            if any(param) !=0:
                for param_seasonal in seasonal_pdq:
                    for exog_chosen in exogs_possibilites:
                        exog_true = exog_var_matrix[:, exog_chosen]
                        try:
                            mod = SARIMAX(endo, exog=exog_true, order=param, seasonal_order=param_seasonal,
                                        enforce_stationarity=False, enforce_invertibility=False)

                            results = mod.fit(disp=False)
                            print('ARIMA {0}, S {1}, Exog {2} - AIC:{3}'.format(param, param_seasonal, exog_chosen, results.aic))
                            if results.aic < best_AIC:
                                best_AIC = results.aic
                                best_model = results.predict()
                                print('BEST: ', best_AIC, param, param_seasonal)
                        except:
                            continue
    
    else:
        mod = SARIMAX(endo, exog=exog_var_matrix, order=param_default, seasonal_order=param_seasonal_default,
                      enforce_stationarity=False,enforce_invertibility=False)
        
        results = mod.fit(disp=True)
        print('ARIMA{}, x{} - AIC:{}'.format(param_default, param_seasonal_default, results.aic))
        best_model = results.predict()

    return best_model

def sarimax_ACO_search(endo_var, exog_var_matrix, antNumber, antTours, alpha, beta, rho, Q, searchSpace, verbose=False):
    
    def SARIMAX_aic(X, *args):
        endo = args[0][0]
        exog = args[0][1]
        param = X[0:3]
        param_seasonal = X[3:7]
        if param_seasonal[-1] < 0:
            param_seasonal[-1] = 1
        IntBinPos = int(X[-1])
        listPosb = convertInt2BinaryList(IntBinPos)
        if len(listPosb) > 0:
            true_exog = exog[:, listPosb]
        else:
            true_exog = None
        mod = SARIMAX(endo, exog=true_exog, order=param, seasonal_order=param_seasonal,
                                    enforce_stationarity=False, enforce_invertibility=False)

        results = mod.fit(disp=False)
            
        return results.aic
    
    X = searchSpace
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ACOsearch = ACO(alpha, beta, rho, Q)

    best_result, _ = ACOsearch.optimize(antNumber, antTours, dimentionsRanges=X, function=SARIMAX_aic,
                                        functionArgs=[endo_var, exog_var_matrix],  verbose=verbose)
    
    param = best_result[0:3]
    param_seasonal = best_result[3:]
    IntBinPos = int(best_result[-1])
    listPosb = convertInt2BinaryList(IntBinPos)
    if len(listPosb) > 0:
        true_exog = exog_var_matrix[:, listPosb]
    else:
        true_exog = None 
        
    mod = SARIMAX(endo_var, exog=true_exog, order=param, seasonal_order=param_seasonal,
                                  enforce_stationarity=False, enforce_invertibility=False)

    results = mod.fit(disp=False)

    return results.predict()

def sarimax_PSO_search(endo_var, exog_var_matrix, searchSpace, pso_particles, pso_iterations, options_PSO):
    """
        endo_var - is the principal variable
        exog_var_matrix - are a matrix of exogenous variables
        searchSpace - is the space of search for the particles. EG:
            p = d = q = range(0, 2)
            sp = sd = sq = range(0, 2)
            s = [12,24,48] 
            qt_exog_variables = 4
            searchSpace = [p, d, q, sp, sd, sq, s, qt_exog_variables]
        pso_particles - is the number of particles
        pso_interations - is the number of interations
        options_PSO - are the options for pyswarm.single.LocalBestPSO object. EG:
            options_ACO = {'antNumber':2, 'antTours':1, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2,'searchSpace':X,
                            'endo':gen,'exog':exog, 'verbose':False}
    """
    def SARIMAX_aic_matrix(XX, **kwargs):
        endo = kwargs['endo']
        exog = kwargs['exog']
        return_matrix = np.zeros(XX.shape[0])
        for Index, X  in enumerate(XX):
            param = X[0:3].astype('int')
            param_seasonal = X[3:7].astype('int') 
            if param_seasonal[-1] < 0:
                param_seasonal[-1] = 1
                
            IntBinPos = int(X[-1])
            listPosb = convertInt2BinaryList(IntBinPos)
            
            if len(listPosb) > 0:
                true_exog = exog[:, listPosb]
            else:
                true_exog = None
                
            mod = SARIMAX(endo, exog=true_exog, order=param, seasonal_order=param_seasonal,
                                        enforce_stationarity=False, enforce_invertibility=False)

            results = mod.fit(disp=False)
            return_matrix[Index] = results.aic
            
        return return_matrix

    optimizer = ps.single.LocalBestPSO(n_particles=pso_particles, dimensions=len(searchSpace), bounds=([0,0,0,0,0,0,0], searchSpace),
                                       options=options_PSO)

    # Perform optimization
    kwargs_pso = {'endo':endo_var, 'exog':exog_var_matrix}
    stats = optimizer.optimize(SARIMAX_aic_matrix, iters=pso_iterations, **kwargs_pso)
    
    # return predicted array
    best_result = stats[1]
    param = best_result[0:3].astype('int')
    param_seasonal = best_result[3:7].astype('int')
    if param_seasonal[-1] < 0:
        param_seasonal[-1] = 1
    IntBinPos = int(best_result[-1])
    listPosb = convertInt2BinaryList(IntBinPos)
    
    if len(listPosb) > 0:
        true_exog = exog_var_matrix[:, listPosb]
    else:
        true_exog = None 
    
    mod = SARIMAX(endo_var, exog=true_exog, order=param, seasonal_order=param_seasonal,
                                  enforce_stationarity=False, enforce_invertibility=False)

    results = mod.fit(disp=False)

    return results.predict()

def sarimax_PSO_ACO_search(pso_particles, pso_interations, endo_var, exog_var_matrix, options_PSO, options_ACO):
    # TODO: test again, divide the work between ACO and PSO
    
    def SARIMAX_ACO_search_MAPE(XX, **kwargs):
        antNumber = kwargs['antNumber']
        antTours = kwargs['antTours']
        alpha = kwargs['alpha']
        beta = kwargs['beta']
        rho = kwargs['rho']
        Q = kwargs['Q']
        searchSpace = kwargs['searchSpace']
        endo = kwargs['endo']
        exog = kwargs['exog']
        verbose = kwargs['verbose']
        
        return_matrix = np.zeros(XX.shape)
        
        for x in range(XX.shape[0]):
            
            listPosb = convertInt2BinaryList(int(x))
            
            if len(listPosb) > 0:
                true_exog = exog[:, listPosb]
            else:
                true_exog = None
            
            y_sarimax = sarimax_ACO_search(endo_var=endo, exog_var_matrix=true_exog, antNumber=antNumber, antTours=antTours, alpha=alpha, beta=beta, rho=rho,
                                           Q=Q, searchSpace=searchSpace,  verbose=verbose)
            
            print(endo, y_sarimax)
            return_matrix[x] = MAPE(endo, y_sarimax)
            
        return return_matrix
    
    exog_possibilities_qt = exog_var_matrix.shape[1]
    
    # Call instance of LBestPSO with a neighbour-size of 3 determined by
    # the L2 (p=2) distance.
    optimizer = ps.single.LocalBestPSO(n_particles=pso_particles, dimensions=1, bounds=([0], [exog_possibilities_qt]),
                                       options=options_PSO)

    # Perform optimization
    stats = optimizer.optimize(SARIMAX_ACO_search_MAPE, iters=pso_interations, **options_ACO)
    
    best_result = stats[1]
    param = best_result[0:3]
    param_seasonal = best_result[3:]
    IntBinPos = int(best_result[-1])
    listPosb = convertInt2BinaryList(IntBinPos)
    if len(listPosb) > 0:
        true_exog = exog_var_matrix[:, listPosb]
    else:
        true_exog = None 
    mod = SARIMAX(endo_var, exog=true_exog, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

    results = mod.fit(disp=False)

    return results.predict()