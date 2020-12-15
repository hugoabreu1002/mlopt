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

def sarimax_serial_search(endo, exog, search=False, param_default = (0, 1, 1), param_seasonal_default=(0,0,0,12)):
    
    if search:
        p = d = q = range(0, 2)
        s = [6,12,24,48]
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))

        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]

        warnings.filterwarnings("ignore") # specify to ignore warning messages

        best_model = None
        best_AIC = np.inf 

        for param in pdq:
            if any(param) !=0:
                for param_seasonal in seasonal_pdq:
                    try:
                        mod = SARIMAX(endo, exog=exog, order=param, seasonal_order=param_seasonal,
                                      enforce_stationarity=False, enforce_invertibility=False)

                        results = mod.fit(disp=False)
                        print('ARIMA{}, x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                        if results.aic < best_AIC:
                            best_AIC = results.aic
                            best_model = results.predict()
                            print('BEST: ', best_AIC, param, param_seasonal)
                    except:
                        continue
    
    else:
        mod = SARIMAX(endo, exog=exog, order=param_default, seasonal_order=param_seasonal_default,
                      enforce_stationarity=False,enforce_invertibility=False)
        
        results = mod.fit(disp=True)
        print('ARIMA{}, x{} - AIC:{}'.format(param_default, param_seasonal_default, results.aic))
        best_model = results.predict()

    return best_model

def SARIMAX_aic(x, *args):
    endo = args[0][0]
    exog = args[0][1]
    param = x[0:3]
    param_seasonal = x[3:] 
    mod = SARIMAX(endo, exog=exog, order=param, seasonal_order=param_seasonal,
                                enforce_stationarity=False, enforce_invertibility=False)

    results = mod.fit(disp=False)
        
    return results.aic

def SARIMAX_MAPE(XX, **kwargs):
    endo = kwargs['endo']
    exog = kwargs['exog']
    verbose = kwargs['verbose']
    
    return_matrix = np.zeros(XX.shape)
    
    for x in range(XX.shape[0]):
        param = x[0:3]
        param_seasonal = x[3:] 
        IntBinPos = int(x[-1])
        listPosb = convertInt2ListBinaryPossibilites(IntBinPos)
        
        if len(listPosb) > 0:
            true_exog = exog[:, listPosb]
        else:
            true_exog = None
            
        mod = SARIMAX(endo, exog=true_exog, order=param, seasonal_order=param_seasonal,
                                    enforce_stationarity=False, enforce_invertibility=False)

        results = mod.fit(disp=False)
        y_sarimax = results.predict()
        return_matrix[x] = MAPE(endo, y_sarimax)
        
    return return_matrix

def sarimax_ACO_search(antNumber, antTours, alpha, beta, rho, Q, searchSpace, endo, exog, verbose=False):
    
    X = searchSpace

    warnings.filterwarnings("ignore") # specify to ignore warning messages
    
    ACOsearch = ACO(alpha, beta, rho, Q)

    best_result, _ = ACOsearch.optimize(antNumber, antTours, dimentionsRanges=X, function=SARIMAX_aic,
                                        functionArgs=[endo, exog],  verbose=verbose)
    
    param = best_result[0:3]
    param_seasonal = best_result[3:] 
    mod = SARIMAX(endo, exog=exog, order=param, seasonal_order=param_seasonal,
                                  enforce_stationarity=False, enforce_invertibility=False)

    results = mod.fit(disp=False)

    return results.predict

def sarimax_Exog_PSO_search(pso_particles, pso_interations, searchSpace, exog_variables_matrix, options_PSO):
    """
        p = d = q = range(0, 2)
        sp = sd = sq = range(0, 2)
        s = [12,24,48] 
        qt_exog_variables = 4
        searchSpace = [p, d, q, sp, sd, sq, s, qt_exog_variables]
    """
    exog_possibilities_qt = exog_variables_matrix.shape[1]**2 

    # Call instance of LBestPSO with a neighbour-size of 3 determined by
    # the L2 (p=2) distance.
    optimizer = ps.single.LocalBestPSO(n_particles=pso_particles, dimensions=1, bounds=([0,0,0,0,0,0,0], searchSpace),
                                       options=options_PSO)

    # Perform optimization
    stats = optimizer.optimize(SARIMAX_MAPE, iters=pso_interations)


def convertInt2ListBinaryPossibilites(number):
    """
    exemple:
    number = 10
    maxLen = 4
    
    0 - [0, 0, 0, 0]
    1 - [0, 0, 0, 1]
    2 - [0, 0, 1, 0]
    3 - [0, 0, 1, 1]
    ...
    
    10 - [1, 0, 1, 0]
    
    returns [3, 1]
    """
    def bitfield(n):
        return np.array([1 if digit=='1' else 0 for digit in bin(n)[2:]])
    
    arrayBit = bitfield(number) 
    maxLen = arrayBit.shape[0]
    
    return np.arange(maxLen)[arrayBit == 1]

def sarimax_Exog_PSO_Wrapper_search(pso_particles, pso_interations, exog_variables_matrix, options_PSO, options_ACO):
    
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
            
            listPosb = convertInt2ListBinaryPossibilites(int(x))
            
            if len(listPosb) > 0:
                true_exog = exog[:, listPosb]
            else:
                true_exog = None
            
            y_sarimax = sarimax_ACO_search(antNumber=antNumber, antTours=antTours, alpha=alpha, beta=beta, rho=rho,
                                           Q=Q, searchSpace=searchSpace, endo=endo, exog=true_exog, verbose=verbose)
            
            print(endo, y_sarimax)
            return_matrix[x] = MAPE(endo, y_sarimax)
            
        return return_matrix
    
    exog_possibilities_qt = exog_variables_matrix.shape[1]**2 

    # Call instance of LBestPSO with a neighbour-size of 3 determined by
    # the L2 (p=2) distance.
    optimizer = ps.single.LocalBestPSO(n_particles=pso_particles, dimensions=1, bounds=([0], [exog_possibilities_qt]),
                                       options=options_PSO)

    # Perform optimization
    stats = optimizer.optimize(SARIMAX_ACO_search_MAPE, iters=pso_interations, **options_ACO)