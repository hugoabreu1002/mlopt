from mlopt.ACO import ACO
import pyswarms as ps
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings
import itertools
import copy
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

consoleInfo = logging.StreamHandler()
consoleInfo.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s: %(levelname)s - %(message)s")
consoleInfo.setFormatter(formatter)
logging.getLogger('').addHandler(consoleInfo)
rotatingHandler = RotatingFileHandler(filename='timeSeriesUtils.log', encoding=None, mode='a', maxBytes=5*1024*1024, 
                                 backupCount=2, delay=0)
rotatingHandler.setLevel(logging.INFO)
logging.getLogger('').addHandler(rotatingHandler)

def MAPE(y_pred, y_true): 
    mask = y_true != 0
    return (np.fabs(y_true - y_pred)/y_true)[mask].mean()

def train_test_split(serie, num_lags, tr_vd_ts_percents = [80, 20], print_shapes = False):
    """
        Slipts a time series to train and test Data.
        X data are data num_lags behind y data.
        
        serie : is the time serie data
        
        num_lags : quantity of data behind y data
        
        tr_vd_ts_percents : divistion percentages
        
        print_shapes : True chose to print final shapes. Default is False.
    """
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
    """
        Slipts a time series to train and test Data.
        X data are data num_lags_pass behind and num_lags_fut ahead y data.
        
        serie : is the time serie data
        
        num_lags_pass : quantity of data behind y data
        
        num_lags_fut : quantity of data ahead y data
        
        tr_vd_ts_percents : divistion percentages
        
        print_shapes : True chose to print final shapes. Default is False.
    """
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
    binaryList = [int(x) for x in bin(number)[2:]]
    binaryList.reverse()
    return binaryList

def convertInt2PosList(number):
    binaryList = convertInt2BinaryList(number)
    returnList=[]
    for i in range(0,len(binaryList)):
        yesOrNo = binaryList[i]
        if yesOrNo == 1:
            listNumber = i*yesOrNo
            returnList.append(listNumber)

    return returnList

def sarimax_serial_search(endo, exog_var_matrix, search=False, search_exog=False, pdq_ranges=[0,1,2], s_possibilities=[6,12,24,48],
                          param_default = (0, 1, 1), param_seasonal_default=(0,0,0,12)):
    """
        endo_var: is the principal variable.
        
        exog_var_matrix: are a matrix of exogenous variables.
        
        search: True if want to make the search
            
        search_exog: True if want to search an arrange of exogenous matrix possibilities
        
        pdq_ranges: ranges to search for pdq and seasonal pdq parameters. E.G: pdq_ranges=[0,1,2].
        
        s_possibilites: list of S parameter possibilites. E.G: s_possibilities=[6,12,24,48].
        
        param_default: default pdq parameters tuple.
        
        param_seasonal_default: default seasonal parameters tuple.
    """
    if search:
        p = d = q = pdq_ranges
        s = s_possibilities
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        
        if search_exog:
            exogs_possibilites = list(map(lambda L: convertInt2PosList(2**L), range(exog_var_matrix.shape[1])))
        else:
            exogs_possibilites = np.ones(exog_var_matrix.shape[1], 1)
            
        logging.info(exogs_possibilites)

        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]

        warnings.filterwarnings("ignore") # specify to ignore warning messages

        best_model = None
        best_AICc = np.inf 

        for param in pdq:
            if any(param) !=0:
                for param_seasonal in seasonal_pdq:
                    for exog_chosen in exogs_possibilites:
                        exog_true = exog_var_matrix[:, exog_chosen]
                        try:
                            mod = SARIMAX(endo, exog=exog_true, order=param, seasonal_order=param_seasonal,
                                        enforce_stationarity=False, enforce_invertibility=False)

                            results = mod.fit(disp=False)
                            logging.info("ARIMA {0}, S {1}, Exog {2} - AICc:{3}".format(param, param_seasonal, exog_chosen, results.aicc))
                            if results.aicc < best_AICc:
                                best_AICc = results.aicc
                                best_model = results.predict()
                                logging.info("BEST - AICc: {0} param: {1} param_seasonal: {2}".format(best_AICc, param, param_seasonal))
                        except:
                            continue
    
    else:
        mod = SARIMAX(endo, exog=exog_var_matrix, order=param_default, seasonal_order=param_seasonal_default,
                      enforce_stationarity=False,enforce_invertibility=False)
        
        results = mod.fit(disp=True)
        logging.info('ARIMA{}, x{} - AICc:{}'.format(param_default, param_seasonal_default, results.aicc))
        best_model = results.predict()

    return best_model

def sarimax_ACO_search(endo_var, exog_var_matrix, searchSpace, options_ACO, verbose=False):
    """
        endo_var: is the principal variable.
        
        exog_var_matrix: are a matrix of exogenous variables.
        
        searchSpace: is the space of search for the particles. EG:
            p = d = q = range(0, 2)
            sp = sd = sq = range(0, 2)
            s = [12,24,48] 
            qt_exog_variables = 4
            searchSpace = [p, d, q, sp, sd, sq, s, range(2**qt_exog_variables)]
            
        pso_particles: is the number of particles.
        
        pso_interations: is the number of interations.
        
        options_ACO: parametrization for ACO algorithm. EG:
            {'antNumber':2, 'antTours':1, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2}
    """
    def SARIMAX_aicc(X, *args):
        endo = args[0][0]
        exog = args[0][1]
        param = X[0:3]
        param_seasonal = X[3:7]
        if param_seasonal[-1] < 0:
            param_seasonal[-1] = 1
        IntBinPos = int(X[-1])
        listPosb = convertInt2PosList(IntBinPos)
        if len(listPosb) > 0:
            true_exog = exog[:, listPosb]
        else:
            true_exog = None
        mod = SARIMAX(endo, exog=true_exog, order=param, seasonal_order=param_seasonal,
                                    enforce_stationarity=False, enforce_invertibility=False)

        aicc = np.inf
        try:
            results = mod.fit(disp=False)
            aicc = results.aicc
        except:
            pass

        return aicc
    
    antNumber = options_ACO['antNumber']
    antTours = options_ACO['antTours']
    alpha = options_ACO['alpha']
    beta = options_ACO['beta']
    rho = options_ACO['rho']
    Q = options_ACO['Q']
    
    logging.info("Original search Space: {0}".format(searchSpace))
    exogs_possibilites = range(0,2**exog_var_matrix.shape[1]) 
    searchSpace.append(exogs_possibilites)
    logging.info("search Space with Exog Possibilities: {0}".format(searchSpace))
    
    X = searchSpace
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ACOsearch = ACO(alpha, beta, rho, Q)

    best_result, _ = ACOsearch.optimize(antNumber, antTours, dimentionsRanges=X, function=SARIMAX_aicc,
                                        functionArgs=[endo_var, exog_var_matrix],  verbose=verbose)
   
    param = best_result[0:3]
    param_seasonal = best_result[3:7]
    IntBinPos = int(best_result[-1])
    listPosb = convertInt2PosList(IntBinPos)
    if len(listPosb) > 0:
        true_exog = exog_var_matrix[:, listPosb]
    else:
        true_exog = None 
    
    mod = SARIMAX(endo_var, exog=true_exog, order=param, seasonal_order=param_seasonal,
                                  enforce_stationarity=False, enforce_invertibility=False)

    results = mod.fit(disp=False)
    logging.info("BEST result: \n PDQ-parameters: {0} \n PDQS-parameters: {1} \n Exgoneous Var: {2}".format(param, param_seasonal, listPosb))
    
    return results.predict()

def sarimax_PSO_search(endo_var, exog_var_matrix, searchSpace, options_PSO, verbose=False):
    """
        endo_var: is the principal variable.
        
        exog_var_matrix: are a matrix of exogenous variables.
        
        searchSpace: is the space of search for the particles. EG:
            p = d = q = range(0, 2)
            sp = sd = sq = range(0, 2)
            s = [12,24,48] 
            qt_exog_variables = 4
            searchSpace = [p, d, q, sp, sd, sq, s], exog possibilities are appended after
        
        options_PSO: are the options for pyswarm.single.LocalBestPSO object. EG:
            options_PSO = {'n_particles':10,'n_iterations':100,'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
    """
    def SARIMAX_AICc_matrix(XX, **kwargs):
        endo = kwargs['endo']
        exog = kwargs['exog']
        S_parameter_posb = kwargs['S_parameter_posb']
        return_matrix = np.zeros(XX.shape[0])
        
        for Index, X  in enumerate(XX):
            param = X[0:3].astype('int')
            param_seasonal = X[3:7].astype('int') 
            param_seasonal[-1] = S_parameter_posb[param_seasonal[-1]]
            
            IntBinPos = int(X[-1])
            listPosb = convertInt2PosList(IntBinPos)
            if len(listPosb) > 0:
                true_exog = exog[:, listPosb]
            else:
                true_exog = None
                
            mod = SARIMAX(endo, exog=true_exog, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            aicc = np.inf
            try:
                results = mod.fit(disp=False)
                aicc = results.aicc
            except:
                pass
                
            return_matrix[Index] = aicc
            
        return return_matrix

    # changes in S possibilites
    S_parameter_posb = copy.copy(searchSpace[-1])
    qt_s_posb = len(S_parameter_posb)
    searchSpace[-1] = [0, qt_s_posb]
    
    # changes for exog possibilities
    logging.info("Original search Space:", searchSpace)
    exogs_possibilites = range(0,2**exog_var_matrix.shape[1]) 
    searchSpace.append(exogs_possibilites)
    logging.info("search Space with Exog Possibilities: {0}".format(searchSpace))
    
    searchSpacePSO = list(map(lambda L: max(L), searchSpace))
    min_boudaries = np.zeros(len(searchSpacePSO))
    logging.info("PSO boundaries: {0} {1}".format(min_boudaries, searchSpacePSO))
    
    rows = 1
    for d in searchSpacePSO:
        rows = d*rows

    logging.info("number of Space Possibilities (rows): {0}".format(rows))
    
    options_PSO_GB = {'c1': options_PSO['c1'], 'c2': options_PSO['c1'],
                      'w': options_PSO['w'], 'k': options_PSO['k'], 'p': options_PSO['p']}
    optimizer = ps.global_best.GlobalBestPSO(n_particles=options_PSO['n_particles'], dimensions=len(searchSpacePSO),
                                             bounds=(min_boudaries, searchSpacePSO), options=options_PSO_GB)

    # Perform optimization
    kwargs_pso = {'endo':endo_var, 'exog':exog_var_matrix, 'S_parameter_posb':S_parameter_posb}
    stats = optimizer.optimize(SARIMAX_AICc_matrix, iters=options_PSO['n_iterations'], verbose=verbose, **kwargs_pso)
    
    # return predicted array
    best_result = stats[1]
    param = best_result[0:3].astype('int')
    param_seasonal = best_result[3:7].astype('int')
    param_seasonal[-1] = S_parameter_posb[param_seasonal[-1]]
    if param_seasonal[-1] < 0:
        param_seasonal[-1] = 1
    IntBinPos = int(best_result[-1])
    listPosb = convertInt2PosList(IntBinPos)
    
    if len(listPosb) > 0:
        true_exog = exog_var_matrix[:, listPosb]
    else:
        true_exog = None 
    
    mod = SARIMAX(endo_var, exog=true_exog, order=param, seasonal_order=param_seasonal,
                                  enforce_stationarity=False, enforce_invertibility=False)

    results = mod.fit(disp=False)
    logging.info("BEST result {0}: \n PDQ-parameters: {1} \n SPDQ-parameters: {2} \n Exgoneous Var: {3}".format(
        best_result,param, param_seasonal, listPosb))

    return results.predict()

def sarimax_ACO_PDQ_search(endo_var, exog_var_matrix, PDQS, searchSpace, options_ACO, verbose=False):
    """
        Searchs SARIMAX PDQ parameters.
        
        endo_var: is the principal variable.
        
        exog_var_matrix: is the matrix of exogenous variables.
        
        PDQS: list of pdqs parameters. EG: [1, 1, 1, 24].
        
        searchSpace: is the space of search for the particles. E.G.:
            p = d = q = range(0, 2)
            searchSpace = [p, d, q]
            
        pso_particles: is the number of particles.
        
        pso_interations: is the number of interations.
        
        options_ACO: parametrization for ACO algorithm. E.G.:
            {'antNumber':2, 'antTours':1, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2}
    """
    def SARIMAX_AICc(X, *args):
        endo = args[0][0]
        exog = args[0][1]
        param_seasonal = args[0][2]
        param = X[0:3]
        if param_seasonal[-1] < 0:
            param_seasonal[-1] = 1

          
        mod = SARIMAX(endo, exog=exog, order=param, seasonal_order=param_seasonal,
                                    enforce_stationarity=False, enforce_invertibility=False)
        aicc = np.inf
        try:  
            results = mod.fit(disp=False)
            aicc = results.aicc
        except:
            pass

        return aicc
    
    antNumber = options_ACO['antNumber']
    antTours = options_ACO['antTours']
    alpha = options_ACO['alpha']
    beta = options_ACO['beta']
    rho = options_ACO['rho']
    Q = options_ACO['Q']
    
    if verbose:    
        logging.info("Original search Space: {0}".format(searchSpace))

    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ACOsearch = ACO(alpha, beta, rho, Q)
    best_result, _ = ACOsearch.optimize(antNumber, antTours, dimentionsRanges=searchSpace, function=SARIMAX_AICc,
                                        functionArgs=[endo_var, exog_var_matrix, PDQS],  verbose=verbose)
    
    logging.info("BEST result: {0}.".format(best_result))
    param = best_result
    param_seasonal = PDQS
    mod = SARIMAX(endo_var, exog=exog_var_matrix, order=param, seasonal_order=param_seasonal,
                                  enforce_stationarity=False, enforce_invertibility=False)

    results = mod.fit(disp=False)

    return results.predict(), best_result

def sarimax_PSO_ACO_search(endo_var, exog_var_matrix, searchSpace, options_PSO, options_ACO, verbose=False):
    """ 
        PCO - ACO Sariamx Search.
        It divides the tasks in two. PDQ Search is done by ACO. PDQS Search and Exogenous Variables searches is
        done by PSO.
        
        endo_var: is the principal variable.
        
        exog_var_matrix: is the matrix of exogenous variables.
        
        searchSpace: is the space of search for the particles and ants. E.G.:
            p = d = q = [0, 1] #range(0, 2)
            sp = sd = sq = [0, 1] #range(0, 2)
            s = [12,24] #como são dados horarios...
            searchSpace = [p, d, q, sp, sd, sq, s]
        
        options_PSO: is the options for all PSO parametrization. E.G.:
            options_PSO = {'n_particles':10,'n_iterations':100,'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
        
        options_ACO: parametrization for ACO algorithm. E.G.:
            {'antNumber':2, 'antTours':1, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2}
    """

    search_results = []
    def sarimax_ACO_PDQ_search_MAPE(XX, **Allkwargs):
        kwargs = Allkwargs['kwargs']
        S_parameter_posb = kwargs['S_parameter_posb']
        searchSpaceACO = kwargs['searchSpaceACO']
        endo = kwargs['endo']
        exog = kwargs['exog']
        verbose = kwargs['verbose']
        options_ACO = Allkwargs['options_ACO']
        
        return_matrix = np.zeros(XX.shape[0])
        for Index, X  in enumerate(XX):
            pdqs = X[0:4].astype('int')
            pdqs[-1] = S_parameter_posb[pdqs[-1]]
            exogenous_int_pos = int(X[-1])
            listPosb = convertInt2PosList(exogenous_int_pos)
            if len(listPosb) > 0:
                true_exog = exog[:, listPosb]
            else:
                true_exog = None
            logging.info("ACO Search will start with PDQS: {0} and Exogenous Columns {1}".format(pdqs, listPosb))
            y_sarimax, pdq_param = sarimax_ACO_PDQ_search(endo_var=endo, exog_var_matrix=true_exog,
                                               PDQS=pdqs, searchSpace=searchSpaceACO,
                                               options_ACO=options_ACO,  verbose=verbose)
                       
            mape_result = MAPE(endo, y_sarimax)
            return_matrix[Index] = mape_result
            
            search_results.append((mape_result, pdq_param, pdqs, exogenous_int_pos))
            
        return return_matrix
    
    # changes in S possibilites
    S_parameter_posb = copy.copy(searchSpace[-1])
    qt_s_posb = len(S_parameter_posb)
    searchSpace[-1] = [0, qt_s_posb]
    
    searchSpace = copy.copy(searchSpace)
    logging.info("Original search Space: {0}".format(searchSpace))
    exogs_possibilites = range(0,2**exog_var_matrix.shape[1]) 
    searchSpace.append(exogs_possibilites)
    logging.info("search Space with Exog Possibilities: {0}".format(searchSpace))
    
    searchSpacePSO = searchSpace[3:]
    searchSpacePSO = list(map(lambda L: max(L), searchSpacePSO))
    min_boudaries = np.zeros(len(searchSpacePSO))
    logging.info("PSO boundaries: {0} {1}".format(min_boudaries, searchSpacePSO))
    
    # Call instance of LBestPSO with a neighbour-size of 3 determined by
    # the L2 (p=2) distance.
    options_PSO_GB = {'c1': options_PSO['c1'], 'c2': options_PSO['c1'],
                      'w': options_PSO['w'], 'k': options_PSO['k'], 'p': options_PSO['p']}
    
    dimensions = len(searchSpacePSO)
    logging.info(dimensions)
    optimizer = ps.global_best.GlobalBestPSO(n_particles=options_PSO['n_particles'], dimensions=dimensions,
                                             bounds=(min_boudaries, searchSpacePSO), options=options_PSO_GB)

    # Perform optimization
    searchSpaceACO = searchSpace[:3]
    AllKwargs = {'kwargs': {'searchSpaceACO':searchSpaceACO, 'endo':endo_var, 'exog':exog_var_matrix,'verbose':verbose, 'S_parameter_posb':S_parameter_posb},
                 'options_ACO':options_ACO}
    
    optimizer.optimize(sarimax_ACO_PDQ_search_MAPE, iters=options_PSO['n_iterations'],verbose=verbose, **AllKwargs)
    
    global_best_result = sorted(search_results, key=lambda x: x[0])[-1]
    
    param = global_best_result[1]
    param_seasonal = global_best_result[2]
    listPosb = convertInt2PosList(global_best_result[3])
    
    logging.info("Global best result: pdq={0}, pdqs={1}, X={2}".format(param, param_seasonal, listPosb))
    
    if len(listPosb) > 0:
        true_exog = exog_var_matrix[:, listPosb]
    else:
        true_exog = None 
    mod = SARIMAX(endo_var, exog=true_exog, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

    results = mod.fit(disp=False)

    return results.predict()