import scipy.io


def loadMatlabData(path):
    mat = scipy.io.loadmat(path)
    x_data = mat.get('X')
    categorical_data = mat.get('C')
    return x_data.transpose(), categorical_data.transpose()


def loadMatlabDataForPaper(path):
    mat = scipy.io.loadmat(path)
    x_data = mat.get('Xtrans')
    categorical_data = mat.get('Ctrans')
    x_data_noisy = mat.get('Xnoisetrans')
    categorical_data_noisy = mat.get('Ctrans')

    return x_data.transpose(), categorical_data.transpose(), x_data_noisy.transpose(), categorical_data_noisy.transpose()


def loadMatlabDataRealData(path):
    data = loadmat(path)
    xTrue = data['PLRNN']['data'].transpose()
    externalInputs = data['PLRNN']['Inp'].transpose()
    movementRegressors = data['PLRNN']['rp']
    #print(xTrue, externalInputs, movementRegressors)

    return xTrue, externalInputs, movementRegressors

def loadMatlabDataRealDataMultimodal(path):
    data = loadmat(path)
    xTrue = data['PLRNN']['data'].transpose()
    externalInputs = data['PLRNN']['Inp'].transpose()
    movementRegressors = data['PLRNN']['rp']
    cTrue = data['PLRNN']['responses']['resp1']
    cTrue2 = data['PLRNN']['responses']['resp2']
    #print(xTrue.shape, externalInputs.shape, movementRegressors.shape)

    return xTrue, externalInputs, movementRegressors, cTrue


def loadmat(filename):
    ''' this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

