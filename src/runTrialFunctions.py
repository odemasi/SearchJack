import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import clone
import os
from string import Template
import sys
import time


homedir = "/Users/odemasi/LocalProjects/SearchJack/"
datafile = homedir + 'data/results-vcs.txt'
areafile = homedir + 'data/results-dcs.txt'
benchmarks = ['dgemm', 'dhrystone', 'median', 'multiply', 'qsort', 'spmv','towers', 'vvadd']
Nbench = len(benchmarks)


def importdata(datafile, areafile):
    df = open(datafile, 'r')
    af = open(areafile,'r')
    X = {} # table of instructions and cycles for each benchmark: X[designFromStr(design)][bench]
    y = {} # table of area measurements for each configuration
    for line in af:
        design, area = line.strip('\n').split(',')
        y[designFromStr(design)] = area
        X[designFromStr(design)] = {}
    for line in df:
        design, bench, cycles, instructions = line.strip('\n').split(',')
        X[designFromStr(design)][bench] = [cycles, instructions]
    return X, y


def vec2design(datavec):
    return 'design_nm-%d_is-%d_iw-%d_ds-%d_dw-%d' % datavec

def designFromStr(designstr):
    return '_'.join(designstr.split('_')[:-1])
     

def design2vec(design):
    datavec = map(float, [x.split('-')[1] for x in design.split('_')[1:]])
    return datavec


def enumerateSearchSpace(X):
    Ndesigns = len(X.keys())
    Nattr = len(design2vec(X.keys()[0]))
    space = np.NaN*np.zeros((Ndesigns, Nattr))
    for i, design in enumerate(X.keys()):
        space[i,:] = design2vec(design)
    return space


# def enumerateSearchSpace(X, y):
# 	#***
# 	# needs to return a np.array of all possible search combinations. 
# 	# each row is a feasible configuration
# 	space = np.random.rand(100,5)
# 	return space
	
# def selectTestSet(SIZETESTSET):
# 
#     newdata = {'Mvalue':[], 'perf':[], 'time':[], 'maxperf':[], 'minperf':[] }
#     for i in range(SIZETESTSET):
#         #% choose a random M
#         Mvalue = np.random.randint( MATMIN, high = MATMAX, size = (1,2))
# 
#         
#         Bs = chooseB(exhaustiveB, [], [])
# 
#         #% run every configuration to know best configuration for that M
#         # dictionary of arrays 
#         newdata['Mvalue'].append(Mvalue)
#         # newdata['Bs'].append(Bs)
#         perf, time = runTests(Mvalue, Bs)
#         newdata['perf'].append(perf) #this should append an np.array
#         newdata['time'].append(time)
#         print "newdata['maxperf']"
#         print newdata['maxperf'] # [array([ 68152.12]), array([ 49552.59]), ...]
#         
#         newdata['maxperf'].append(max(perf)) # this should be appending a scalar... is it?  
#         newdata['minperf'].append(min(perf))
# 
#     return newdata


def runTests( Bs):
    #OUTPUT: perf - np.array. One row for each B, each row is set of performance metrics for given B. 
    perf = NaN * np.zeros((Bs.shape[0], NUMPERFMETRICS))
    for b in range(0, Bs.shape[0]):
        perf[b,:] = runTest(Bs[b,:])
    return perf



def runTest(B):
    # ***
    # must be written. Return numpy row vector of performance metrics. 
    # can be modified to return whichever metrics are important. 
    design = vec2design(tuple(B))
    inst = float(X[design][bench][0])
    cycles = float(X[design][bench][1])
    area = float(y[design])
    perf = (inst/cycles)/area*10e+7
    return perf 

def oracle():
    optperf = 0
    for design in y.keys():
        perf = runTest(design2vec(design))
        if perf > optperf:
            optdesign = np.array(design2vec(design))
            optperf = perf
    bestB = np.NaN*np.zeros((1,optdesign.shape[0]))
    bestB[0,:] = optdesign
    return bestB

def myPredict(F, featurespace):
	# INPUT: 
	# F  - trained model, random forest
	# featurespace - 2D np.array of configurations for which predictions are need. each row is a configuration. 
    # OUTPUT:
    # mu - np.array of predicted values
    # sig - np.array of standard deviation for each prediction
    
    mu = []
    sig = []
    for i in range(featurespace.shape[0]):
        x = featurespace[i, :]
        treepreds = NaN*np.zeros((1, NTREES))
        for tree in range(F.n_estimators):
            treepreds[0,tree] = F.estimators_[tree].predict(x)

        mu.append(np.mean(treepreds))
        sig.append(np.sqrt(np.var(treepreds)))

    return np.array(mu), np.array(sig)


    
def myRandI(high, numSelect):
	# takes random sample without replacement. 
	# OUTPUT: np.array of integers, len = numSelect, between (1, high)
    ints = np.unique(np.random.randint(0, high, size=numSelect))
    while ints.size <  numSelect:
        ints = np.unique(np.concatenate((ints, np.random.randint(0,high, size=1)),axis=0))
    return ints

def chooseB(funcB, F):
    Bs = funcB(F); 
    # Bs = np.array([0,0,0]).reshape(1, 3)
    return Bs

# def exhaustiveB(F):
#     return designspace

def randomB(F):
    temp = myRandI(SIZESPACE, BSUBSETSIZE)
    Bs = designspace[temp,:]
    return Bs

def treeB(F):
	# B with highest predicted variance.
    if F == []:
        Bs = randomB(F)
    else:
        #% choose point that has most uncertainty
        pred, stdev = myPredict(F, designspace)
        obj = stdev
        inds = np.nonzero(obj == np.max(obj))[0]
        Bs = designspace[inds[np.random.randint(len(inds), size=BSUBSETSIZE)],:]
    return Bs

def treeucbB(F):
	# Pick B that has best trade off of prediction with uncertainty
    if F == []:
        Bs = randomB(F)
    else:
        # % choose point trades off exploration/exploitation
        pred, stdev = myPredict(F, designspace)
        obj = pred + stdev
        inds = np.nonzero(obj == np.max(obj))[0]
        Bs = designspace[inds[np.random.randint(len(inds), size=BSUBSETSIZE)],:]
    return Bs

def treeoptB(F):
    # Pick B that has best predicted value
    if F == []:
        Bs = randomB(F)
    else:
        # % choose point trades off exploration/exploitation
        pred, stdev = myPredict(F, designspace)
        obj = pred
        inds = np.nonzero(obj == np.max(obj))[0]
        Bs = designspace[inds[np.random.randint(len(inds), size=BSUBSETSIZE)],:]
    return Bs

def oracleB(F):
    # Pick B that has best predicted value
    return ORACLE

# def treeucbB(F):
#     # Pick B that has best trade off of prediction with uncertainty
#     if F == []:
#         Bs = randomB(F)
#     else:
#         # % choose point trades off exploration/exploitation
#         pred, stdev = myPredict(F, designspace)
#         obj = pred + stdev
#         inds = np.nonzero(obj == np.max(obj))[0]
#         Bs = designspace[inds[np.random.randint(len(inds), size=BSUBSETSIZE)],:]
#     return Bs


# def measureMethodPerf(F, testdata):
# 	# measure performance on held out test set (for which we know true best)
# 	numTests = len(testdata['maxperf'])
# 
#     temp = NaN*np.zeros((1,numTests))
#     time = np.zeros(1)
#     for i in range(numTests):
#        # % predict which one to run on 
#        nBs = designspace.shape[0]
#        X = np.concatenate((np.tile(testdata['Mvalue'][i], (nBs, 1)), designspace),axis=1)
#        preds, discard = myPredict(F, X)
#        ind = np.argmax(preds) #debugging: need to randomize if multiple values achieve max
#         
#        # % look up how config would have performed
#        perffrompred = testdata['perf'][i][ind]
#        
#        # % look up time 
#        time = time[0] + testdata['time'][i][ind]
#        
#        # % compare with true best
#        temp[0,i] = (perffrompred - testdata['minperf'][i])/(testdata['maxperf'][i]  - testdata['minperf'][i] ); 
# #        temp(i) = (perfOfPred - trueMin)/(trueMax - trueMin); 
#     flopsHat = np.mean(temp)
#     return flopsHat, time



def runSearch(Bfunc):
    F = []
    
    observed = NaN*np.zeros((MAXOBS, NUMPARAMS + NUMPERFMETRICS))
    methodPerf = {'numtried' : NaN*np.zeros(MAXOBS), 'numobs': NaN*np.zeros(MAXOBS), 'best': NaN*np.zeros(MAXOBS)}

    numObs = 0
    iter = 0 
    while numObs < MAXOBS:
        
        # these are the new configurations that we are going to try
        Bs = chooseB(Bfunc, F)
        
        # query function to get new data
        perf = runTests( Bs)
        
        #store "observed" data
        nBs = Bs.shape[0]
        
        if numObs+nBs > MAXOBS:
            newObserved = NaN*np.zeros((numObs + nBs, NUMPARAMS + NUMPERFMETRICS))
            newObserved[0:numObs, :] = observed[0:numObs, :]
            newObserved[numObs : numObs + nBs,:] = np.concatenate((Bs, perf), axis=1)
            observed = newObserved
        else:
             observed[numObs : numObs + nBs,:] =  np.concatenate((Bs, perf), axis=1)
        
        numObs = numObs + nBs; 
        print nBs, numObs
        print Bs, perf
        # update model
        F = clone(model)
        F.fit(observed[0:numObs, 0:NUMPARAMS], np.ravel(observed[0:numObs, NUMPARAMS]))
        
        # update stats on search
        methodPerf['numtried'][iter] = nBs 
        methodPerf['numobs'][iter] = numObs 
        iter += 1;
    methodPerf['observed'] = observed 
    methodPerf['observedPerf'] = observed[:, NUMPARAMS]
    methodPerf['best'][0]= methodPerf['observedPerf'][0]
    for i, item in enumerate(methodPerf['observedPerf'][1:]):
        methodPerf['best'][i+1] = np.max([methodPerf['best'][i], item])
    return methodPerf   



homedir = "/Users/odemasi/LocalProjects/SearchJack/"
datafile = homedir + 'data/results-vcs.txt'
areafile = homedir + 'data/results-dcs.txt'
benchmarks = ['dgemm', 'dhrystone', 'median', 'multiply', 'qsort', 'spmv','towers', 'vvadd']
bench = 'dhrystone'
Nbench = len(benchmarks)

X, y = importdata(datafile, areafile)
ORACLE = oracle()


designspace = enumerateSearchSpace(X)
SIZESPACE = designspace.shape[0]
NUMPARAMS = designspace.shape[1]
NUMPERFMETRICS = 1 #***
NaN = np.nan

MAXOBS = 100#SIZESPACE # maximum number of observations
# NUMMs = 500 # number of random M's that we consider in treeM
NTREES = 5 
model = RandomForestRegressor(n_estimators=NTREES)

BSUBSETSIZE = 1 
# SIZETESTSET = 25 

