import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import clone
import os
from string import Template
import sys
import time


homedir = "/scratch/odemasi/SearchJack/"



def SpecialEnumerateSearchSpace(paramRanges):
	#***
	# needs to return a np.array of all possible search combinations. 
	# each row is a feasible configuration
	space = np.array()
    return space



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


def  runTests(M, Bs):
    perf = []
    timespent = []
    for b in range(0, Bs.shape[0]):
        # execute python script to write and execute script
        p, t = runTest(M, Bs[b:]) 
        perf.append(p)
        timespent.append(t)
        
        pstack = np.vstack(np.array(perf))
        tstack = np.vstack(np.array(timespent))
        return pstack, tstack



def runTest(M, B):
    # ***
    # must be written. Returns two lists of length=1. 
    # can be modified to return whichever metrics are important. 
    return perf, timespent



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


def chooseM(F, Mfunc, inputRanges):
    M, predsM, stdevM = Mfunc(F, inputRanges)
    return M, predsM, stdevM


def unifRandM(F, inputRanges):
    #% randomly choose an input value for every input coordinate
    # OUTPUT row vectors 
    Mvalue = np.random.randint( MATMIN, high = MATMAX, size = (1,2))

    if F == []:
        predsM = []
        stdevM = []
    else:
        featurespace = np.concatenate((np.tile(Mvalue, (SIZESPACE, 1)), fullspace), axis = 1)
        predsM, stdevM = myPredict(F, featurespace) #debugging

    return Mvalue, predsM, stdevM


# %%% --- M for which we are least certain about upper envelope --- %%%
def treeM(F, inputRanges):
    if F == []:
        # % choose a random M
        Mvalue, predsM, stdevM = unifRandM(F, inputRanges)
    else:
        # % variance in the prediction for the best prediction hat(B)
        maxstdev = 0;
        # for m = 1:NUMMs
        for m in range(NUMMs):
            # %choose a random M
            #% randomly choose an input value for every input coordinate
            tmpM = np.random.randint( MATMIN, high = MATMAX, size = (1,2))
            featurespace = np.concatenate((np.tile(tmpM, (SIZESPACE, 1)), fullspace), axis = 1)
            preds, stdev = myPredict(F, featurespace)
            
            # % check predictions on that M
            i = np.argmax(preds) #debugging, only the first occurance in returned...
            
            # % keep M if there is high variance in the prediction of its max
            if stdev[i] >= maxstdev: #debugging - I think this should be strict inequality.
                predsM = preds
                stdevM = stdev 
                Mvalue = tmpM
                maxstdev = stdev[i]
    return Mvalue, predsM, stdevM



def chooseB(funcB, predsM, stdevM):
    Bs = funcB(predsM, stdevM); 
    # Bs = np.array([0,0,0]).reshape(1, 3)
    return Bs

def exhaustiveB(predsM, stdevM):
    return fullspace

def randomB(predsM, stdevM):
    temp = myRandI(SIZESPACE, BSUBSETSIZE)
    Bs = fullspace[temp,:]
    return Bs

def treeB(predsM, stdevM):
	# B with highest predicted variance.
    if predsM == []:
        Bs = randomB(predsM, stdevM)
    else:
        #% choose point that has most uncertainty
        # [~, ind] = sort(stdevM, 'descend')
        ind = np.argsort(stdevM) #ascending
        ind = ind[::-1]
        Bs = fullspace[ind[1:BSUBSETSIZE],:]
    return Bs

def treeucbB(predsM, stdevM):
	# Pick B that has best trade off of prediction with uncertainty
    if predsM == []:
        Bs = randomB(predsM, stdevM)
    else:
        # % choose point trades off exploration/exploitation
        # [~, ind] = sort(predsM + stdevM, 'descend'); 
        # Bs = fullspace(ind(1:BSUBSETSIZE),:);  
        ind = np.argsort(predsM + stdevM) #ascending
        ind = ind[::-1]
        Bs = fullspace[ind[1:BSUBSETSIZE],:]
    return Bs

def measureMethodPerf(F, testdata):
	# measure performance on held out test set (for which we know true best)
	numTests = len(testdata['maxperf'])

    temp = NaN*np.zeros((1,numTests))
    time = np.zeros(1)
    for i in range(numTests):
       # % predict which one to run on 
       nBs = fullspace.shape[0]
       X = np.concatenate((np.tile(testdata['Mvalue'][i], (nBs, 1)), fullspace),axis=1)
       preds, discard = myPredict(F, X)
       ind = np.argmax(preds) #debugging: need to randomize if multiple values achieve max
        
       # % look up how config would have performed
       perffrompred = testdata['perf'][i][ind]
       
       # % look up time 
       time = time[0] + testdata['time'][i][ind]
       
       # % compare with true best
       temp[0,i] = (perffrompred - testdata['minperf'][i])/(testdata['maxperf'][i]  - testdata['minperf'][i] ); 
#        temp(i) = (perfOfPred - trueMin)/(trueMax - trueMin); 
    flopsHat = np.mean(temp)
    return flopsHat, time



def runSearch(testdata, Mfunc, Bfunc):
	
    F = []
    
    observed = NaN*np.zeros((MAXOBS, len(inputRanges) + NUMPARAMS + 1))
    methodPerf= {'numtried': []}
    methodPerf['numtried'] = NaN*np.zeros((1, MAXOBS))
    methodPerf['flopsHat'] = NaN*np.zeros((1, MAXOBS))
    methodPerf['timespent'] = NaN*np.zeros((1, MAXOBS))
    methodPerf['timetraining'] = NaN*np.zeros((1, MAXOBS))
    #% methodPerf.configstried = NaN * zeros( MAXOBS, length(inputRanges)  + NUMPARAMS);

    numObs = 0
    iter = 0 
    while numObs < MAXOBS:
        
        #% return M and predictions for entire M space
        M, predsM, stdevM = chooseM(F, Mfunc, inputRanges)
        
        #% these are the new configurations that we are going to try
        Bs = chooseB(Bfunc, predsM, stdevM)
        
        #% query function to get new data
        perf, timespent = runTests(M, Bs)
        nBs = Bs.shape[0]
        temp = np.concatenate((np.tile(M, (nBs, 1)), Bs), axis=1)
        if numObs+nBs > MAXOBS:
            newObserved = NaN*np.zeros((numObs + nBs, len(inputRanges) + NUMPARAMS + 1))
            newObserved[0:numObs, :] = observed[0:numObs, :]
            newObserved[numObs : numObs + nBs,:] = np.concatenate((temp, perf), axis=1)
            observed = newObserved
        else:
             observed[numObs : numObs + nBs,:] =  np.concatenate((temp, perf), axis=1)
        
        numObs = numObs + nBs; 
        
        #% update model
        F = clone(model)
        F.fit(observed[0:numObs, 0:len(inputRanges) + NUMPARAMS], np.ravel(observed[0:numObs, len(inputRanges) + NUMPARAMS]))
        # F = TreeBagger(NTREES,observed(:, 1:end-1), observed(:, end), 'method', 'regression', 'Options', treeOptions); 
        #%F = TreeBagger(NTREES,observed(:, 1:end-1), observed(:, end),'method', 'regression'); 

        #% update stats on search
        methodPerf['numtried'][0,iter] = nBs 
        methodPerf['flopsHat'][0,iter], methodPerf['timespent'][0,iter] = measureMethodPerf(F, testdata) #*** debugging
        methodPerf['timetraining'][0,iter] = np.sum(timespent)
        print 'measured time training'
        iter = iter + 1; 
    return methodPerf   





inputRanges = [range(MATMIN,MATMAX), range(MATMIN,MATMAX)]
paramRanges=[[8, 16, 24, 32, 40, 48, 56, 64],[1, 2, 4, 8, 16], [16, 8, 4, 2, 1]]
NUMPARAMS = len(paramRanges)


fullspace = SpecialEnumerateSearchSpace(paramRanges)
SIZESPACE = np.size(fullspace, axis = 0) 
NaN = np.nan

MAXOBS = 800 # maximum number of observations
NUMMs = 500 # number of random M's that we consider in treeM
NTREES = 15 
model = RandomForestRegressor(n_estimators=NTREES)

BSUBSETSIZE = 5 
SIZETESTSET = 25 




