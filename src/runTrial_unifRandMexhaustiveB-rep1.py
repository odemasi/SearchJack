import pickle 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import clone
import os
import sys
import runTrialFunctions as myFuncs

np.random.seed(1)

outputfilename = 'unifRandMexhaustiveB'+'_NUMMs_'+str(myFuncs.NUMMs)+'_NTREES_'+str(myFuncs.NTREES)+'_SIZETESTSET_'+str(myFuncs.SIZETESTSET)+'_rep1.pkl'
output = open(outputfilename, 'wb')

#testsetfilename = 'testset_rep1.pkl'
#testsetfile = open(testsetfilename, 'wb')

rep = 0
MAXREP = 1
MethodPerf = []
while rep < MAXREP:
    rep += 1
    #%%% Generate testset %%%#
    #testdata = myFuncs.selectTestSet(myFuncs.SIZETESTSET)
    #pickle.dump(testdata, testsetfile)
    
    
    np.random.seed(1)
    
    Mfunc = myFuncs.unifRandM
    Bfunc = myFuncs.exhaustiveB
    MethodPerf.append(myFuncs.runSearch(testdata, Mfunc, Bfunc))
    print 'unifRandM, exhaustiveB done'
    
output.close()
pickle.dump(MethodPerf, output)
#testsetfile.close()
exit()
# Need to randomize so that random value that achieves max is returned, not just the first one in argmax. 
