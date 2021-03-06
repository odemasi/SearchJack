import pickle 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import clone
import os
import sys
import runTrialFunctions as myFuncs

np.random.seed(1)

outputfilename = 'unifRandMexhaustiveB'+'_NTREES_'+str(myFuncs.NTREES)+'_rep1.pkl'
output = open(outputfilename, 'wb')


rep = 0
MAXREP = 10
MethodPerf = []
while rep < MAXREP:
    rep += 1    
    Bfunc = myFuncs.treeucbB
    MethodPerf.append(myFuncs.runSearch(Bfunc))
 print ' exhaustiveB done'
    
pickle.dump(MethodPerf, output)
output.close()
# Need to randomize so that random value that achieves max is returned, not just the first one in argmax. 

