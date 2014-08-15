import pickle 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import clone
import os
from string import Template
import sys
import runTrialFunctions as myFuncs

np.random.seed(0)

outputfilename = 'treeoptB'+'_NTREES_'+str(myFuncs.NTREES)+ '_' + myFuncs.bench +'.pkl'
output = open(outputfilename, 'wb')

rep = 0
MAXREP = 10
MethodPerf = []
while rep < MAXREP:
    rep += 1
    Bfunc = myFuncs.treeoptB
    MethodPerf.append(myFuncs.runSearch(Bfunc))
    
pickle.dump(MethodPerf, output)
print 'treeoptB done'
    
output.close()
# Need to randomize so that random value that achieves max is returned, not just the first one in argmax. 

