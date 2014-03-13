import os
from string import Template


T = Template("import pickle \n\
import numpy as np\n\
from sklearn.ensemble import RandomForestRegressor\n\
from sklearn import clone\n\
import os\n\
from string import Template\n\
import sys\n\
import runTrialFunctions as myFuncs\n\
\n\
np.random.seed($REPNUM)\n\
\n\
outputfilename = '$MSEARCH$BSEARCH'+'_NUMMs_'+str(myFuncs.NUMMs)+'_NTREES_'+str(myFuncs.NTREES)+'_SIZETESTSET_'+str(myFuncs.SIZETESTSET)+'_rep$REPNUM.pkl'\n\
output = open(outputfilename, 'wb')\n\
\n\
#testsetfilename = 'testset_rep$REPNUM.pkl'\n\
#testsetfile = open(testsetfilename, 'wb')\n\
\n\
rep = 0\n\
MAXREP = 1\n\
MethodPerf = []\n\
while rep < MAXREP:\n\
    rep += 1\n\
    #%%% Generate testset %%%#\n\
    testdata = myFuncs.selectTestSet(myFuncs.SIZETESTSET)\n\
    #pickle.dump(testdata, testsetfile)\n\
    \n\
    \n\
    np.random.seed($REPNUM)\n\
    \n\
    Mfunc = myFuncs.$MSEARCH\n\
    Bfunc = myFuncs.$BSEARCH\n\
    MethodPerf.append(myFuncs.runSearch(testdata, Mfunc, Bfunc))\n\
    pickle.dump(MethodPerf, output)\n\
    print '$MSEARCH, $BSEARCH done'\n\
    \n\
output.close()\n\
#testsetfile.close()\n\
exit()\n\
# Need to randomize so that random value that achieves max is returned, not just the first one in argmax. \n\
\n\
")





for x in range(1, 2):
	Msearch = "unifRandM"
	Bsearch = "exhaustiveB"
	file = open("runTrial_" + Msearch + Bsearch + "-rep" + str(x) + ".py",'w')
	file.write(T.substitute(REPNUM=str(x), MSEARCH=Msearch, BSEARCH=Bsearch))
	file.close()





