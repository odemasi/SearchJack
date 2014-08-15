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
np.random.seed(0)\n\
\n\
outputfilename = '$BSEARCH'+'_NTREES_'+str(myFuncs.NTREES)+ '_' + myFuncs.bench +'.pkl'\n\
output = open(outputfilename, 'wb')\n\
\n\
rep = 0\n\
MAXREP = 10\n\
MethodPerf = []\n\
while rep < MAXREP:\n\
    rep += 1\n\
    Bfunc = myFuncs.$BSEARCH\n\
    MethodPerf.append(myFuncs.runSearch(Bfunc))\n\
    \n\
pickle.dump(MethodPerf, output)\n\
print '$BSEARCH done'\n\
    \n\
output.close()\n\
# Need to randomize so that random value that achieves max is returned, not just the first one in argmax. \n\
\n\
")




Bsearches = ['randomB', 'treeB', 'treeucbB','oracleB','treeoptB']
for search in Bsearches:
	file = open("runTrial_" + search + ".py",'w')
	file.write(T.substitute( BSEARCH=search))
	file.close()





