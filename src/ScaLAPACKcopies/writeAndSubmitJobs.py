import os
from string import Template


T = Template("import pickle \n\
import numpy as np\n\
from sklearn.ensemble import RandomForestRegressor\n\
from sklearn import clone\n\
import os\n\
from string import Template\n\
import sys\n\
import runInteractiveJobsFunctions as myFuncs\n\
\n\
np.random.seed($REPNUM)\n\
\n\
outputfilename = 'NUMMs_'+str(myFuncs.NUMMs)+'_NTREES_'+str(myFuncs.NTREES)+'_SIZETESTSET_'+str(myFuncs.SIZETESTSET)+'_rep$REPNUM.pkl'\n\
output = open(outputfilename, 'wb')\n\
\n\
\n\
rep = 0\n\
MAXREP = 1\n\
randMexhBPerf= []\n\
randMrandBPerf = []\n\
randMtreeBperf = []\n\
randMtreeucbBPerf = []\n\
treeMrandBPerf  = []\n\
treeMexhBPerf = []\n\
treeMtreeBPerf = []\n\
while rep < MAXREP:\n\
    rep += 1\n\
    #%%% Generate testset %%%#\n\
    testdata = myFuncs.selectTestSet(myFuncs.SIZETESTSET)\n\
    \n\
\n\
    Mfunc = myFuncs.unifRandM\n\
    Bfunc = myFuncs.exhaustiveB\n\
    randMexhBPerf.append(myFuncs.runSearch(testdata, Mfunc, Bfunc))\n\
    pickle.dump(randMexhBPerf, output)\n\
    print 'random M, exhaustive B done'\n\
	\n\
    Mfunc = myFuncs.treeM;\n\
    Bfunc = myFuncs.treeB;\n\
    treeMtreeBPerf.append(myFuncs.runSearch(testdata, Mfunc, Bfunc))\n\
    pickle.dump(treeMtreeBPerf, output)\n\
    print 'tree M, tree B done'\n\
	\n\
	\n\
	\n\
output.close()\n\
exit()\n\
# Need to randomize so that random value that achieves max is returned, not just the first one in argmax. \n\
\n\
")



 #    Mfunc = myFuncs.unifRandM\n\
 #    Bfunc = myFuncs.randomB\n\
 #    randMrandBPerf.append(myFuncs.runSearch(testdata, Mfunc, Bfunc))\n\
 #    pickle.dump(randMrandBPerf, output)\n\
 #    print 'random M, random B done'\n\
 #    \n\
 #    Mfunc = myFuncs.unifRandM\n\
 #    Bfunc = myFuncs.treeB\n\
 #    randMtreeBperf.append(myFuncs.runSearch(testdata, Mfunc, Bfunc))\n\
 #    pickle.dump(randMtreeBperf, output)\n\
 #    print 'random M, tree B done'\n\
 #    \n\
 #    Mfunc = myFuncs.unifRandM\n\
 #    Bfunc = myFuncs.treeucbB\n\
 #    randMtreeucbBPerf.append(myFuncs.runSearch(testdata, Mfunc, Bfunc))\n\
 #    pickle.dump(randMtreeucbBPerf, output)\n\
 #    print 'random M, treeUCB B done'\n\
 #    \n\
 #    \n\
 #    \n\
 #    Mfunc = myFuncs.treeM\n\
 #    Bfunc = myFuncs.randomB\n\
 #    treeMrandBPerf.append(myFuncs.runSearch(testdata, Mfunc, Bfunc))\n\
 #    pickle.dump(treeMrandBPerf, output)\n\
 #    print 'treeM, random B done'\n\
 #    \n\
 #    Mfunc = myFuncs.treeM\n\
 #    Bfunc = myFuncs.exhaustiveB\n\
 #    treeMexhBPerf.append(myFuncs.runSearch(testdata, Mfunc, Bfunc))\n\
 #    pickle.dump(treeMexhBPerf, output)\n\
 #    print 'tree M, exhaustive B done'\n\
 #    \n\


S = Template("#PBS -q regular\n\
#PBS -l mppwidth=48,walltime=10:00:00\n\
#PBS -N rep$REPNUM\n\
#PBS -j oe\n\
#PBS -V\n\
\n\
cd $$PBS_O_WORKDIR\n\
export CRAY_ROOTFS=DSL\n\
export PYTHONPATH=$$HOME/python_modules/hopper/lib/python2.7/site-packages/\n\
module load python\n\
module load scipy\n\
module load numpy\n\
\n\
python src/runInteractiveJob_rep$REPNUM.py\n\
")






for x in range(1, 6):
	file = open("src/runInteractiveJob_rep" + str(x) + ".py",'w')
	file.write(T.substitute(REPNUM=str(x)))
	file.close()

	job = open("job.rep"+str(x),'w')
	job.write(S.substitute(REPNUM=str(x)))
	job.close()




