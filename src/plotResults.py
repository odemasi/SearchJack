import pickle
import sys
from matplotlib import pyplot as plt
import numpy as np

datafiles = ['randomB_NTREES_5.pkl','treeB_NTREES_5.pkl','treeucbB_NTREES_5.pkl']

for datafile in datafiles:
	methodPerf = pickle.load(open(datafile,'r'))
	perf = np.NaN*np.zeros((len(methodPerf),len(methodPerf[0]['best'])))
	for i, run in enumerate(methodPerf):
		# perf += run['observed'][:,5]
		perf[i,:] = run['best']

	plt.errorbar(range(perf.shape[1]), np.mean(perf, axis=0), np.std(perf,axis=0), fmt='o')
	# plt.plot(np.mean(perf, axis=0),'o-', label=datafile)
plt.legend(datafiles, loc=4)
plt.show()


# 1. randomize sample within set of same prediction values
# 2. force exploration
# 3. implement standard (Nelder Meade?) methods
# 4. implement genetic methods





