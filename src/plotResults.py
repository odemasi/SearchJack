import pickle
import sys
from matplotlib import pyplot as plt
import numpy as np


benchmarks = ['dgemm', 'dhrystone', 'median', 'multiply', 'qsort', 'spmv','towers', 'vvadd']
NTREES = 10
searchMethods = ['randomB', 'treeB', 'treeoptB', 'treeucbB', 'oracleB']
for bench in benchmarks:
	for search in searchMethods:
		datafile = '_'.join([search, 'NTREES', str(NTREES), bench]) + '.pkl'
		methodPerf = pickle.load(open(datafile,'r'))
		perf = np.NaN*np.zeros((len(methodPerf),len(methodPerf[0]['best'])))
		for i, run in enumerate(methodPerf):
			# perf += run['observed'][:,5]
			# perf[i,:] = run['observed'][:,5]
			perf[i,:] = run['best']

		# plt.plot(range(perf.shape[1]), np.mean(perf, axis=0), 'o-')
		plt.errorbar(range(perf.shape[1]), np.mean(perf, axis=0), np.std(perf,axis=0), fmt='o')
		
		# plt.plot(np.mean(perf, axis=0),'o-', label=datafile)
	plt.legend(searchMethods, loc=4)
	plt.xlabel('# points sampled', fontsize = 16)
	plt.ylabel('Instructions per cycle per unit area', fontsize = 16)
	plt.title('Search progress over design space, benchmark = '+ bench, fontsize=16)
	plt.savefig('ResultsObjective_'+bench+'.png')
	plt.show()


	# 1. randomize sample within set of same prediction values
	# 2. force exploration
	# 3. implement standard (Nelder Meade?) methods
	# 4. implement genetic methods





