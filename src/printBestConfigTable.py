import runTrialFunctions as myFuncs
import numpy as np


def optimizeForBench(X, y,  bench):
	optPerf = 0
	for design in X.keys():
		perf = perfMetric(X[design][bench], y[design])
		if perf > optPerf:
			optPerf = perf
			perfVec = [perfMetric(X[design][b],y[design]) for b in myFuncs.benchmarks]
			optDesign = myFuncs.design2vec(design)
	return perfVec + optDesign


def perfMetric(instCycles, area):
	return (float(instCycles[0])/float(instCycles[1]))/float(area)*10e+7

def strTrunc(num):
	digits = 5
	sp = str(num).split('.')
	return '.'.join([sp[0], sp[1][:digits]])

if __name__ == '__main__': 
	X, y = myFuncs.importdata(myFuncs.datafile, myFuncs.areafile)
	# X: table of instructions and cycles for each benchmark: X[designFromStr(design)][bench]
	# y: table of area measurements for each configuration
	
	Ncols = len(myFuncs.benchmarks) + 5
	M = np.NaN*np.zeros((len(myFuncs.benchmarks), Ncols))
	for i, bench in enumerate(myFuncs.benchmarks):
		M[i,:] = optimizeForBench(X, y, bench)

	f = open('table.txt','w')
	f.write('\t' + ' '.join(myFuncs.benchmarks) + '\n')
	for i, row in enumerate(M):
		f.write(myFuncs.benchmarks[i][:4] + '\t | ' +' '.join( map(strTrunc, row[:Ncols-5]) )+ '\t|' +' '.join( map(strTrunc, row[Ncols-5:]) )+ '\n')




