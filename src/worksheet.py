
import numpy as np



homedir = "/Users/odemasi/LocalProjects/SearchJack/"
datafile = homedir + 'data/results-vcs.txt'
areafile = homedir + 'data/results-dcs.txt'
benchmarks = ['dgemm', 'dhrystone', 'median', 'multiply', 'qsort', 'spmv','towers', 'vvadd']
Nbench = len(benchmarks)


def importdata():
	df = open(datafile, 'r')
	af = open(areafile,'r')
	X = {}
	y = {}

	for line in af:
		design, area = line.strip('\n').split(',')
		y[designFromStr(design)] = area
		X[designFromStr(design)] = {}

	for line in df:
		design, bench, cycles, instructions = line.strip('\n').split(',')
		X[designFromStr(design)][bench] = [cycles, instructions]
	return X, y


def vec2design(datavec):
	return 'design_nm-%d_is-%d_iw-%d_ds-%d_dw-%d' % datavec

def designFromStr(designstr):
	return '_'.join(designstr.split('_')[:-1])
	 

def design2vec(design):
	datavec = [x.split('-')[1] for x in design.split('_')[1:]]
	return datavec


def sliceOnBenchMark(X, y, bench):
	Ndesigns = len(X.keys())
	Nattr = len(design2vec(X.keys()[0]))
	mat = np.NaN*np.zeros((Ndesigns,Nattr))
	perf = np.NaN*np.zeros(Ndesigns)
	for i, design in enumerate(X.keys()):
		mat[i,:] = X[design][bench]

	return mat, perf

def enumerateSearchSpace(X):
	Ndesigns = len(X.keys())
	print X.keys()[0]
	print design2vec(X.keys()[0])
	Nattr = len(design2vec(X.keys()[0]))
	mat = np.NaN*np.zeros((Ndesigns, Nattr))

	for i, design in enumerate(X.keys()):
		mat[i,:] = design2vec(design)
	
	return mat


if __name__ == '__main__':
	X, y = importdata()
	vec2design(tuple([2,1,3,4,2]))
	mat = enumerateSearchSpace(X)
	for i in xrange(mat.shape[1]):
		print i, np.unique(mat[:,i])

