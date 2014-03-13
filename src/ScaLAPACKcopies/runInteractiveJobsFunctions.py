import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import clone
import os
from string import Template
import sys
import time

# import runInteractiveConstants

homedir = "AutoTuning/ScaLAPCK/LU/testset1/"


T = Template("'SCALAPACK, LU factorization input file'\n\
'MPI Machine'\n\
'LU.out'        output file name (if any)\n\
6           device out\n\
1           number of problems sizes\n\
$M  values of M\n\
$N  values of N\n\
1           number of NB's\n\
$NBvalue        values of NB\n\
1           number of NRHS's\n\
1 3 9 28        values of NRHS\n\
1           Number of NBRHS's\n\
1 3 5 7         values of NBRHS\n\
1           number of process grids (ordered pairs of P & Q)\n\
$Pvalue         values of P\n\
$Qvalue         values of Q\n\
1.0         threshold\n\
T           (T or F) Test Cond. Est. and Iter. Ref. Routines\n")

def writeLUdat(M, B):
    # temp = open(homedir + 'LU.dat.template')
    output = open('LU.dat','w')
    contents = T.substitute(M = M[0], N = M[1], NBvalue = B[0], Pvalue = B[1], Qvalue = B[2])
    output.write(contents)
    output.close




def parseTest(filename, datafile):
    rawfile = open(filename,'r')
    outputfile = open(datafile, 'w')
    data = []
    for line in rawfile:
        if line[0:4] == 'WALL' and 'PASSED' in line:
            pieces = line.split(' ')
            data.append(filter(lambda x: x != '', pieces[1:-1]))
            outputfile.write(' '.join(filter(lambda x: x != '', pieces[1:-1])) + '\n')
    
    perf = map(float, [x[9] for x in data])
    timespent = map(float, [x[7] for x in data])
    outputfile.close()
    rawfile.close()    
    print "perf and timespent (Should each be list len 1):"
    print perf
    print timespent
    if len(perf) != 1 and len(timespent) != 1:
        print perf
        print timespent
        print filename
    return perf, timespent

def readDataFile(datafile):
	outputfile = open(datafile, 'r')
	data = []
	for line in outputfile:
	    pieces = line.strip('\n').split(' ')
	    data.append(pieces)
	outputfile.close()
	perf = map(float, [x[9] for x in data])
	timespent = map(float, [x[7] for x in data])
	return perf, timespent 



def SpecialEnumerateSearchSpace(paramRanges):
    A = np.array(paramRanges[1])
    B = np.array(paramRanges[2])
    procBlockings = np.concatenate( (np.vstack(A), np.vstack(B)), axis=1)
    N = len(paramRanges[1]) 
    space = np.concatenate((paramRanges[0][0] * np.ones((N, 1)), procBlockings), axis=1)


    for b in range(1,len(paramRanges[0])):
        newblock = np.concatenate((paramRanges[0][b] * np.ones((N, 1)), procBlockings), axis=1)
        space = np.concatenate((space, newblock), axis=0)

    return space



def selectTestSet(SIZETESTSET):

    newdata = {'Mvalue':[], 'perf':[], 'time':[], 'maxperf':[], 'minperf':[] }
    for i in range(SIZETESTSET):
        #% choose a random M
        Mvalue = np.random.randint( MATMIN, high = MATMAX, size = (1,2))

        
        Bs = chooseB(exhaustiveB, [], [])

        #% run every configuration to know best configuration for that M
        # dictionary of arrays 
        newdata['Mvalue'].append(Mvalue)
        # newdata['Bs'].append(Bs)
        perf, time = runTests(Mvalue, Bs)
        newdata['perf'].append(perf) #this should append an np.array
        newdata['time'].append(time)
        print "newdata['maxperf']"
        print newdata['maxperf'] # [array([ 68152.12]), array([ 49552.59]), ...]
        
        newdata['maxperf'].append(max(perf)) # this should be appending a scalar... is it?  
        newdata['minperf'].append(min(perf))

    return newdata


def  runTests(M, Bs):
    perf = []
    timespent = []
    # if len(Bs.shape) == 1:
    #     Bs = Bs.reshape(1, Bs.shape[0])
    for b in range(0, Bs.shape[0]):
        #% execute python script to write and execute script
        #handle = sprintf('MatSize_%ix%i/LUresults_%ix%i_%i-%i-%i', M(1), M(2), M(1), M(2), Bs(b, 1), Bs(b, 2), Bs(b, 3)); 
        p, t = runTest(M, Bs[b:]) 
        perf.append(p)
        timespent.append(t)
        
        try:
        	pstack = np.vstack(np.array(perf))
        	tstack = np.vstack(np.array(timespent))
        except Exception, e: 
        	print perf
        	print timespent
        	print M
        	print Bs
        	print e
        	sys.exit()
    return pstack, tstack



def runTest(M, B):
    
    if M == [] or B ==[]:
        sys.exit('exited - failed to pass correct arguments')
    
    dirname = 'MatSize_'+ 'x'.join(map(str, M[0])) 
    filehandle = 'LUresults_'+ 'x'.join(map(str, M[0])) + '_' + '-'.join(map(str, B[0]))

    outfile = filehandle + '.out'
    errfile = filehandle + '.err'
    datafile = filehandle +'.data'
    
    print "editing: " + dirname
    if not os.path.isdir(dirname): #first touch - test must be run
        os.mkdir(dirname)
        open(dirname + '/LU.locked', 'w').close()
        open(dirname + '/' + datafile + 'locked', 'w').close()
        
        os.chdir(dirname)
        # check if test has been run before 
        writeLUdat(map(str, M[0]), map(str, B[0]))
        #os.system('ccmrun mpirun -np 16 -hostfile $PBS_NODEFILE ../../xdlu > ' + outfile + ' 2> '+  errfile)
        os.system('aprun -n 16 -N 8 -S 2 -ss ../../xdlu > ' + outfile + ' 2> '  + errfile)
        
        perf, timespent = parseTest(outfile, datafile)
        
        os.remove(datafile + 'locked')
        os.remove('LU.locked')
        os.chdir('..')
        
    
    else: #the directory exists, must check test though
		os.chdir(dirname)
		
		if os.path.isfile('LU.locked'): # a test is running
			# wait for it, if it is the same test...
			t = time.time()
			# wait 1500 seconds for the test to complete.
			while os.path.isfile(datafile + 'locked') and time.time() - t < 1500: #while test is running/if specific test is running
				time.sleep(5)
				
			if time.time() - t > 1500:
				#check lock file was not forgotten
				if os.path.isfile(datafile) and os.path.getsize(datafile) != 0: # looks like test was run successfully
					perf, timespent = readDataFile(datafile)
					if len(perf) == 1 and leng(timespent) == 1:#output looks good
					    os.remove(datafile + 'locked')
					    os.remove('LU.locked')
					else:
					    print perf
					    print timespent
					    print dirname
					    print filehandle
					    raise Exception('test did not complete in 1500 seconds... check for a problem')
				elif os.path.isfile(datafile) and os.path.getsize(datafile) == 0: #maybe test ran and failed, try rerunning  #*** DEBUG - could cause race if test i s >1500, but it shouldn't be...
					print 'making a bad decision, rerunning code and removing lock'
					writeLUdat(map(str, M[0]), map(str, B[0]))
					os.system('aprun -n 16 -N 8 -S 2 -ss ../../xdlu > ' + outfile + ' 2> '  + errfile)
					perf, timespent = parseTest(outfile, datafile)
					os.remove(datafile + 'locked')
					os.remove('LU.locked')
				else:
					raise Exception('I dunno, datafile is not a file....')
# 				else: #This is terrible, try running code. It should have been completed by now, maybe it failed. 
			elif not os.path.isfile(datafile):#test has not been run yet, spin on locked dir
				t = time.time()
				while os.path.isfile('LU.locked') and time.time() - t < 1500:
					time.sleep(5)
				if time.time()-t < 1500:
					#run test
					open('LU.locked', 'w').close()
					open( datafile + 'locked', 'w').close()
					writeLUdat(map(str, M[0]), map(str, B[0]))
					os.system('aprun -n 16 -N 8 -S 2 -ss ../../xdlu > ' + outfile + ' 2> '  + errfile)
					perf, timespent = parseTest(outfile, datafile)
					os.remove(datafile + 'locked')
					os.remove('LU.locked')
				else: # directory is locked on other test, which is failed. # DEBUG: Also a bad decision, but test should be done after 1500s
					print dirname
			    	print filehandle
			    	print 'I think directory is locked on failed test, going to try running'
			    	open( datafile + 'locked', 'w').close()
			    	writeLUdat(map(str, M[0]), map(str, B[0]))
			    	os.system('aprun -n 16 -N 8 -S 2 -ss ../../xdlu > ' + outfile + ' 2> '  + errfile)
			    	perf, timespent = parseTest(outfile, datafile)
			    	os.remove(datafile + 'locked')
			    	os.remove('LU.locked')
# 				else: #something is really wrong
# 				    print dirname
# 				    print filehandle
# 				    raise Exception('something is really wrong')
			else:
				perf, timespent = readDataFile(datafile)	
		elif os.path.isfile(datafile) and os.path.getsize(datafile) != 0 and not os.path.isfile('LU.locked'): #Nothing is running and test is ready
			perf, timespent = readDataFile(datafile)
			
		elif (not os.path.isfile(datafile) or os.path.getsize(datafile) == 0) and not os.path.isfile('LU.locked'): # test needs to be run
			open('LU.locked', 'w').close()
			open( datafile + 'locked', 'w').close()
			writeLUdat(map(str, M[0]), map(str, B[0]))
			os.system('aprun -n 16 -N 8 -S 2 -ss ../../xdlu > ' + outfile + ' 2> '  + errfile)
			perf, timespent = parseTest(outfile, datafile)
			os.remove(datafile + 'locked')
			os.remove('LU.locked')
		else: # something funny  
			print filehandle
			raise Exception('something is funky,  check above test')
								
		os.chdir('..')
    return perf, timespent



def myPredict(F, featurespace):
    mu = []
    sig = []
    for i in range(featurespace.shape[0]):
        x = featurespace[i, :]
        treepreds = NaN*np.zeros((1, NTREES))
        for tree in range(F.n_estimators):
            treepreds[0,tree] = F.estimators_[tree].predict(x)

        mu.append(np.mean(treepreds))
        sig.append(np.sqrt(np.var(treepreds)))

    return np.array(mu), np.array(sig)


    
def myRandI(high, numSelect):
    ints = np.unique(np.random.randint(0, high, size=numSelect))
    while ints.size <  numSelect:
        ints = np.unique(np.concatenate((ints, np.random.randint(0,high, size=1)),axis=0))
    return ints


def chooseM(F, Mfunc, inputRanges):
    M, predsM, stdevM = Mfunc(F, inputRanges)
    return M, predsM, stdevM


def unifRandM(F, inputRanges):
    #% randomly choose an input value for every input coordinate
    Mvalue = np.random.randint( MATMIN, high = MATMAX, size = (1,2))

    if F == []:
        predsM = []
        stdevM = []
    else:
        featurespace = np.concatenate((np.tile(Mvalue, (SIZESPACE, 1)), fullspace), axis = 1)
        predsM, stdevM = myPredict(F, featurespace) #debugging

    return Mvalue, predsM, stdevM


# %%% --- M we are most uncertain of --- %%%
def treeM(F, inputRanges):
    if F == []:
        # % choose a random M
        Mvalue, predsM, stdevM = unifRandM(F, inputRanges)
    else:
        # % variance in the prediction for the best prediction hat(B)
        maxstdev = 0;
        # for m = 1:NUMMs
        for m in range(NUMMs):
            # %choose a random M
            #% randomly choose an input value for every input coordinate
            tmpM = np.random.randint( MATMIN, high = MATMAX, size = (1,2))
            featurespace = np.concatenate((np.tile(tmpM, (SIZESPACE, 1)), fullspace), axis = 1)
            preds, stdev = myPredict(F, featurespace)
            
            # % check predictions on that M
            i = np.argmax(preds) #debugging, only the first occurance in returned...
            
            # % keep M if there is high variance in the prediction of its max
            if stdev[i] >= maxstdev: #debugging - I think this should be strict inequality.
                predsM = preds
                stdevM = stdev 
                Mvalue = tmpM
                maxstdev = stdev[i]
    return Mvalue, predsM, stdevM



def chooseB(funcB, predsM, stdevM):
    Bs = funcB(predsM, stdevM); 
    # Bs = np.array([0,0,0]).reshape(1, 3)
    return Bs

def exhaustiveB(predsM, stdevM):
    return fullspace

def randomB(predsM, stdevM):
    temp = myRandI(SIZESPACE, BSUBSETSIZE)
    Bs = fullspace[temp,:]
    return Bs

def treeB(predsM, stdevM):
    if predsM == []:
        Bs = randomB(predsM, stdevM)
    else:
        #% choose point that has most uncertainty
        # [~, ind] = sort(stdevM, 'descend')
        ind = np.argsort(stdevM) #ascending
        ind = ind[::-1]
        Bs = fullspace[ind[1:BSUBSETSIZE],:]
    return Bs

def treeucbB(predsM, stdevM):
    if predsM == []:
        Bs = randomB(predsM, stdevM)
    else:
        # % choose point trades off exploration/exploitation
        # [~, ind] = sort(predsM + stdevM, 'descend'); 
        # Bs = fullspace(ind(1:BSUBSETSIZE),:);  
        ind = np.argsort(predsM + stdevM) #ascending
        ind = ind[::-1]
        Bs = fullspace[ind[1:BSUBSETSIZE],:]
    return Bs

def measureMethodPerf(F, testdata):
    numTests = len(testdata['maxperf'])

    temp = NaN*np.zeros((1,numTests))
    time = np.zeros(1)
    for i in range(numTests):
       # % predict which one to run on 
       nBs = fullspace.shape[0]
       X = np.concatenate((np.tile(testdata['Mvalue'][i], (nBs, 1)), fullspace),axis=1)
       preds, discard = myPredict(F, X)
       ind = np.argmax(preds) #debugging: need to randomize if multiple values achieve max
        
       # % look up how config would have performed
       perffrompred = testdata['perf'][i][ind]
       
       # % look up time 
       time = time[0] + testdata['time'][i][ind]
       
       # % compare with true best
       temp[0,i] = (perffrompred - testdata['minperf'][i])/(testdata['maxperf'][i]  - testdata['minperf'][i] ); 
#        temp(i) = (perfOfPred - trueMin)/(trueMax - trueMin); 
    flopsHat = np.mean(temp)
    return flopsHat, time



def runSearch(testdata, Mfunc, Bfunc):
    #global MAXOBS NUMPARAMS NTREES treeOptions
    F = []
    # observed = NaN* zeros(MAXOBS, length(inputRanges) + NUMPARAMS + 1);
    # methodPerf.numtried = NaN * zeros(1, MAXOBS); 
    observed = NaN*np.zeros((MAXOBS, len(inputRanges) + NUMPARAMS + 1))
    methodPerf= {'numtried': []}
    methodPerf['numtried'] = NaN*np.zeros((1, MAXOBS))
    methodPerf['flopsHat'] = NaN*np.zeros((1, MAXOBS))
    methodPerf['timespent'] = NaN*np.zeros((1, MAXOBS))
    methodPerf['timetraining'] = NaN*np.zeros((1, MAXOBS))
    #% methodPerf.configstried = NaN * zeros( MAXOBS, length(inputRanges)  + NUMPARAMS);

    numObs = 0
    iter = 0 
    while numObs < MAXOBS:
        
        #% return M and predictions for entire M space
        M, predsM, stdevM = chooseM(F, Mfunc, inputRanges)
        
        #% these are the new configurations that we are going to try
        Bs = chooseB(Bfunc, predsM, stdevM)
        
        #% query function to get new data
        perf, timespent = runTests(M, Bs)
        nBs = Bs.shape[0]
        temp = np.concatenate((np.tile(M, (nBs, 1)), Bs), axis=1)
        if numObs+nBs > MAXOBS:
            newObserved = NaN*np.zeros((numObs + nBs, len(inputRanges) + NUMPARAMS + 1))
            newObserved[0:numObs, :] = observed[0:numObs, :]
            newObserved[numObs : numObs + nBs,:] = np.concatenate((temp, perf), axis=1)
            observed = newObserved
        else:
             observed[numObs : numObs + nBs,:] =  np.concatenate((temp, perf), axis=1)
        
        numObs = numObs + nBs; 
        
        #% update model
        F = clone(model)
        F.fit(observed[0:numObs, 0:len(inputRanges) + NUMPARAMS], np.ravel(observed[0:numObs, len(inputRanges) + NUMPARAMS]))
        # F = TreeBagger(NTREES,observed(:, 1:end-1), observed(:, end), 'method', 'regression', 'Options', treeOptions); 
        #%F = TreeBagger(NTREES,observed(:, 1:end-1), observed(:, end),'method', 'regression'); 

        #% update stats on search
        methodPerf['numtried'][0,iter] = nBs 
        methodPerf['flopsHat'][0,iter], methodPerf['timespent'][0,iter] = measureMethodPerf(F, testdata) #*** debugging
        methodPerf['timetraining'][0,iter] = np.sum(timespent)
        print 'measured time training'
        iter = iter + 1; 
    return methodPerf   



MATMIN = 1000
MATMAX = 8000

inputRanges = [range(MATMIN,MATMAX), range(MATMIN,MATMAX)]
paramRanges=[[8, 16, 24, 32, 40, 48, 56, 64],[1, 2, 4, 8, 16], [16, 8, 4, 2, 1]]
NUMPARAMS = len(paramRanges)


fullspace = SpecialEnumerateSearchSpace(paramRanges)
SIZESPACE = np.size(fullspace, axis = 0) 
NaN = np.nan

MAXOBS = 800 # maximum number of observations
NUMMs = 500 # number of random M's that we consider in treeM
NTREES = 15 
model = RandomForestRegressor(n_estimators=NTREES)

BSUBSETSIZE = 5 
SIZETESTSET = 25 




