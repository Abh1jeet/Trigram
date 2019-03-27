import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import ngrams,FreqDist
import matplotlib.pyplot as plt 
import codecs

def tokenization(data):
	sentence=sent_tokenize(data)
	lineNo=1
	sentdata=[]
	for i in sentence:
		tokens=word_tokenize(i)
		#Remove single-character tokens (mostly punctuation)
		tokens = [word for word in tokens if len(word) > 1]
		tokens = [word.lower() for word in tokens]
		tokens = [word for word in tokens if word not in ['``' ,"''","'s" ] ]
		#adding start and end token to data
		tokens.insert(0,'<s>')
		tokenLen=len(tokens)
		tokens.insert(tokenLen,'</s>')
		for i in tokens:
			sentdata.append(i)

		lineNo=lineNo+1
	
	return sentdata


def unigramCalc(data):
	unigram=ngrams(data,1)
	freqDistUnigram=FreqDist(unigram)
	unigramCount={}
	for k,v in freqDistUnigram.items():     
    		unigramCount[k[0]]=v
	
	return unigramCount

def bigramCalc(data):
	bigram=ngrams(data,2)
	freqDistBigram=FreqDist(bigram)
	bigramCount={}
	for k,v in freqDistBigram.items():     
		bigramCount[k[0] +" "+k[1]]=v
	
	return bigramCount

def trigramCalc(data):
	trigram=ngrams(data,3)
	freqDistTrigram=FreqDist(trigram)
	trigramCount={}        
	for k,v in freqDistTrigram.items():
		trigramCount[k[0] +" "+ k[1] +" "+ k[2]]=v

	return trigramCount	

def backOff(unigramCount,bigramCount,trigramCount,testToken,N,vocab):

	perplexity = float(1.0)
    
	#probability calulation using backoff
	testLen=len(testToken)
	for i in range(0,testLen):
		if i==0:
			prob=(unigramCount[testToken[i]])*1.0/N
			#print(testToken[i],prob)
		elif i==1:
			str=testToken[i-1]+" "+ testToken[i]
			prob=(bigramCount[str])*1.0/unigramCount[testToken[i-1]]
			#print(str,prob)
		else:
			str=testToken[i-2]+" "+ testToken[i-1]+" "+testToken[i]
			#check for trigram
			if str not in trigramCount:
				#check for bigram
				str2=testToken[i-1]+" "+testToken[i]
				if str2 not in bigramCount:
					#check for unigram
					str3=testToken[i]
					if str3 not in unigramCount:
						#OOV
						prob=(vocab['<UNK>'])*1.0/N
					else:
						prob=(unigramCount[str3])*1.0/N
				else:
					prob=(bigramCount[str2])*1.0/unigramCount[testToken[i-1]]
			else:
				str2=testToken[i-2]+" "+testToken[i-1]
				prob=(trigramCount[str])*1.0/bigramCount[str2]
			#print(str,prob)
		perplexity = perplexity * ( prob**(1./N))
			#print(str,trigramCount[str],prob)
				#print(str,prob)
	print("using backoff :", 1/perplexity)

def backOffADD1(unigramProbDictAdd1,bigramProbDictAdd1,trigramProbDictAdd1,testToken,N,vocab,unigramCount,bigramCount,trigramCount,t):

	perplexity = float(1.0)
    
	#probability calulation using backoff
	testLen=len(testToken)
	for i in range(0,testLen):
		if i==0:
			prob=(unigramCount[testToken[i]])*1.0/N
			#print(testToken[i],prob)
		elif i==1:
			str=testToken[i-1]+" "+ testToken[i]
			prob=(bigramCount[str])*1.0/unigramCount[testToken[i-1]]
			#print(str,prob)
		else:
			str=testToken[i-2]+" "+ testToken[i-1]+" "+testToken[i]
			#check for trigram
			if str not in trigramCount or trigramCount[str]<t:
				#check for bigram
				str2=testToken[i-1]+" "+testToken[i]
				if str2 not in bigramCount or bigramCount[str2]<t:
					#check for unigram
					str3=testToken[i]
					if str3 not in unigramCount:
						#OOV
						prob=(1*1.0)/V
					else:
						prob=unigramProbDictAdd1[str3]
				else:
					prob=bigramProbDictAdd1[str2]
			else:
				prob=trigramProbDictAdd1[str]
			#print(str,prob)
		perplexity = perplexity * ( prob**(1./N))
			#print(str,trigramCount[str],prob)
				#print(str,prob)
	print("using backoff ADD1:", 1/perplexity)



def interpolation(unigramCount,bigramCount,trigramCount,testToken,N,vocab):

	l1=0.6
	l2=0.3
	l3=0.1
	perplexity=float(1.0)
	testLen=len(testToken)
	for i in range(0,testLen):
		if i==0:
			prob=unigramCount[testToken[i]]*1.0/N
			#print(testToken[i],prob)
		elif i==1:
			str2=testToken[i-1]+" "+testToken[i]
			prob=bigramCount[str2]*1.0/unigramCount[testToken[i-1]]
			#print(str2,prob)
		else:
			str2=testToken[i-1]+" "+testToken[i]
			str3=testToken[i-2]+" "+testToken[i-1]+" "+testToken[i]
			if str3 in trigramCount:
				trigramProb=trigramCount[str3]*1.0/bigramCount[str2]
			else:
				trigramProb=0
			if str2 in bigramCount:
				bigramProb=bigramCount[str2]*1.0/unigramCount[testToken[i-1]]	
			else:
				bigramProb=0
			if testToken[i] in unigramCount:
				unigramProb=unigramCount[testToken[i]]*1.0/N
			else:
				unigramProb=unigramCount['<UNK>']*1.0/N
			prob=l1*trigramProb + l2*bigramProb +l3*unigramProb
			#print(str3 ,prob)
		perplexity = perplexity * ( prob**(1./N))

	print("using interpolation :", 1/perplexity)

def interpolationADD1(unigramProbDictAdd1,bigramProbDictAdd1,trigramProbDictAdd1,testToken,N,vocab,unigramCount,bigramCount,trigramCount):
	V=len(vocab)+1
	l1=0.6
	l2=0.3
	l3=0.1
	perplexity=float(1.0)
	testLen=len(testToken)
	for i in range(0,testLen):
		if i==0:
			prob=unigramProbDictAdd1[testToken[i]]
			#print(testToken[i],prob)
		elif i==1:
			str2=testToken[i-1]+" "+testToken[i]
			prob=bigramProbDictAdd1[str2]
			#print(str2,prob)
		else:
			str2=testToken[i-1]+" "+testToken[i]
			str3=testToken[i-2]+" "+testToken[i-1]+" "+testToken[i]
			if str3 in trigramCount:
				trigramProb=trigramProbDictAdd1[str3]
			else:
				if str2 in bigramCount:
					trigramProb=(1*1.0)/(bigramCount[str2]+V)
				else:
					trigramProb=(1*1.0)/V
			if str2 in bigramCount:
				bigramProb=bigramProbDictAdd1[str2]	
			else:
				if testToken[i] in unigramCount:
					bigramProb=(1*1.0)/(unigramCount[testToken[i]]+V)
				else:
					bigramProb=(1*1.0)/V

			if testToken[i] in unigramCount:
				unigramProb=unigramProbDictAdd1[testToken[i]]
			else:
				unigramProb=vocab['<UNK>']*1.0/N
			prob=l1*trigramProb + l2*bigramProb +l3*unigramProb
			#print(str3 ,prob)
		perplexity = perplexity * ( prob**(1./N))

	print("using interpolation ADD1", 1/perplexity)


## Regression related stuff
#calculate best fit line for simple regression 
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style

#finds the slope for the best fit line
def findBestFitSlope(x,y):
    m = (( mean(x)*mean(y) - mean(x*y) ) / 
          ( mean(x)** 2 - mean(x**2)))

    return m
      
#finds the intercept for the best fit line
def findBestFitIntercept(x,y,m):
    c = mean(y) - m*mean(x)
    return c


def unigramFreqCountCalc(unigramCount,N):
	unigramFreqCount={}
	for i in range(1,N):
		unigramFreqCount[i]=0
	for k,v in unigramCount.items():
		unigramFreqCount[v]=unigramFreqCount[v]+1
	data_pts = {}
	i=0
	xc=[]
	yc=[]
	for k,v in unigramFreqCount.items():
		if v not in data_pts:
			if v!=0:
				data_pts[v]=k
				xc.append(v)
				yc.append(k)
				i=i+1
				if i>50:
					break

	x = np.array(xc, dtype = np.float64)
	y = np.array(yc , dtype = np.float64)

	#now do regression
	#find the slope and intercept for the regression equation
	slope_m = findBestFitSlope(x,y)
	intercept_c = findBestFitIntercept(x,y,slope_m)

	#now find the missing Nc terms and give them value using regression
	for k,v in unigramFreqCount.items():
		if v==0:
			unigramFreqCount[k] = (slope_m*i) + intercept_c

	return unigramFreqCount

def bigramFreqCountCalc(bigramCount,N):
	bigramFreqCount={}
	for i in range(1,N):
		bigramFreqCount[i]=0
	for k,v in bigramCount.items():
		bigramFreqCount[v]=bigramFreqCount[v]+1
	data_pts = {}
	i=0
	xc=[]
	yc=[]
	for k,v in bigramFreqCount.items():
		if v not in data_pts:
			if v!=0:
				data_pts[v]=k
				xc.append(v)
				yc.append(k)
				i=i+1
				if i>50:
					break

	x = np.array(xc, dtype = np.float64)
	y = np.array(yc , dtype = np.float64)

	#now do regression
	#find the slope and intercept for the regression equation
	slope_m = findBestFitSlope(x,y)
	intercept_c = findBestFitIntercept(x,y,slope_m)

	#now find the missing Nc terms and give them value using regression
	for k,v in bigramFreqCount.items():
		if v==0:
			bigramFreqCount[k] = (slope_m*i) + intercept_c

	return bigramFreqCount
def trigramFreqCountCalc(trigramCount,N):
	trigramFreqCount={}
	for i in range(1,N):
		trigramFreqCount[i]=0
	for k,v in trigramCount.items():
		trigramFreqCount[v]=trigramFreqCount[v]+1
	data_pts = {}
	i=0
	xc=[]
	yc=[]
	for k,v in trigramFreqCount.items():
		if v not in data_pts:
			if v!=0:
				data_pts[v]=k
				xc.append(v)
				yc.append(k)
				i=i+1
				if i>50:
					break

	x = np.array(xc, dtype = np.float64)
	y = np.array(yc , dtype = np.float64)

	#now do regression
	#find the slope and intercept for the regression equation
	slope_m = findBestFitSlope(x,y)
	intercept_c = findBestFitIntercept(x,y,slope_m)

	#now find the missing Nc terms and give them value using regression
	for k,v in trigramFreqCount.items():
		if v==0:
			trigramFreqCount[k] = (slope_m*i) + intercept_c

	return trigramFreqCount
    


def interpolationGT(unigramCount,bigramCount,trigramCount,testToken,N,vocab):
		k=N
		V=len(vocab)
		triNc=trigramFreqCountCalc(trigramCount,N)
		biNc=bigramFreqCountCalc(bigramCount,N)
		uniNc=unigramFreqCountCalc(unigramCount,N)
		l1=0.6
		l2=0.3
		l3=0.1
		testLen=len(testToken)
		perplexity=float(1.0)
		for i in range(2,testLen):
			str=testToken[i-2]+" "+testToken[i-1]+" "+testToken[i]
			if str in trigramCount:
				count=((trigramCount[str]+1)*(triNc[trigramCount[str]+1]))*1.0/(triNc[trigramCount[str]])
			else:
				count=triNc[1]
			triProb=(count*1.0)/N
			str2=testToken[i-1]+" "+testToken[i]
			if str2 in bigramCount:
				count=((bigramCount[str2]+1)*(biNc[bigramCount[str2]+1]))*1.0/(biNc[bigramCount[str2]])
			else:
				count=biNc[1]
			biProb=(count*1.0)/N
			if testToken[i] in unigramCount:
				count=((unigramCount[testToken[i]]+1)*(uniNc[unigramCount[testToken[i]]+1]))*1.0/(uniNc[unigramCount[testToken[i]]])
			else:
				count=uniNc[1]
			uniProb=(count)*1.0/N
			prob=l1*triProb+l2*biProb+l3*uniProb
			perplexity = perplexity * ( prob**(1.0/N))

		print("interpolation GT:",1/perplexity)

def backOffGT(unigramCount,bigramCount,trigramCount,testToken,N,vocab,t):
		k=N
		V=len(vocab)
		triNc=trigramFreqCountCalc(trigramCount,N)
		biNc=bigramFreqCountCalc(bigramCount,N)
		uniNc=unigramFreqCountCalc(unigramCount,N)
		testLen=len(testToken)
		perplexity=float(1.0)
		for i in range(2,testLen):
			str=testToken[i-2]+" "+testToken[i-1]+" "+testToken[i]
			if str in trigramCount:
				count=((trigramCount[str]+1)*(triNc[trigramCount[str]+1]))*1.0/(triNc[trigramCount[str]])
			else:
				str2=testToken[i-1]+" "+testToken[i]
				if str2 in bigramCount:
					count=((bigramCount[str2]+1)*(biNc[bigramCount[str2]+1]))*1.0/(biNc[bigramCount[str2]])
				else:
					if testToken[i] in unigramCount:
						count=((unigramCount[testToken[i]]+1)*(uniNc[unigramCount[testToken[i]]+1]))*1.0/(uniNc[unigramCount[testToken[i]]])
					else:
						count=uniNc[1]
			prob=(count)*1.0/N
			perplexity = perplexity * ( prob**(1.0/N))

		print("backOFF GT:",1/perplexity)




def unigramProbCalcAdd1(unigramCount,bigramCount,trigramCount,vocab,N):
	V=len(vocab)+1
	unigramProbDict={}
	for unigram in unigramCount:
		prob = ( unigramCount[unigram]+1 )*1.0 / (N+V)
		unigramProbDict[unigram]=prob
	return unigramProbDict

def bigramProbCalcAdd1(unigramCount,bigramCount,trigramCount,vocab):
	V=len(vocab)+1
	bigramProbDict={}
	for bigram in bigramCount:
		words=bigram.split()
		prob = ( bigramCount[bigram]+1)*1.0 / ( unigramCount[words[0]] + V)
		bigramProbDict[bigram]=prob
	return bigramProbDict

def trigramProbCalcAdd1(unigramCount,bigramCount,trigramCount,vocab):
	V=len(vocab)+1
	trigramProbDict={}
	for trigram in trigramCount:
		words=trigram.split()
		bigram=words[0]+" "+words[1]
		prob=(trigramCount[trigram]+1)*1.0/(bigramCount[bigram]+V)
		trigramProbDict[trigram]=prob
	return trigramProbDict



filePath="D:\\mtech2\\isi\\assignment2\\finalTrain.txt"
#opening file 
#f=codecs.open(filePath,'rU')

import io
with io.open(filePath, "r", encoding="utf-8") as my_file:
     raw = my_file.read() 


#reading the file
#raw=f.read()

#getting test and training data
data=tokenization(raw)
unigramCount=unigramCalc(data)
bigramCount=bigramCalc(data)
trigramCount=trigramCalc(data)
#building vocablury
vocab=unigramCount
vocab['<UNK>']=1
V=len(vocab)		#size of vocablury
N=len(data)			#size of data




unigramProbDictAdd1=unigramProbCalcAdd1(unigramCount,bigramCount,trigramCount,vocab,N)
bigramProbDictAdd1=bigramProbCalcAdd1(unigramCount,bigramCount,trigramCount,vocab)
trigramProbDictAdd1=trigramProbCalcAdd1(unigramCount,bigramCount,trigramCount,vocab)
#unigramFreqCount=unigramFreqCountCalc(unigramCount)
#bigramFreqCount=bigramFreqCountCalc(bigramCount)
#trigramFreqCount=trigramFreqCountCalc(trigramCount)


#testing data
filePath="D:\\mtech2\\isi\\assignment2\\finalTest.txt"
#f=open(filePath,'rU')
#raw=f.read()

with io.open(filePath, "r", encoding="utf-8") as my_file:
     raw = my_file.read() 

testToken=tokenization(raw)


#perplexity using simple interpolation
interpolation(unigramCount,bigramCount,trigramCount,testToken,N,vocab)


#smoothing
#perplexity using interpolation using ADD1
interpolationADD1(unigramProbDictAdd1,bigramProbDictAdd1,trigramProbDictAdd1,testToken,N,vocab,unigramCount,bigramCount,trigramCount)


#perplexity using interpolation using ADD1
interpolationGT(unigramCount,bigramCount,trigramCount,testToken,N,vocab)

#perplexity using simple backoff
backOff(unigramCount,bigramCount,trigramCount,testToken,N,vocab)

backOffADD1(unigramProbDictAdd1,bigramProbDictAdd1,trigramProbDictAdd1,testToken,N,vocab,unigramCount,bigramCount,trigramCount,4)

backOffGT(unigramCount,bigramCount,trigramCount,testToken,N,vocab,2)