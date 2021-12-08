# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 15:00:20 2021

@author: Thijs

"""
import pandas as pd
import math, os
os.chdir("C:/Users/Thijs/Desktop/Erasmus/Master/Periode 2/Computer Science/Paper")
import json #https://realpython.com/python-json/
import re
import numpy as np
import time
from sklearn.utils import shuffle
import sys
from itertools import chain, combinations
import random
#%% parameters
#split dataset
split_idx = 1262

# Keyreading
keyreadingthreshold = 0.8
# ModelWordChoice
minimumlengthofword = 0
useKVP              = True
input2modelwords    = True
brandweight         = 2
midweight           = 3

#LSH
rowsLSH             = 9
repsLSH             = 2#50 (0.024) 60 (0.02) 
frac                = 0.5
#LSH Performance
beta = 3
# clustering performance
epsilon = 0.21

#%% Load data and create dataset
print('Load data and create dataset ...')

#Load data
with open("TVs-all-merged.json", "r") as data:
    DataF = json.load(data)
split_idx = split_idx
Data = dict(list(DataF.items())[:split_idx])
DataT = dict(list(DataF.items())[split_idx:])

#%% cleaning
def replaceunit(text):
    """
    Parameters
    ----------
    text : String
        Input string possibly containing a units representation.

    Returns
    -------
    text : String
        Output string replaced units for one common unit
    """
    hz = [r' hz', 'hz',' HZ.',' Hz','Hz.','Hz']
    for item in hz:
        text = text.replace(item, 'hz ')
        
    inch = [r'inches',"'",'”','in',' inch', ' inches','Inches',' Inches','-Inch','-Inches','-inch','-inches', '"','""']
    for item in inch:
        text = text.replace(item, 'inch ')
    
    lb = [r' lb', ' lbs.',' lb.',' pounds','pounds','lb','lbs.','lb.','lb']
    for item in lb:
        text = text.replace(item, 'lb ')
    
    cdma = [r' cd/mâ²',' cdm2','cdm2','lm',' lm',' cd/m²','cd/m²',' cd/m2','nit']
    for item in cdma:
        text = text.replace(item, 'cdma ')
    
    watt = [r' w','w',' watt','watt']
    for item in watt:
        text = text.replace(item, 'watt ')
            
    p = [r' p','p','i/p',' i/p','/24p']
    for item in p:
        text = text.replace(item, 'p ' )
    
    kg = [r' kg','kg','KG',' KG','Kg']
    for item in kg:
        text = text.replace(item, 'kg ')
        
        
    return text
def cleantitle(text):
    text = replaceunit(text)
    text = text.replace(')','')
    text = text.replace('(','')
    text = text.replace(']','')
    text = text.replace('[','')
    text = text.replace('/','')
    text = text.replace('.0','')
    text = text.replace(',','')
    text = text.replace('inchwatt','inch')


    if input2modelwords == True:
        text = ' '.join(w for w in text.split() if any(x.isdigit() for x in w))

    return text
def cleanvalue(text):
    text = text.lower()
    text = text.replace('+','')
    text = text.replace('-','')
    text = text.replace('without','-')
    text = text.replace('with','+')

    
    text = replaceunit(text)
    text = text.replace('and',' ')
    text = text.replace('|',' ')
    text = text.replace(' x ','x')
    text = text.replace('no','0')
    text = text.replace('yes','1')
    text = text.replace('false','0')
    text = text.replace('true','1')
    text = text.replace(',','')
    text = text.replace('.','')
    text = text.replace(')','')
    text = text.replace('(','')
    text = text.replace('/','')
    text = text.replace('+','')
    text = text.replace('-','')
    text = text.replace('&#','')

    return text
def cleanshop(text):
    text = text.lower()
    text = text.replace('.','')
    text = text.replace(' ','')
    return text
def cleanbrand(text):
    text = text.lower()
    return text  
def cleanvaluereading(text):
    """
    Ment for the values of the key value pairs.
    
    Parameters
    ----------
    text : String
        Input string that cointains possible special caracters.

    Returns
    -------
    String that is cleaned.
    """
    text = text.lower()
    text = text.replace('+','')
    text = text.replace('-','')
    text = text.replace('without','-')
    text = text.replace('with','+')
    text = replaceunit(text)
    text = text.replace(' and ',' ')
    text = text.replace('|',' ')
    text = text.replace(' x ','x')
    text = text.replace('no','0')
    text = text.replace('yes','1')
    text = text.replace('false','0')
    text = text.replace('true','1')
    text = text.replace(',','')
    text = text.replace('.','')
    text = text.replace(')','')
    text = text.replace('(','')
    text = text.replace('/','')
    text = text.replace('+','')
    text = text.replace('-','')
    text = text.replace('&#','')
    
    return text 
#%% Data Read
if useKVP == True:
# Listing values of kvp's
    keyvaluepairs = {}
    for key in Data.keys():
        for i in range(len(Data[key])):
            for k in Data[key][i]['featuresMap'].keys():
                if k not in keyvaluepairs.keys():
                    keyvaluepairs[k] = [cleanvaluereading(Data[key][i]['featuresMap'][k])]
                else: 
                    keyvaluepairs[k].append(cleanvaluereading(Data[key][i]['featuresMap'][k]))
                
                        
    shortlistkeys = {k:v for k,v in keyvaluepairs.items() if len(v) > 400}
# Reading and matching keys
    from difflib import SequenceMatcher
    import operator
    
    def similar(a, b):
        return 0.6*SequenceMatcher(None, a, b).ratio()+0.4*mysim(a.split(),b.split())
    
    def mysim(a,b):
        s1 = set(a)
        s2 = set(b)
        if min(len(s1),len(s2)) == 0:
            return 0.0
        else:
            return len(s1.intersection(s2))/min(len(s1),len(s2))
    
    keyfeatures = {}
    for key in Data.keys():
        for i in range(len(Data[key])):
            for k in Data[key][i]['featuresMap'].keys():
                if k not in keyfeatures.keys():
                    keyfeatures[k] = 1
                else:
                    keyfeatures[k] += 1
    keyfeatures = {k:v for k,v in keyfeatures.items() if v > 1}
    sorted_x = sorted(keyfeatures.items(), key=operator.itemgetter(1),reverse=True)
    sth = keyreadingthreshold
    sim = {}
    sim[sorted_x[0][0]] = []
    th = sth
    for i in range(1,len(sorted_x)):
        j = 0
        while similar(sorted_x[i][0],sorted_x[j][0]) < th:
            j += 1
        if j == i:
            sim[sorted_x[i][0]] = []
        else: 
            if sorted_x[j][0] in sim.keys(): 
                sim[sorted_x[j][0]].append(sorted_x[i][0]) 
                th = sth
            else:
                th -= 0.1
                i  -= 1
    simcut = {k:v for k,v in sim.items() if len(v) >= 1}

        
def replacekeys(text):
    for key in simcut.keys():
        for value in simcut[key]:
            text = text.replace(value,key)
    return text

def cleankey(text):
    if useKVP == True:
        text = replacekeys(text)
    text = text.lower()
    text = text.replace(' ','')
    return text

# Model ID
def findmodelID(text):
    text = text.replace('(','')
    text = text.replace(')','')

    textlist = ' '.join(w for w in text.split() if any(x.isdigit() for x in w)).split()
    maxlenword = ''
    for word in textlist:
        len1 = len(maxlenword)
        len2 = len(word)
        if len2 > len1:
            maxlenword = word
    if len(maxlenword) < 6:
        return 'None'
    else:
        return maxlenword
        
    
    
    
#%% Fill Dataset:
    
shop1 = 'amazon.com'
shop2 = 'newegg.com'
shop3 = 'bestbuy.com'
shop4 = 'thenerds.net'
shops = [shop1,shop2,shop3,shop4]
cleanShops = [cleanshop(item) for item in shops]




brandnames = {}
for key in Data.keys():
    for i in range(len(Data[key])):
        for k in Data[key][i]['featuresMap'].keys():
            if k == 'Brand' or k == 'Brand Name' or k == 'Brand Name:':
                if Data[key][i]['featuresMap'][k] not in brandnames.keys():
                    brandnames[Data[key][i]['featuresMap'][k]] = 1
                else:
                    brandnames[Data[key][i]['featuresMap'][k]] += 1
                

DListOfBrands = brandnames.keys()
ListOfBrands = [cleanbrand(item) for item in DListOfBrands]
        

# Create Dataset and cleaning
DataSet = pd.DataFrame(columns = ['key','potmodelID','title','shop','kvp'])
keys = []
titles = []
shop = []
kvps = []
potmodelID = []
for key in Data.keys():
    for i in range(len(Data[key])):
        keys.append(key)
        potmodelID.append(findmodelID(Data[key][i]['title']))
        titles.append(cleantitle(Data[key][i]['title']).split())
        shop.append(cleanshop(Data[key][i]['shop']))
        kvpi = []
        for kvp in Data[key][i]['featuresMap']:
            if kvp == 'Brand' or kvp == 'Brand Name' or kvp == 'Brand Name:':
                value = cleanbrand(Data[key][i]['featuresMap'][kvp])
                kvpi.append(value)
            if useKVP == True:
                if replacekeys(kvp) not in simcut.keys():
                    continue
                else:
                    k = cleankey(kvp)
                    value = cleanvalue(Data[key][i]['featuresMap'][kvp])
                    kvpi.append(k+':'+value)
        kvps.append(kvpi)
            
        
DataSet['key'] = keys
DataSet['potmodelID']  = potmodelID

DataSet['title']  = titles
DataSet['shop']  = shop
DataSet['kvp']  = kvps

  
    

del keys, titles, shop, kvps

#%% Functions

def progressBar(name, value, endvalue, bar_length = 50, width = 20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent*bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(\
                         name, width, arrow + spaces, int(round(percent*100))))
            
def findrbp(n,r):
    b = 1
    while r*b <= n*2/3:
        b += 1
    p = r*b
    return p,r,b
#%% WordCount
WordCount = {}

    
for title in DataSet['title']:
    for word in title:
        if word not in WordCount.keys():
            WordCount[word] = 1
        else:
            WordCount[word] += 1
del word, title
    
if useKVP == True:
    for kvp in DataSet['kvp']:
        for word in kvp:
            if word not in WordCount.keys():
                WordCount[word] = 1
            else:
                WordCount[word] += 1
    del word, kvp
    
#%% remove words
wordstoremove = cleanShops
for item in wordstoremove:
    if item in WordCount.keys():
        del WordCount[item] 

del item, wordstoremove
        
#%% remove  words from wordcount      
WordCount = {k:v for k,v in WordCount.items() if v >=2}
inputWords = list(WordCount.keys())
for i in range(brandweight-1):
    for brand in ListOfBrands:
        inputWords.append(brand)
for i in range(midweight-1):
    for mid in potmodelID:
        inputWords.append(mid)    

#%% creating input matrix
print('Number of model words used:  '+ str(len(inputWords)))

products = DataSet.index
inputMatrix = np.zeros((len(inputWords),len(products)))#pd.DataFrame(index = modelWords,columns = products)

for p in range(inputMatrix.shape[1]):
    progressBar("Creating input matrix    ",p, inputMatrix.shape[1]-1)
    title = DataSet['title'][p]
    kvp   = DataSet['kvp'][p]
    for word_i in range(inputMatrix.shape[0]):
        if inputWords[word_i] in title: 
            inputMatrix[word_i][p] = 1   
del kvp, title, word_i
#%% Creating signature matrix

def isPrime(x):
  for j in range(2,int(x**0.5)+1):
    if x%j==0:
      return False
  return True

def findPrimeNum(num):
  for i in range(num,10000,1):
    if isPrime(i):
      return i
r = rowsLSH
(permutations,rows,bands) = findrbp(len(DataSet),r)
print('')
print('Number of permutations    :  '+ str(permutations))
th = pow((1/bands),(1/rows))

SignatureMatrix = np.ones((permutations,len(products)))*1000000
a = random.sample(range(inputMatrix.shape[0]),permutations)
b = random.sample(range(inputMatrix.shape[0]),permutations)
def hash(a,b,r):
    return (int(a*r+b))%findPrimeNum(permutations)
def hash_factory(n):
    return lambda x: hash(a[n],b[n],x)


hashes = [hash_factory(_) for _ in range(permutations)] # a list 8 hash functions

for r in range(inputMatrix.shape[0]):
    progressBar("Creating signature matrix",r, inputMatrix.shape[0] - 1)
    hashvaluesr = []
    for permutation in range(permutations):
        hashvaluesr.append(hashes[permutation](r))
    for p in range(inputMatrix.shape[1]):
        if inputMatrix[r][p] == 1:
            for permutation in range(permutations):
                SignatureMatrix[permutation][p] = min(hashvaluesr[permutation],SignatureMatrix[permutation][p])
   
#%% LSH
reps = repsLSH
buckets = {}
print('')
print("LSH threshold is          :  "+ str(round(th,2)))
for r in range(reps):
    shuff = SignatureMatrix.copy()
    np.random.shuffle(shuff)
    for p in range(shuff.shape[1]):
        for b in range(bands):
            h = str(reps) +' '+ str(b) +' '+(str([round(item) for item in shuff[:,p][b:b+rows]]))
            if h not in buckets.keys():
                buckets[h] = [p]
            else:
                if p not in buckets[h]:
                    buckets[h].append(p)
                

buckets = {k:v for k,v in buckets.items() if len(v) >= 2}

del reps, p, b, h, shuff, r
#%%find potential pairs from potmodelID
potmodelIDpairs = []
for i in list(DataSet.index):
    for j in list(DataSet.index):
        if i >= j:
            continue
        else:
            if DataSet['potmodelID'][i] == DataSet['potmodelID'][j] and DataSet['potmodelID'][i] != 'None' :
                pair = tuple([i,j])
                potmodelIDpairs.append(pair)
#%% Define candidate pairs
candidates = {}
for key in buckets.keys():
    for comb in list(combinations(buckets[key],2)):
        if comb[0]> comb[1]:
            comb = tuple([comb[1],comb[0]])
        if comb not in candidates.keys():
            candidates[comb] = 1
        else:
            candidates[comb] += 1
        
        


del key, comb

#%% remove same shop and different brand 
begin = len(candidates.keys())
del_keys = []
for key in candidates.keys():
    i = key[0]
    j = key[1]
    if DataSet['shop'][i] == DataSet['shop'][j]:
       del_keys.append(key)
            
for key in del_keys:
    del candidates[key]
end = len(candidates.keys())
print('Removed '+str(begin-end)+ ' pairs out of ' + str(begin)+ ' candidate pairs due to same shop ' + str(end/begin*100)+"%") 
#%% Brand Search
begin = len(candidates.keys())

product_brand = {}
for i in DataSet.index:
    product_brand[i] = 'none'    
    for brand in ListOfBrands:
        if brand in DataSet['title'][i] or brand in DataSet['kvp'][i]:
            product_brand[i] = brand
            break
        
product_withbrand = {k:v for k,v in product_brand.items() if v != 'none'}

    
#%%    
del_keys = []
for key in candidates.keys():
    brand_i = product_brand[key[0]]
    brand_j = product_brand[key[1]]
    if brand_i == 'none' or brand_j == 'none':
        continue
    else:
        if brand_i != brand_j:
            del_keys.append(key)

        
for key in del_keys:
    del candidates[key]
end = len(candidates.keys())
print('Removed '+str(begin-end)+ ' pairs out of ' + str(begin)+ ' candidate pairs due to brand inequality ' + str(end/begin*100)+"%")            

#%%find true pairs
listoftruepairs = []
for i in list(DataSet.index):
    for j in list(DataSet.index):
        if i >= j:
            continue
        else:
            if DataSet['key'][i] == DataSet['key'][j]:
                pair = tuple([i,j])
                listoftruepairs.append(pair)

#%% evalueate
        
            

real = 0
l = []
for key in Data.keys():
    index = len(Data[key])
    if index == 1:
        continue
    if index == 2:
        real += 1
        l.append(key)
    if index == 3:
        real += 3  
        l.append(key)
    if index == 4:
        real += 6
        l.append(key)

        
        
possible = {}
    
counter = 0
#candidates = {k:v for k,v in candidates.items() if v >= 400}
x = []
for candidate in list(candidates.keys()):
    if DataSet['key'][candidate[0]] == DataSet['key'][candidate[1]]:
        counter +=1
        x.append(DataSet['key'][candidate[0]])
    else:
        continue
        
        
totalcombinationspossible = sum(np.arange(0, len(DataSet)+1, 1).tolist())
print('|----------------LSH------------------|')
print('')
print('LSH filtered '+ str((totalcombinationspossible-end)/totalcombinationspossible*100)+"% of the total combinations out")
PQ = counter/len(candidates)
PC = counter/real
F1 = (2*PQ*PC)/(PQ+PC)
print('Original pairs found       =  '+str(counter))
print('Original pairs lost        =  '+str(real - counter))
print('candidates to be compaired =  '+str(len(candidates)))
print('')
print("LSH threshold is "+ str(th))
print(str(counter)+' out of ' + str(real) + " duplicates found")
print(str(counter)+' out of ' + str(len(candidates)) + " candidates are duplicates")
print('')
# print('F1 = ' + str(F1))

fbeta = (1+beta*beta)*(PQ*PC)/(beta*beta*PQ+PC)
print('PQ        = ' + str(PQ))
print('PC        = ' + str(PC))
print('Fbeta     = ' + str(fbeta))
print('F1*       = ' + str(F1))

#%% CLustering:
from sklearn.cluster import AgglomerativeClustering

    
def jacsim(a,b):
    v1 = SignatureMatrix[:,a]
    v2 = SignatureMatrix[:,b]
    v3 = v1-v2
    v3[v3!=0] = 1
    jdis  = sum(v3)/len(v3)
    # if tuple([a,b]) in  potmodelIDpairs:
    #     return 1 - 0.5*(0+jdis)
    return 1 - jdis

    
DistanceMatrix = np.ones((len(products),len(products)))*10000000
for i in range(DistanceMatrix.shape[0]):
    for j in range(DistanceMatrix.shape[1]):
        if tuple([i,j]) in candidates.keys():
            DistanceMatrix[i][j] = 1 - jacsim(i,j)




clustering = AgglomerativeClustering(affinity='precomputed', linkage='single', distance_threshold = epsilon, n_clusters = None).fit_predict(DistanceMatrix)

bucketscl = {}
for i in range(len(clustering)):
    if clustering[i] not in bucketscl.keys():
        bucketscl[clustering[i]] = [i]
    else:
        bucketscl[clustering[i]].append(i)
bucketscl = {k:v for k,v in bucketscl.items() if len(v)>=2}        
result = []
for key in bucketscl.keys():
    for comb in list(combinations(bucketscl[key],2)):
        if comb[0]> comb[1]:
            comb = tuple([comb[1],comb[0]])
        result.append(comb)

 
        
        
# del key, comb
#%% Performance
print('-----------------CLustering-------------------')
TP = set(result).intersection(set(listoftruepairs))
FP = set(result) - set(listoftruepairs)
FN = set(listoftruepairs) - set(result)

precision = len(TP)/(len(TP)+len(FP)) #PQ
recall    = len(TP)/(len(TP)+len(FN)) #PC  

F1 = 2*(precision*recall)/(precision+recall)
print('Precision = ' + str(precision))
print('recall    = ' + str(recall))

print('F1        = ' + str(F1))
        




    