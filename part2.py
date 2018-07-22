# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 09:36:34 2018

@author: atharvavyas
"""
import os
import re
import nltk
import random
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.porter import *
import math
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
random.seed(7)
#############################function definitions #########################
def plot_dict(d,i):
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples    
    plt.plot(x, y)
    s="figure_2_"+str(i)+".png"
    plt.savefig(s)
    plt.close()
def find_tokens(file_name):
    fo=open(file_name,"r")
    fc=fo.read()
    tokens=set(re.split('[ !?()"\']+|[\t]+|[\n]+|[.]+|[,]+|[:]+|[-]+',fc))
    nltk.download('stopwords')
    stop_words=set(stopwords.words('english'))
    tokens=tokens-stop_words
    tokensl=list(tokens)
    stemmer = PorterStemmer()
    tokens_final=set([stemmer.stem(tokensi) for tokensi in tokensl])
    return tokens_final
def create_vector(msg,tokens_set):
    global ystatus0,ystatus1
    l=len(tokens_set)
    v=np.zeros((l,1),dtype=int)
    msgl=list(set(re.split('[ !?()"\']+|[\t]+|[\n]+|[.]+|[,]+|[:]+|[-]+',msg)))
    stemmer = PorterStemmer()
    msg_final=list(set([stemmer.stem(tokensi) for tokensi in msgl]))
    cnt=0
    for t in msg_final:  
        if t in tokens_set:
            i=tokens_set.index(t)
            v[i,0]=1
            cnt=cnt+1
        if t=='ham':
            ystatus0=1
            ystatus1=0
    #print("cnt:"+str(cnt))
    return v
############################ main flow ####################################
tokens=find_tokens("Assignment_2_data.txt")
tokens.remove('')
tokens.remove('ham')
tokens.remove('spam')
tokensl=list(tokens)
tokens_length=len(tokensl)
fo=open("Assignment_2_data.txt","r")
line=fo.readline()
line_list=[]
while(line):
    line_list.append(line)
    line=fo.readline()
### diving line list into two partitions with random indices###
ll=len(line_list)
test_list=[]
for i in range((int)(0.2*ll)):
    r=(int(random.random()*ll))%((int)(ll*0.8))
    test_list.append(line_list[r])
    line_list.remove(line_list[r])
train_list=line_list
train_len=len(train_list)
test_len=(len(test_list))

##########################pre processing done till here ####################
def forward_propogate(v,tp):
    global xl1,xl2,xl3
    vcopy=np.copy(v)
    vcopy=np.insert(vcopy,0,1,axis=0)
    if tp==1:
        xl1=np.tanh(np.dot(np.transpose(wl1),vcopy))
    else:
        xl1=sigmoid(np.dot(np.transpose(wl1),vcopy))
    xl1copy=np.copy(xl1)
    xl1copy=np.insert(xl1copy,0,1,axis=0)
    if tp==1:
        xl2=np.tanh(np.dot(np.transpose(wl2),xl1copy))
    else:
        xl2=sigmoid(np.dot(np.transpose(wl2),xl1copy))
    #print("xl2")
    #print(xl2)
    xl2copy=np.copy(xl2)
    #print("xl2copy")
    #print(xl2copy)
    xl2copy=np.insert(xl2copy,0,1,axis=0)
    if tp==1:
        xl3=np.dot(np.transpose(wl3),xl2copy)#no activation function in the last layer
    else:
        xl3=np.dot(np.transpose(wl3),xl2copy)
def backward_propogate(tp):
    global dl1,dl2,dl3,xl3,xl2,xl1,wl1,wl2,wl3,ystatus0,ystatus1
    if tp==1:
        a30=math.exp(xl3[0,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
        a31=math.exp(xl3[1,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
        dl3[0,0]= (a30-ystatus0) #derivative as per cross entropy
        dl3[1,0]= (a31-ystatus1)
        #print("dl3:")
        #print(dl3)
        dl2=(1-xl2**2)*np.dot(wl3[1:,:],dl3)
        dl1=(1-xl1**2)*np.dot(wl2[1:,:],dl2)
    else:
        a30=math.exp(xl3[0,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
        a31=math.exp(xl3[1,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
        dl3[0,0]=(a30-ystatus0)
        dl3[1,0]=(a31-ystatus1)
        dl2=(xl2)*(1-xl2)*np.dot(wl3[1:,:],dl3)
        dl1=(xl1)*(1-xl1)*np.dot(wl2[1:,:],dl2)

def update_weights(v,tp):
    global wl1,wl2,wl3,xl1,xl2,dl1,dl2,dl3
    xl2copy=np.copy(xl2)
    #print(xl2)
    xl2copy=np.insert(xl2copy,0,1,axis=0)
    #print("xl2copy")
    #print(xl2copy)
    xl1copy=np.copy(xl1)
    xl1copy=np.insert(xl1copy,0,1,axis=0)
    vcopy=np.copy(v)
    vcopy=np.insert(vcopy,0,1,axis=0)
    #print(wl3)
    if tp==1:
        wl3=wl3-0.1*np.dot(xl2copy,np.transpose(dl3))
        #print("prod")
        #print(np.dot(xl2copy,np.transpose(dl3)))
        #print("wl3")
        #print(wl3)
        wl2=wl2-0.1*np.dot(xl1copy,np.transpose(dl2))
        wl1=wl1-0.1*np.dot(vcopy,np.transpose(dl1))
    else:
        wl3=wl3-0.1*np.dot(xl2copy,np.transpose(dl3))
        #print("prod")
        #print(np.dot(xl2copy,np.transpose(dl3)))
        #print("wl3")
        #print(wl3)
        wl2=wl2-0.1*np.dot(xl1copy,np.transpose(dl2))
        wl1=wl1-0.1*np.dot(vcopy,np.transpose(dl1))

d1={}
d2={}
d3={}
d4={}#all for tanh insample,out sample,in accuracy,out accuracy
D1={}
D2={}
D3={}
D4={}#to plot sigmoid insample,outsample,in accuracy,out accuracy     
############################using sigmoid as activation function#####################################
print("for sigmoid function")
wl1=(np.random.uniform(low=-1,high=1,size=(tokens_length+1,100)))
wl2=(np.random.uniform(low=-1,high=1,size=(100+1,50)))
wl3=(np.random.uniform(low=-1,high=1,size=(50+1,2)))
xl1=np.zeros((100,1),dtype=float)
xl2=np.zeros((50,1),dtype=float)
xl3=np.zeros((2,1),dtype=float) #since there are two output neurons
dl1=np.zeros((100,1),dtype=float)
dl2=np.zeros((50,1),dtype=float)
dl3=np.zeros((2,1),dtype=float)
ystatus0=0 #for spam ystatus=[0,1] i.e ystatus0=0 and ystatus1=1 and for ham, ystatus=[1,0]
ystatus1=1 
ind=0;
for it in range(60001):
    print("\r it now"+str(it), end='\r')
    if it%2000==0:
        sm=0.0;
        correct=0
        for x in range(test_len):
            ystatus0=0
            ystatus1=1
            v=create_vector(test_list[x],tokensl)
            forward_propogate(v,0)
            a30=math.exp(xl3[0,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
            a31=math.exp(xl3[1,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
            sm=sm+(a30-ystatus0)**2
            sm=sm+(a31-ystatus1)**2
            if a30>a31 and ystatus0==1:
                correct=correct+1
            if a31>=a30 and ystatus1==1:
                print("\r spam spotted"+str(it), end='\r')
                correct=correct+1
        print("external_cost:"+str(float(sm)/test_len))
        print("external accuracy:"+str(float(correct)/test_len))
        sm=0.0
        correct=0
        for x in range(train_len):
            ystatus0=0
            ystatus1=1
            v=create_vector(train_list[x],tokensl)
            forward_propogate(v,0)
            a30=math.exp(xl3[0,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
            a31=math.exp(xl3[1,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
            sm=sm+(a30-ystatus0)**2
            sm=sm+(a31-ystatus1)**2
            if a30>a31 and ystatus0==1:
                correct=correct+1
            if a31>=a30 and ystatus1==1:
                correct=correct+1
        print("internal_cost:"+str(float(sm)/train_len))
        print("internal accuracy:"+str(float(correct)/train_len))
        D1[it]=float(sm)/test_len
        D2[it]=float(correct)/test_len
        D3[it]=float(sm)/train_len
        D4[it]=float(correct)/train_len
    ystatus0=0
    ystatus1=1
    #print(train_list[ind])
    v=create_vector(train_list[ind],tokensl)
    #print(ystatus)
    ind=(ind+1)%train_len
    forward_propogate(v,0)
    backward_propogate(0)
    update_weights(v,0)
plot_dict(D1,1)
plot_dict(D2,2)
plot_dict(D3,3)
plot_dict(D4,4)
############################using tanh as activation function#####################################
print("for tanh function")
wl1=(np.random.uniform(low=-1,high=1,size=(tokens_length+1,100)))
wl2=(np.random.uniform(low=-1,high=1,size=(100+1,50)))
wl3=(np.random.uniform(low=-1,high=1,size=(50+1,2)))
xl1=np.zeros((100,1),dtype=float)
xl2=np.zeros((50,1),dtype=float)
xl3=np.zeros((2,1),dtype=float) #since there are two output neurons
dl1=np.zeros((100,1),dtype=float)
dl2=np.zeros((50,1),dtype=float)
dl3=np.zeros((2,1),dtype=float)
ystatus0=0 #for spam ystatus=[0,1] i.e ystatus0=0 and ystatus1=1 and for ham, ystatus=[1,0]
ystatus1=1 
ind=0;
for it in range(60001):
    print("\r it now"+str(it), end='\r')
    if it%4000==0:
        sm=0.0;
        correct=0
        for x in range(test_len):
            ystatus0=0
            ystatus1=1
            v=create_vector(test_list[x],tokensl)
            forward_propogate(v,1)
            a30=math.exp(xl3[0,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
            a31=math.exp(xl3[1,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
            sm=sm+(a30-ystatus0)**2
            sm=sm+(a31-ystatus1)**2
            if a30>a31 and ystatus0==1:
                correct=correct+1
            if a31>=a30 and ystatus1==1:
                print("\r spam spotted"+str(it), end='\r')
                correct=correct+1
        print("external_cost:"+str(float(sm)/test_len))
        print("external accuracy:"+str(float(correct)/test_len))
        sm=0.0
        correct=0
        for x in range(train_len):
            ystatus0=0
            ystatus1=1
            v=create_vector(train_list[x],tokensl)
            forward_propogate(v,1)
            a30=math.exp(xl3[0,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
            a31=math.exp(xl3[1,0])/(math.exp(xl3[0,0])+math.exp(xl3[1,0]))
            sm=sm+(a30-ystatus0)**2
            sm=sm+(a31-ystatus1)**2
            if a30>a31 and ystatus0==1:
                correct=correct+1
            if a31>=a30 and ystatus1==1:
                correct=correct+1
        print("internal_cost:"+str(float(sm)/train_len))
        print("internal accuracy:"+str(float(correct)/train_len))
        d1[it]=float(sm)/test_len
        d2[it]=float(correct)/test_len
        d3[it]=float(sm)/train_len
        d4[it]=float(correct)/train_len
    ystatus0=0
    ystatus1=1
    #print(train_list[ind])
    v=create_vector(train_list[ind],tokensl)
    #print(ystatus)
    ind=(ind+1)%train_len
    forward_propogate(v,1)
    backward_propogate(1)
    update_weights(v,1)
plot_dict(d1,5)
plot_dict(d2,6)
plot_dict(d3,7)
plot_dict(d4,8)