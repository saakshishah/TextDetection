#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saakshishah
"""
import pickle
import numpy as np
import sklearn.neural_network as nn
import sklearn.utils as su
import matplotlib.pyplot as plt 
import sklearn.cluster as sc
import sklearn.mixture as sm
import bonnerlib2D as bl2d
import math

print('\n\nQuestion 1')
print('----------')

with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)
    
# Showing an image - this one shows a '2'
# plt.imshow(Xtrain[0].reshape(28,28),interpolation='nearest')
    
print('\nQuestion 1(a):') 

smallX = Xtrain[0:500]
smallT = Ttrain[0:500]

print('\nQuestion 1(b):') 
# 1 hidden unit
np.random.seed(7)
nn5 = nn.MLPClassifier(max_iter=10000, activation='logistic', hidden_layer_sizes=(5,), solver="sgd", learning_rate_init=0.1, alpha=0)
nn5.fit(smallX,smallT)
accuracy_1b = nn5.score(smallX,smallT)
accuracy_1b2 = nn5.score(Xval,Tval)
print(accuracy_1b)
print(accuracy_1b2)

print('\nQuestion 1(d):') 
pred = np.argmax(nn5.predict_proba(Xval),axis=1)
accuracy = np.mean(pred == Tval)
print(accuracy - accuracy_1b2)

print('\nQuestion 1(e):')
np.random.seed(7)
matrices = []
for i in range(7):
    newX,newT = su.resample(smallX,smallT)
    nn5.fit(newX,newT)
    matrices.append(nn5.predict_proba(Xval))

average_matrice = np.mean(matrices,axis=0)
pred = np.argmax(average_matrice,axis=1)
accuracy_average = np.mean(pred == Tval)

print('\nQuestion 1(g):')
np.random.seed(7)
accuracies = []
matrices_2 = []
for i in range(100):
    newX,newT = su.resample(smallX,smallT)
    nn5.fit(newX,newT)
    matrices_2.append(nn5.predict_proba(Xval))
    average_matrice = np.mean(matrices_2,axis=0)
    pred = np.argmax(average_matrice,axis=1)
    accuracy_average = np.mean(pred == Tval)
    accuracies.append(accuracy_average)



print('\n\nQuestion 2')
print('----------')

q = np.zeros((100, 4))
moves = [0,1,2,3]

print('\nQuestion 2(a):') 
environment = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.54509807, 0., 0., 0., 0.],
                [0., 0.54509807, 0.54509807, 0., 0., 0.54509807, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.54509807, 0.54509807, 0.54509807, 0.54509807, 0.],
                [0., 0., 0., 0., 0., 0.54509807, 0.9921569, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.54509807, 0., 0., 0., 0.],
                [0., 0., 0.54509807, 0.54509807, 0.54509807, 0.54509807, 0., 0., 0.54509807, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],]


plt.imshow(environment,interpolation='nearest')
plt.title('Question 2(a): Grid World')

print('\nQuestion 2(b):') 


def Trans(L, a):
 
    new_loc = [0,0]
    # down
    if a == 0:
        new_loc = [L[0] + 1, L[1]]
    # up
    elif a == 1:
        new_loc = [L[0] - 1, L[1]]
    # left
    elif a == 2:
        new_loc = [L[0], L[1] - 1]
     # right
    else:
        new_loc = [L[0], L[1] + 1]
        
    if not(all(ele >= 0 and ele < 10 for ele in new_loc)):
            return [L, 0]
      
    if new_loc == [5,6]:
        return [new_loc, 25]
    
    if (environment[new_loc[0]][new_loc[1]] != 0.54509807):
        reward = 0
        return [new_loc, reward]
    return [L, 0]


print('\nQuestion 2(c):') 
def choose(L,beta):
    x = np.exp(q*beta)
    softmax = (x/(np.sum(x,axis=1,keepdims=True)))
    row_of_q = (L[1] * 10) + L[0]
    new_action = np.random.choice(4,size=1,p=softmax[row_of_q])
    return new_action

print('\nQuestion 2(d):') 
def updateQ(L,a,alpha,gamma):
    new = Trans(L,a) 
    new_row_of_q = (new[0][1] * 10) + new[0][0]
    row_of_q = (L[1] * 10)+ L[0]
    if not new_row_of_q == row_of_q:
        q[row_of_q, a] = q[row_of_q, a] + alpha * (new[1] + (gamma * np.max(q[new_row_of_q]))- q[row_of_q, a])
    return (new[1], new[0]) 
        
    
print('\nQuestion 2(e):')
def episode(L,alpha,gamma,beta):
    
    iter = 0
    temp_reward = 0
    temp_loc = L
    while temp_reward != 25:
        new_action = choose(temp_loc,beta)
        updated_q = updateQ(temp_loc,new_action,alpha,gamma)
        temp_reward = updated_q[0]
        temp_loc = updated_q[1]
        iter = iter + 1
    return iter


print('\nQuestion 2(f):')
def learn(N,L,alpha,gamma,beta):
    global q 
    q = np.zeros((100, 4))
    
    num_of_epi = []
    i = 0
    while i < N:
        num_of_epi.append(episode(L,alpha,gamma,beta))
        i = i + 1
    return num_of_epi


print('\nQuestion 2(g):')
np.random.seed(7)
plt.figure()
plt.plot(learn(50,[5,3],1,0.9,1))
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.grid()
plt.title('Question 2(g): one run of Q learning.')

np.random.seed(7)
print('\nQuestion 2(h):')
plt.figure()
plt.plot(learn(50,[5,3],1,0.9,0))
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.grid()
plt.title('Question 2(h): one run of Q learning. (beta=0)')


print('\nQuestion 2(i):')
np.random.seed(7)
list1 = []
for i in range(100):
    list1.append(learn(50,[5,3],1,0.9,1))
    
avg = np.mean(list1,axis=0)                  
                      
plt.figure()    
plt.plot(list(range(1, 51)),avg)                      
plt.xlim(0, 50)
plt.ylim(0, 1000)
plt.title("Question 2(i): 100 runs of Q learning.")

print('\nQuestion 2(j):')
np.random.seed(7)
list2 = []
for i in range(100):
    list2.append(learn(50,[5,3],1,0.9,0.1))
    
avg1 = np.mean(list2,axis=0)                  
                      
plt.figure()    
plt.plot(list(range(1, 51)),avg1)                      
plt.xlim(0, 50)
plt.ylim(0, 1000)
plt.title("Question 2(j): 100 runs of Q learning. (beta=0.1)")

print('\nQuestion 2(k):')
np.random.seed(7)
list3 = []
for i in range(100):
    list3.append(learn(50,[5,3],1,0.9,0.01))
    
avg2 = np.mean(list3,axis=0)                  
                      
plt.figure()    
plt.plot(list(range(1, 51)),avg2)                      
plt.xlim(0, 50)
plt.ylim(0, 1000)
plt.title("Question 2(k): 100 runs of Q learning. (beta=0.01)")

print('\nQuestion 2(m):')
np.random.seed(7)  
learn(50,[5,3],1,0.9,1) 
plt.figure()
plt.imshow(np.max(q,axis=1).reshape(10,10))
plt.title("Question 2(m): Qmax for beta=1.")

print('\nQuestion 2(o):')
np.random.seed(7)  
plt.figure()

for i in range(9):
    learn(50,[5,3],1,0.9,1) 
    plt.subplot(3,3,i+1) 
    plt.imshow(np.max(q,axis=1).reshape(10,10))
    plt.axis('off')
plt.suptitle("Question 2(o): Qmax for beta=1.")

print('\nQuestion 2(p):')
np.random.seed(7)  
plt.figure()

for i in range(9):
    learn(50,[5,3],1,0.9,0) 
    plt.subplot(3,3,i+1) 
    plt.imshow(np.max(q,axis=1).reshape(10,10))
    plt.axis('off')
plt.suptitle("Question 2(p): Qmax for beta=0.")


        

print('\n\nQuestion 3')
print('----------')

with open('cluster_data.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file) 
    CXtrain,CTtrain = dataTrain
    CXtest,CTtest = dataTest
    
print('\nQuestion 3(a):') 
kmeans = sc.KMeans(n_clusters=3, random_state=0).fit(CXtrain)
accuracy_3 = kmeans.score(CXtrain,CTtrain)
accuracy_3b = kmeans.score(CXtest,CTtest)
print(accuracy_3)
print(accuracy_3b)

one_hot_labels = np.zeros((kmeans.labels_.size, kmeans.labels_.max() + 1))
one_hot_labels[np.arange(kmeans.labels_.size), kmeans.labels_] = 1
bl2d.plot_clusters(CXtrain, one_hot_labels)
plt.title("Question 3(a): K means")
x = kmeans.cluster_centers_[:,0]
y = kmeans.cluster_centers_[:,1]
plt.scatter(x,y,color='black')

print('\nQuestion 3(b):') 
gmixture = sm.GaussianMixture(n_components=3, covariance_type='diag',tol=math.pow(10, -7)).fit(CXtrain)
accuracy_3b1 = gmixture.score(CXtrain,CTtrain)
accuracy_3b2 = gmixture.score(CXtest,CTtest)
print(accuracy_3b1)
print(accuracy_3b2)

bl2d.plot_clusters(CXtrain, gmixture.predict_proba(CXtrain))
plt.title("Question 3(b): Gaussian mixture model (diagonal)")
plt.scatter(gmixture.means_[:,0],gmixture.means_[:,1],color='black')

print('\nQuestion 3(c):') 
gmixture2 = sm.GaussianMixture(n_components=3, covariance_type='full',tol=math.pow(10, -7)).fit(CXtrain)
accuracy_3c1 = gmixture2.score(CXtrain,CTtrain)
accuracy_3c2 = gmixture2.score(CXtest,CTtest)
print(accuracy_3c1)
print(accuracy_3c2)    

bl2d.plot_clusters(CXtrain, gmixture2.predict_proba(CXtrain))
plt.title("Question 3(c): Gaussian mixture model (full)")
plt.scatter(gmixture2.means_[:,0],gmixture2.means_[:,1],color='black')

print(accuracy_3c2-accuracy_3b2)

print('\nQuestion 3(e):') 
print("I don't know")

print('\nQuestion 3(h):') 
with open('mnistTVT.pickle','rb') as f:
    MXtrain,MTtrain,MXval,MTval,MXtest,MTtest = pickle.load(f)
    
gmixture3 = sm.GaussianMixture(n_components=10, covariance_type='diag',tol=math.pow(10, -3)).fit(MXtrain)
accuracy_3h = gmixture3.score(MXtrain,MTtrain)
accuracy_3h2 = gmixture3.score(MXtest,MTtest)
print(accuracy_3h)
print(accuracy_3h2)

plt.figure()
for i in range(10):
    
    plt.subplot(4,3,i+1) 
    plt.imshow(gmixture3.means_[i].reshape(28,28))

plt.suptitle("Question 3(h): mean vectors for 50,000 MNIST training points")

print('\nQuestion 3(i):') 
gmixture4 = sm.GaussianMixture(n_components=10, covariance_type='diag',tol=math.pow(10, -3)).fit(MXtrain[0:500])
accuracy_3i = gmixture4.score(MXtrain[0:500])
accuracy_3i2 = gmixture4.score(MXtest[0:500])
print(accuracy_3i)
print(accuracy_3i2)

plt.figure()
for i in range(10):
    
    plt.subplot(4,3,i+1) 
    plt.imshow(gmixture4.means_[i].reshape(28,28))

plt.suptitle("Question 3(i): mean vectors for 500 MNIST training points")


print('\nQuestion 3(j):') 
gmixture5 = sm.GaussianMixture(n_components=10, covariance_type='diag',tol=math.pow(10, -3)).fit(MXtrain[0:10])
accuracy_3j = gmixture5.score(MXtrain[0:10])
accuracy_3j2 = gmixture5.score(MXtest[0:10])
print(accuracy_3j)
print(accuracy_3j2)

plt.figure()
for i in range(10):
    
    plt.subplot(4,3,i+1) 
    plt.imshow(gmixture5.means_[i].reshape(28,28))

plt.suptitle("Question 3(j): mean vectors for 10 MNIST training points")













