# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:18:39 2026

@author: getsh
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml("mnist_784", version=1,return_X_y=True)
X = X.values[:50000] 
y = y.astype(int).values[:50000]

X = (X/255 - 0.5)*2

Xtemp, Xtest_std, ytemp, ytest = train_test_split(X, y, random_state=123,
                                              test_size=0.3, stratify=y)

Xtrain_std, Xvalid, ytrain, yvalid = train_test_split(Xtemp, ytemp, test_size=0.1,
                                                  random_state=123, stratify=ytemp)



class neuralnet: 
    
    def __init__(self,Ksize, num_features, num_classes, nodes,  random_state=123): 
        self.nodes = nodes 
        self.F, self.Kh, self.Kw = Ksize
        self.random_state = random_state 
        self.num_classes = num_classes 
        self.num_features = num_features
        
        rng = np.random.RandomState(self.random_state)
        
        self.k_ = rng.normal(loc=0.0, scale=0.1, size=(self.F, self.Kh, self.Kw))
        self.d_ = np.zeros(self.F).ravel()
        
        self.w1 = rng.normal(loc=0.0, scale=0.1, size=(nodes, self.F*self.num_features))
        self.b1 = np.zeros(nodes).ravel() 
        
        self.w2 = rng.normal(loc=0.0, scale=0.1, size=(num_classes, nodes))
        self.b2 = np.zeros(num_classes).ravel()
        
        
    def forward(self, X): 
        
        m, Nh, Nw = X.shape
        
        #convolutional layer
        z0 = self.conv(X, self.k_, self.d_, mode='forward')
        a0 = self.ReLU(z0) 
        h = a0.reshape(m, self.F*self.num_features) #flattenning to the fully connected layer
        
        #first layer
        z1 = np.dot(h, self.w1.T) + self.b1
        a1 =  self.ReLU(z1)      
        
        #2nd layer 
        z2 = np.dot(a1, self.w2.T) + self.b2
        a2 = self.softmax(z2)       
        
        return a0, a1, a2
    
    def backward(self, X, y, a0, a1, a2): 
        
        #shape of the initial data
        m, Nh, Nw = X.shape
        
        #making ready y for vectorized operations
        y = self.onehot(y, self.num_classes)
        h = a0.reshape(m, self.F*self.num_features)
        
        
        # putting forward E2 , dim = num_examples*num_classes
        #sig2 = a2*(1-a2) for softmax
        E2 = (y - a2)
        
        
        # do the same for E1, dim = num_examples*num_nodes
        sig1 = self.dReLU(a1) 
        E1 = np.dot(E2, self.w2)*sig1
        
        #Now for E0, dim = num_examples*(num_featuers*num_F) --> flattened to (num_examples, num_F, Nh, Nw)
        sig0 = self.dReLU(a0)
        E0 = np.dot(E1, self.w1).reshape(m, self.F, Nh, Nw)*sig0
        
        
        # evaluating changes 
        dw2 = np.dot(E2.T, a1) 
        db2 = np.sum(E2, axis=0) 
        
        dw1 = np.dot(E1.T, h)
        db1 = np.sum(E1, axis=0)
        
        dk = self.conv(X, E0, d=0, mode="backward") 
        dd = np.sum(E0, axis=(0,2,3))
        
        
        return dk, dd, dw1, db1, dw2, db2
        
    
    def padding(self, X, K, p="same", s=(1,1), mode="forward"): 
        
        # size and shapes
        m, Nh, Nw = X.shape 
        Kh, Kw =  self.Kh, self.Kw
        
          
        
        #picking the padding type
        if type(p) == int: 
            ph, pw = 2*p
            
        elif p == "valid": 
            ph, pw = 0, 0
            
        elif p == "same": 
            sh, sw = s
            ph = (Nh - 1)*sh - Nh + Kh 
            pw = (Nw - 1)*sw - Nw + Kw 
                          
        else: 
            raise ValueError("Incorrect value: padding is integer type, 'same' or 'valid'") 
        
        #to be appended for each side
        Lp = np.zeros((m, Nh, pw//2))
        Rp = np.zeros((m, Nh, (pw+1)//2))
        Up = np.zeros((m, ph//2, Nw + pw))
        Bp = np.zeros((m, (ph+1)//2, Nw + pw))
        
        #appending from horizontal first to vertical axis
        Xp = np.concatenate((Lp, X, Rp), axis=2)
        Xp = np.concatenate((Up, Xp, Bp), axis=1) 
        
        return Xp 
    
    def conv(self, X, K, d, p="same", s=(1,1), mode="forward"):
        
        Xp = self.padding(X, self.k_, p, s, mode) 
        
        #shape of the padded image, and size of kernel, and strides
        m, Nh, Nw = Xp.shape 
        Kh, Kw =  self.Kh, self.Kw
        sh, sw = s
        
     
        
        #output size
        Oh = (Nh - Kh)//sh + 1 
        Ow = (Nw - Kw)//sw + 1
        
        #striding size
        strides = (Nh*Nw, sh*Nw, sw, Nh, 1) 
        strides = tuple(i*Xp.itemsize for i in strides)
        
        #strided matrix 
        subM = np.lib.stride_tricks.as_strided(Xp, shape=(m, Oh, Ow, Kh, Kw), strides=strides)
      
        #looking for the mode of the convolution
        if mode == "forward":
            return np.einsum("fkl, mijkl -> mfij", K, subM) + d[None, :, None, None]
        elif mode == "backward":
            return np.einsum("mfij, mijkl -> fkl", K, subM)
    
        
    def ReLU(self, z):
        return np.maximum(0,z)
   
    def dReLU(self, z): 
        return (z>0)
    
    def softmax(self, z): 
        z = np.clip(z, -50, 50)
        
        if z.ndim == 1:
            return np.exp(z)/np.sum(np.exp(z))
        else:
            return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)
    
    def sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1./(1 + np.exp(-z))
    
    def onehot(self, y, num_labels): 
        
        arr = np.zeros((y.shape[0], num_labels))
        
        for i, val in enumerate(y):
            arr[i, val] = 1 
        
        return arr
    


Xtrain_std = Xtrain_std.reshape(Xtrain_std.shape[0], 28, 28) 
Xvalid = Xvalid.reshape(Xvalid.shape[0], 28, 28)   
model = neuralnet(Ksize=(10,3,3), num_features=28*28, num_classes=10, nodes=30) 


def onehot(y, num_labels): 
    
    arr = np.zeros((y.shape[0], num_labels))
    
    for i, val in enumerate(y):
        arr[i, val] = 1 
    
    return arr

def minibatch(X, y, batchsize=100): 
    
    indices = np.arange(0, X.shape[0], batchsize)
    np.random.shuffle(indices)
    
    for idx in indices: 
        Xbatch = X[idx: idx + batchsize, :, :]
        ybatch = y[idx: idx + batchsize]
        
        yield Xbatch, ybatch
        
        
def trainer(model, X, y, Xvalid, yvalid, batchsize=100, n_iter=15, eta=0.5): 
    n, Nh, Nw = X.shape
  
    loss = []
    acc = []
    Vacc = []
    Vloss = []
    
    yvalid_onehot = onehot(yvalid, 10)
    
    for epoch in range(n_iter): 
        
        correctE = 0
        lossE = 0
        for Xbatch, ybatch in minibatch(X, y, batchsize=100): 
            
            a0, a1, a2 = model.forward(Xbatch) 
            dk, dd, dw1, db1, dw2, db2 = model.backward(Xbatch, ybatch, a0, a1, a2)
            
            #Updating
            model.k_ += eta*dk/batchsize 
            model.d_ += eta*dd/batchsize
            model.w1 += eta*dw1/batchsize
            model.b2 += eta*db2/batchsize
            model.b1 += eta*db1/batchsize
            model.w2 += eta*dw2/batchsize
           
            
            ybatch_onehot = onehot(ybatch, 10)
            ypred = np.argmax(a2, axis=1)
            lossE += np.mean((a2 - ybatch_onehot)**2)
            correctE += sum(ypred == ybatch)
            
            
        loss.append(lossE)
        acc.append(correctE/n)
        
        _, _, av2 = model.forward(Xvalid)
        yvalid_pred  = np.argmax(av2, axis=1)
        accVE = np.mean(yvalid_pred == yvalid) 
        lossVE = np.mean((av2 - yvalid_onehot)**2) 
        
        Vacc.append(accVE)
        Vloss.append(lossVE)
        
        print(f"{epoch + 1}|Tloss: {lossE/n: .3f} |Train acc: {correctE/n*100 : .2f}% |Vloss: {lossVE: .3f} |Valid acc: {accVE*100: .2f}% ")
    print("__________________________________________________________________________")
    return acc, loss, Vacc, Vloss

accT, lossT, accV, lossV = trainer(model, Xtrain_std, ytrain, Xvalid, yvalid)


fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(2,2,1)
ax1.plot(np.arange(0, len(accT)), accT, label="Train", marker="+")
ax1.grid()
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2 = fig.add_subplot(2,2,2)
ax2.plot(np.arange(0, len(accV)), accV, label="Valid", marker=".")
ax2.grid()
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

ax3 = fig.add_subplot(2,2,3)
ax3.plot(np.arange(0, len(lossT)), lossT, label="Train")
ax3.grid()
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss")
ax3.legend()

ax4 = fig.add_subplot(2,2,4)
ax4.plot(np.arange(0, len(lossV)), lossV, label="Valid")
ax4.grid()
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Loss")
ax4.legend()

plt.title("CNN Implementation")
plt.tight_layout()
plt.show()           

