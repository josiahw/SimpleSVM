# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:40:56 2016

@author: josiahw
"""
import numpy
import numpy.linalg

def polyKernel(a,b,pwr):
    return numpy.dot(a,b)**pwr #numpy.dot(a,a) - numpy.dot(b,b) # -1 #

def rbfKernel(a,b,gamma):
    return numpy.exp(-gamma * numpy.linalg.norm(a - b))

class SimpleSVM:
    w = None
    a = None
    b = None
    C = None
    sv = None
    kernel = None
    kargs = ()
    tolerance = None
    verbose = True
    
    def __init__(self, 
                 C, 
                 tolerance = 0.001, 
                 kernel = numpy.dot, 
                 kargs = () 
                 ):
        """
        The parameters are: 
         - C: SVC cost
         - tolerance: gradient descent solution accuracy
         - kernel: the kernel function do use as k(a, b, *kargs)
         - kargs: extra parameters for the kernel
        """
        self.C = C
        self.kernel = kernel
        self.tolerance = tolerance
        self.kargs = kargs
        
    
    def fit(self, X, y):
        """
        Fit to data X with labels y.
        """
        
        """
        Construct the Q matrix for solving
        """       
        ysigned = y * 2 - 1
        Q = numpy.zeros((len(data),len(data)))
        for i in xrange(len(data)):
            for j in xrange(i,len(data)):
                Qval = ysigned[i] * ysigned[j]
                Qval *= self.kernel(*(
                                (data[i,:], data[j,:])
                                + self.kargs
                                ))
                Q[i,j] = Q[j,i] = Qval
        
        
        """
        Solve for a and w simultaneously by coordinate descent.
        This means no quadratic solver is needed!
        The support vectors correspond to non-zero values in a.
        """
        self.w = numpy.zeros(X.shape[1])
        self.a = numpy.zeros(X.shape[0])
        delta = 10000000000.0
        while delta > self.tolerance:
            delta = 0.
            for i in xrange(len(data)):
                g = numpy.dot(Q[i,:], self.a) - 1.0
                adelta = self.a[i] - min(max(self.a[i] - g/Q[i,i], 0.0), self.C) 
                self.w += adelta * X[i,:]
                delta += abs(adelta)
                self.a[i] -= adelta
            if self.verbose:
                print "Descent step magnitude:", delta
        #print Q #self.a
        self.sv = X[self.a > 0.0, :]
        self.a = (self.a * ysigned)[self.a > 0.0]
        
        if self.verbose:
            print "Number of support vectors:", len(self.a)
        
        """
        Select support vectors and solve for b to get the final classifier
        """
        self.b = self._predict(self.sv[0,:])[0]
        if self.a[0] > 0:
            self.b *= -1
        
        
        if self.verbose:
            print "Bias value:", self.b
    
    def _predict(self, X):
        if (len(X.shape) < 2):
            X = X.reshape((1,-1))
        clss = numpy.zeros(len(X))
        for i in xrange(len(X)):
            for j in xrange(len(self.sv)):
                clss[i] += self.a[j] * self.kernel(* ((self.sv[j,:],X[i,:]) + self.kargs))
        return clss
    
    def predict(self, X):
        """
        Predict classes for data X.
        """
        
        return self._predict(X) > self.b


if __name__ == '__main__':
    import sklearn.datasets
    data = sklearn.datasets.load_digits(2).data
    labels = sklearn.datasets.load_digits(2).target
    C = 100.0
    clss = SimpleSVM(C,0.001,rbfKernel,(0.5,))
    clss.fit(data,labels)
    
    t = clss.predict(data)
    print "Error", numpy.sum((labels-t)**2) / float(len(data))
        
    
    #print sum(a > 0)
    #print w
    