from math import exp
from random import uniform
import numpy as np

class GANode:
   
   def __init__( self, ttlWeights ):
      self.initWeights( ttlWeights + 1 )
      self.deltaWeight = np.zeros([ttlWeights + 1], dtype=float)
      
   
   def initWeights( self, ttlWeights ):
      self.weights = np.zeros([ttlWeights], dtype=float)
      for weight in xrange(0, ttlWeights):  self.weights[weight] = uniform( -5.0, 5.0 )
   
   
   def actFunc( self, s ):
      return (exp(s) - exp(-s))/(exp(s) + exp(-s))
      

   def computeOtps( self, inputs ):
      assert( len(self.weights) == len(inputs) )
      return  self.actFunc(np.sum( np.multiply(inputs, self.weights )))
   
   
   def lrnRuleOut( self, inp, out, optVal, index, c ):
      
      w = c[0] * ( c[1]*self.weights[index] + c[2]*inp + c[3]*out + c[4]*optVal 
      + c[5]*self.weights[index]*inp + c[6]*self.weights[index]*out + c[7]*
      self.weights[index]*optVal + c[8]*inp*out + c[9]*inp*optVal + c[10]*out*optVal )
      
      temp = self.weights[index]
      self.weights[index] += w
      
      if self.weights[index] > 5.0:     self.weights[index] = 5.0
      elif self.weights[index] < -5.0:  self.weights[index] = -5.0
      
      self.deltaWeight[index] = self.weights[index] - temp


   def lrnRuleHid( self, otpt, outgngLB, outDelta, outLOtpt, optVal, i,decode ):
      coeff, s, index = decode, 0, 1
      cons = (self.weights[i],otpt,outgngLB,outDelta,outLOtpt,optVal)
      for a in xrange(0,len(cons)):
         for value in xrange(a+1, len(cons)):
            if a < 2 and value <= 1:  
               s += np.multiply(cons[a], cons[value]) * coeff[index]
            elif a < 2 and value > 1:
               s += np.sum(np.multiply(cons[a], cons[value])) * coeff[index]
            else:
               s += np.dot( cons[a], cons[value] ) * coeff[index]
            index += 1

      self.weights[i] += s * coeff[0]
      if self.weights[i] > 5.0: self.weights[i] = 5.0
      elif self.weights[i] < -5.0: self.weights[i] = -5.0

