'''-----------------------------------------------------------------------------

   File: GANode
   Description: The file defines a node in a neural network. Each node has a
                set of incoming weights and has a field for change in weights.
-----------------------------------------------------------------------------'''
from math import exp
from random import uniform
import numpy as np

class GANode:
   
   def __init__( self, ttlWeights ):
      self.initWeights( ttlWeights + 1 )
      # deltaWeight: a change in weight that is calculated whenever weights are updated
      self.deltaWeight = np.zeros([ttlWeights + 1], dtype=float)
      
   # initialie the incoming weights for this node. 
   def initWeights( self, ttlWeights ):
      # weights: incoming weights
      self.weights = np.zeros([ttlWeights], dtype=float)
      for weight in xrange(0, ttlWeights):  self.weights[weight] = uniform( -5.0, 5.0 )
   
   # the activaation function for each node 
   def actFunc( self, s ):
      return (exp(s) - exp(-s))/(exp(s) + exp(-s))
      
   # compute the output of this node
   def computeOtps( self, inputs ):
      assert( len(self.weights) == len(inputs) )
      # multiply each incoming weight by the set of inputs and sum them
      return  self.actFunc(np.sum( np.multiply(inputs, self.weights )))
   
   # learning rule for a node in the output layer
   def lrnRuleOut( self, inp, out, optVal, index, c ):
      
      # learning rule as specified in chalmers' research paper
      w = c[0] * ( c[1]*self.weights[index] + c[2]*inp + c[3]*out + c[4]*optVal 
      + c[5]*self.weights[index]*inp + c[6]*self.weights[index]*out + c[7]*
      self.weights[index]*optVal + c[8]*inp*out + c[9]*inp*optVal + c[10]*out*optVal )
      
      # save current weights so they can be used to calculate the change in weights
      temp = self.weights[index]
      self.weights[index] += w
      
      # place a cap on the weights
      if self.weights[index] > 5.0:     self.weights[index] = 5.0
      elif self.weights[index] < -5.0:  self.weights[index] = -5.0
      
      # update the change in weights
      self.deltaWeight[index] = self.weights[index] - temp

   
   # the learning rule for the hidden layer
   # parameters:  1.) otpt: output value of the node
   #              2.) outgngLB: the outgoing weights of the current node
   #              3.) outDelta: the delta weights of the nextlayer
   #              4.) optVal: the desired output value
   #              5.) i: index of the node in the hidden layer
   #              6.) decode: the coefficients that  were determined by decoding bits 0-34 of the
   #                  bitstring
   def lrnRuleHid( self, otpt, outgngLB, outDelta, outLOtpt, optVal, i,decode ):
      coeff, s, index = decode, 0, 1
      cons = (self.weights[i],otpt,outgngLB,outDelta,outLOtpt,optVal)

      # loop through the constant array and generate the values to update the weights
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

