'''-----------------------------------------------------------------------------

   File: GANetwork
   Description: This file defines the neural network. Each network has an array
                of GALayers and trains and tests itself depending on the inputs
                that are provided from an external file. 
-----------------------------------------------------------------------------'''
import math as math
import numpy as np
import GALayer

class GANetwork:
   
   def __init__( self, numberOfLayers, nodesPerLayer ):
      
      # create an array of layers without the input layer.
      self.net = [None] * ( numberOfLayers - 1 )
      
      # loop through an initialize all the necessary layers. 
      for layer in xrange(0,len(self.net)):
         self.net[layer] = GALayer.GALayer( nodesPerLayer[layer+1], nodesPerLayer[layer] )


   # depending on the inputs, either train or test the network
   # parameters: 1.) inputs: the input value for the network
   #             2.) optVal: the optimal value for this input value
   #             3.) individual: the GAFChromosome
   #             4.) train: boolean value that indicates whether or not to train the network
   def runData( self, inputs, optVal, individual, train ):
      for layer in xrange(0,len(self.net)):
         # feed the inputs from one layer to the next, pass in inputs if the current layer
         # is the hidden layer. Pass in a next layer if the current layer has a next layer  
         self.net[layer].feedForward( inputs if layer < 1 else None, 
         None if layer == len(self.net) - 1 else self.net[layer+1] ) 
      
      # update the weights of the network only if we are in training mode. 
      if train == 1:
         o = self.decToBin( optVal )
         for layer in xrange(len(self.net)-1, -1, -1):
            decode = individual.oDecode if layer == (len(self.net) - 1) else individual.hDecode
            lyr = self.net[layer+1] if layer < len(self.net) - 1 else None
            self.net[layer].updateWeights( decode, o,lyr )
      
      # convert the ouputs of the outputlayer to decimal values
      return self.binToDec( self.net[len(self.net) - 1].outputs )
   

   # compute the accuracy of the outputs of the network by comparing it with the target values
   def computeAcc( self, optVals, outputs ):
      assert(outputs.size == len(optVals))
      ttlCorrect = 0.0
      for value in xrange(0,outputs.size,1):
         if( self.convToBinary(optVals[value]) == outputs[value] ):  ttlCorrect += 1 
      return ttlCorrect/outputs.size

   # conver the decimal value to a binary value. These will be used as inputs to each node in a
   # layer
   def decToBin( self, val ):
      if val == 0: return [0,0]
      elif val == 1: return [0,1]
      elif val == 2: return [1,0]
      elif val == 3: return [1,1]

   # change each value to its binary form
   def convToBinary( self, value ):
      if value == 0:  return 0
      elif value == 1: return 1
      elif value == 2: return 10
      elif value == 3: return 11

   # change the binary value to the decimal value
   def binToDec( self, val ):
      if int(val[0]) == 0 and int(val[1]) == 0: return 0
      elif int(val[0]) == 0 and int(val[1]) == 1: return 1
      elif int(val[0]) == 1 and int(val[1]) == 0: return 10
      elif int(val[0]) == 1 and int(val[1]) == 1: return 11
      

