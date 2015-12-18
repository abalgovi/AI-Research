import math as math
import numpy as np
import GALayer

class GANetwork:
   
   def __init__( self, numberOfLayers, nodesPerLayer ):
      self.net = [None] * ( numberOfLayers - 1 )
      for layer in xrange(0,len(self.net)):
         self.net[layer] = GALayer.GALayer( nodesPerLayer[layer+1], nodesPerLayer[layer] )


   def runData( self, inputs, optVal, individual, train ):
      for layer in xrange(0,len(self.net)):
         self.net[layer].feedForward( inputs if layer < 1 else None, 
         None if layer == len(self.net) - 1 else self.net[layer+1] ) 
      
      if train == 1:
         o = self.decToBin( optVal )
         for layer in xrange(len(self.net)-1, -1, -1):
            decode = individual.oDecode if layer == (len(self.net) - 1) else individual.hDecode
            lyr = self.net[layer+1] if layer < len(self.net) - 1 else None
            self.net[layer].updateWeights( decode, o,lyr )
      
      return self.binToDec( self.net[len(self.net) - 1].outputs )
   

   
   def computeAcc( self, optVals, outputs ):
      assert(outputs.size == len(optVals))
      ttlCorrect = 0.0
      for value in xrange(0,outputs.size,1):
         if( self.convToBinary(optVals[value]) == outputs[value] ):  ttlCorrect += 1 
      return ttlCorrect/outputs.size


   def decToBin( self, val ):
      if val == 0: return [0,0]
      elif val == 1: return [0,1]
      elif val == 2: return [1,0]
      elif val == 3: return [1,1]


   def convToBinary( self, value ):
      if value == 0:  return 0
      elif value == 1: return 1
      elif value == 2: return 10
      elif value == 3: return 11


   def binToDec( self, val ):
      if int(val[0]) == 0 and int(val[1]) == 0: return 0
      elif int(val[0]) == 0 and int(val[1]) == 1: return 1
      elif int(val[0]) == 1 and int(val[1]) == 0: return 10
      elif int(val[0]) == 1 and int(val[1]) == 1: return 11
      

