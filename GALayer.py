import numpy as np
import GANode
class GALayer:
   
   def __init__( self, ttlNodes, pl ):
      self.layer = [None] * ttlNodes
      for i in xrange( 0,ttlNodes ):
         self.layer[i] = GANode.GANode( pl )
      self.inputs = np.ones( pl + 1 )
      self.outputs = np.zeros(ttlNodes)


   def feedForward( self, inputVals, nxtLyr ):
      if inputVals:   
         self.inputs = np.array(inputVals)
         self.inputs = np.append( self.inputs, 1.0 )
      self.calcOtps( nxtLyr )


   def calcOtps( self, nxtLyr ):
      for node in self.length():
         self.outputs[node] = self.layer[node].computeOtps( self.inputs ) 
         if nxtLyr !=None: nxtLyr.inputs[node] = self.outputs[node]
         else: self.outputs[node] = 1 if self.outputs[node] > 0.25 else 0


   def updateWeights( self, decode, optVals, nxtLyr ):      
      delta, outLB = None, None
      for node in self.length():
         if nxtLyr != None:
            lbInfo = self.getDlLB( node, nxtLyr )
            delta, outLb = lbInfo[1], lbInfo[0]
  
         for lb in xrange(0,len(self.inputs)):
            if nxtLyr == None :
               self.layer[node].lrnRuleOut(self.inputs[lb],self.outputs[node],
               optVals[node], lb, decode )
            else:
               self.layer[node].lrnRuleHid( self.outputs[node],outLb, delta,
               nxtLyr.outputs,optVals,lb,decode )


   def getDlLB( self, index, nxtLyr ):
      weights, delta = np.zeros(len(nxtLyr.layer)), np.zeros(len(nxtLyr.layer))
      for node in nxtLyr.length():
         weights[node] = nxtLyr.layer[node].weights[index]
         delta[node] = nxtLyr.layer[node].deltaWeight[index]
      return (weights, delta)
         

   def getWeights( self ):
      for node in self.length():
         print( self.layer[node].weights )
      print 


   def length( self ):
      return xrange( 0,len( self.layer ) )
