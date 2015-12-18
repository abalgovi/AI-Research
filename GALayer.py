'''-----------------------------------------------------------------------------

   File: GALayer
   Description: This object is one layer of nodes in a neural network. Each
                layer has a set of inputs and a set of outputs that are fed-
                forward from one layer to the next. 
-----------------------------------------------------------------------------'''
import numpy as np
import GANode
class GALayer:
   
   def __init__( self, ttlNodes, pl ):
      self.layer = [None] * ttlNodes
      
      # loop through the layer and initialize it with nodes
      for i in xrange( 0,ttlNodes ):
         self.layer[i] = GANode.GANode( pl )

      # inputs for each layer + 1 for the bias node
      self.inputs = np.ones( pl + 1 )
      self.outputs = np.zeros(ttlNodes)

   
   # feed inputs from current layer to the nextlayer
   def feedForward( self, inputVals, nxtLyr ):
      
      # if the current layer is the hidden layer
      if inputVals:
         self.inputs = np.array(inputVals)
         self.inputs = np.append( self.inputs, 1.0 )
      
      self.calcOtps( nxtLyr )


   # calculate the outputs of the current layer and set the outputs of the current layer as the
   # inputs of the next layer
   def calcOtps( self, nxtLyr ):
      for node in self.length():
         self.outputs[node] = self.layer[node].computeOtps( self.inputs )

         # if the current layer is not the output layer
         if nxtLyr !=None: nxtLyr.inputs[node] = self.outputs[node]
         # if the current layer is the output layer then round the outputs
         else: self.outputs[node] = 1 if self.outputs[node] > 0.25 else 0


   # update the incoming weights for this layer. Different rules for different layers
   def updateWeights( self, decode, optVals, nxtLyr ):

      # delta: change in weights of the next layer
      # outLB: outgoing weights of the current layer  
      delta, outLB = None, None
     
      for node in self.length():
         
         # if the current layer is a hidden layer
         if nxtLyr != None:
            lbInfo = self.getDlLB( node, nxtLyr )
            delta, outLb = lbInfo[1], lbInfo[0]
  
         for lb in xrange(0,len(self.inputs)):
            
            # call the output layers learning rule method since current layer is an output layer
            if nxtLyr == None :
               self.layer[node].lrnRuleOut(self.inputs[lb],self.outputs[node],
               optVals[node], lb, decode )

            # current layer is hidden, so call hidden layer learning rule
            else:
               self.layer[node].lrnRuleHid( self.outputs[node],outLb, delta,
               nxtLyr.outputs,optVals,lb,decode )


   # retrieve the outgoing weights of the current layer and the change in weights of the next layer
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
