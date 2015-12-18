'''-----------------------------------------------------------------------------

   File: GAFChromsome
   Description: This file defines the object that describes an individual in the
                fitness algorithm. Each individual has a fitness and a bitstring
                member variables
-----------------------------------------------------------------------------'''

from random import randint, uniform
import math as math
import numpy as np

class GAFChromosome: 
 
   
   def __init__( self, l, string=None ):
      # member variables for each instance of this class
      self.bitString, self.fitness = "",0.0
      
      # a string is not provided, so make a new randomized string of 1s and 0s
      if string==None:
         for bit in xrange( 0, l ):  self.bitString += str(randint(0,1))

      # if a string is provided, initialize an object with this string
      else:
         assert(len(string) == 85 )
         self.bitString = string
      
      # hdecode: decoded version of the bitstring for the learning rule of the
      # hidden layer
      # oDecode: decoded version of the bitstring for the learning rule of the
      # output layer  
      self.hDecode, self.oDecode = self.decode( 1 ), self.decode( 0 )

   
   # decodes the string for the learning rule 
   def decode( self, lyr ):
      
      # ss: the substring of the bitstring that is used to generate coeeficients
      # for the learning rule. Bits 0-34 are used for the hidden layer 35-85 for
      # the output layer
      #coeff: array that contains the coefficients that are decoded 
      ss, coeff = (self.bitString[35:len(self.bitString)],np.empty(16)) if lyr > 0 else (self.bitString[0:35], np.empty(11))
      coeff[0], counter =  (-1 if ss[0] == '0' else 1) * int(ss[1:5], 2), 1

      # loop through the substring and decode sequences of bits for the
      # coefficient
      for bit in xrange(5,len(ss),3):
         coeff[counter] = (-1 if ss[bit] == '0' else 1) * int(ss[bit+1:bit+3],2)
         counter += 1
      return coeff


   # the algorithm for selection
   def selection( self, chroms, s ):
      ttlFitness, fitIndvls,prbty = 0.0, [None] * len(chroms), []
      for c in chroms: prbty.append( c.fitness/s )
      return list(np.random.choice(chroms, len(chroms), p=prbty))
      

   # two-point crossover
   def crossover( self, indvl, index=None ):
      ind = indvl
      # get the random indexes for points of crossover
      p1 = randint(0,len(self.bitString)-1)
      p2 = randint(p1,len(self.bitString)-1)
      
      # make the substrings
      s1,s2 = self.bitString[p1:p2], ind.bitString[p1:p2]
      
      # exchange the substrings
      self.bitString = self.bitString[0:p1] + s2 + self.bitString[p2:len(self.bitString)]
      ind.bitString = ind.bitString[0:p1] + s1 + ind.bitString[p2:len(self.bitString)]


   # mutate the bitstring
   def mutate( self ):
      oppString = ""
      for bit in xrange(0,len(self.bitString)):
         if (randint(0,100) != randint(0,100)): oppString += self.bitString[bit]
         else: oppString += str( int(self.bitString[bit]) & 0 )      
      self.bitString = oppString


   # check if the bitstring is a duplicate
   def isDuplicate( self, index, population ):
      for other in xrange(index+1, len(population)):
         if self.bitString == population[other].bitString:  return 1
      return 0
               

   def __string__(self):
      return self.bitString


   def __repr__(self):
      return self.__string__()

