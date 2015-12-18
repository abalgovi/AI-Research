from random import randint, uniform
import math as math
import numpy as np

class GAFChromosome: 
   archive = []

   def __init__( self, l, string=None ):
      self.bitString, self.fitness = "",0.0
      if string==None:
         for bit in xrange( 0, l ):  self.bitString += str(randint(0,1))
      else:
         assert(len(string) == 85 )
         self.bitString = string
      
      self.hDecode, self.oDecode = self.decode( 1 ), self.decode( 0 )


   def decode( self, lyr ):
      ss, coeff = (self.bitString[35:len(self.bitString)],np.empty(16)) if lyr > 0 else (self.bitString[0:35], np.empty(11))
      coeff[0], counter =  (-1 if ss[0] == '0' else 1) * int(ss[1:5], 2), 1
      for bit in xrange(5,len(ss),3):
         coeff[counter] = (-1 if ss[bit] == '0' else 1) * int(ss[bit+1:bit+3],2)
         counter += 1
      return coeff

   
   def selection( self, chroms, s ):
      ttlFitness, fitIndvls,prbty = 0.0, [None] * len(chroms), []
      for c in chroms: prbty.append( c.fitness/s )
      return list(np.random.choice(chroms, len(chroms), p=prbty))
      
   
   def crossover( self, indvl, index=None ):
      ind = indvl
      if index >= 0:
         ind = indvl[randint(0,len(indvl)-1)] 
         while(ind.bitString == self.bitString): ind = indvl[randint(0,len(indvl)-1)]
      
      p1 = randint(0,len(self.bitString)-1)
      p2 = randint(p1,len(self.bitString)-1)
      
      s1,s2 = self.bitString[p1:p2], ind.bitString[p1:p2]
      self.bitString = self.bitString[0:p1] + s2 + self.bitString[p2:len(self.bitString)]
      ind.bitString = ind.bitString[0:p1] + s1 + ind.bitString[p2:len(self.bitString)]


   def mutate( self ):
      oppString = ""
      for bit in xrange(0,len(self.bitString)):
         if (randint(0,100) != randint(0,100)): oppString += self.bitString[bit]
         else: oppString += str( int(self.bitString[bit]) & 0 )      
      self.bitString = oppString


   def isDuplicate( self, index, population ):
      for other in xrange(index+1, len(population)):
         print population[other], ":", population[index]
         if self.bitString == population[other].bitString:  return 1
      return 0
               

   def __string__(self):
      return self.bitString


   def __repr__(self):
      return self.__string__()

