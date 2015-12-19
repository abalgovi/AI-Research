from random import randint, uniform, sample
from math import pow
import numpy as np


class GAChromosome:
   
   archive = []
   
   def __init__( self, l, ttlPts, string=None ):
      self.bitString, self.novelty, self.location = "", 0.0, np.zeros(ttlPts)
      if(string != None):  self.bitString = str(string)
      self.fitness = 0.0
      if string == None:
         for bit in xrange( 0, l ):  self.bitString += str(randint(0,1))
      else:  self.bitString = string
      self.hDecode, self.oDecode = self.decode( 1 ), self.decode( 0 )


   def decode( self, layer ):
      ss, coeff = (self.bitString[35:len(self.bitString)],np.empty(16)) if layer > 0 else (self.bitString[0:35], np.empty(11))
      coeff[0], counter =  (-1 if ss[0] == '0' else 1) * int(ss[1:5], 2), 1
      for bit in xrange(5,len(ss),3):
         coeff[counter] = (-1 if ss[bit] == '0' else 1) * int(ss[bit+1:bit+3],2)
         counter += 1
      return coeff


   def selection( self, chroms, s ):
      prbty = np.empty( len(chroms) )
      for c in xrange(0,len(chroms)): prbty[c] = chroms[c].novelty/s
            
      chrom, c, value = [None] * len(chroms), 0, -1
      while( c < len(chrom) ):
         value = (value+1) if value < len(prbty) - 1 else 0
         if uniform(0,1) < prbty[value]:
            chrom[c], c = GAChromosome(len(self.bitString),4, chroms[value].bitString), c + 1
      return chrom


   def crossover( self, ind ):
      assert(len(ind.bitString) == len(self.bitString))
      p1,r = randint(0,len(self.bitString)/2),uniform(0,1.00)
      p2 = randint(p1+1,len(self.bitString) -1 )
      s1,s2 = self.bitString[p1:p2], ind.bitString[p1:p2]
      self.bitString = self.bitString[0:p1] + s2 + self.bitString[p2:len(self.bitString)]
      ind.bitString = ind.bitString[0:p1] + s1 + ind.bitString[p2:len(self.bitString)]


   def mutate( self ):
      oppString = ""
      for bit in xrange(0,len(self.bitString)):
         if (randint(0,100) != randint(0,100)): oppString +=  self.bitString[bit]
         else: oppString +=  str( int(self.bitString[bit]) & 0 )
      self.bitString = oppString


   def distance( self, loc, loc2 ):
      assert( len(loc) == len(loc2) )
      return np.sqrt(np.sum((loc - loc2)**2))


   def sparsness( self, chromosomes ):
      achCount, chromCount, dist = 0,0,[]
      while( chromCount < len(chromosomes) ):
         if achCount < len(self.archive):
            dist.append(self.distance(self.location, self.archive[achCount].location))
            achCount += 1
            continue
         if self is not chromosomes[chromCount]:
            dist.append(self.distance(self.location, chromosomes[chromCount].location))
         chromCount += 1
      
      dist.sort()
      self.novelty = (sum(dist[0:16])) * 1/15
      if self.novelty < 0.10: self.novelty = 0.10


   def addToArchive( self ):
      self.archive.append( self )


   def isDuplicate( self, index, population ):
      for other in xrange(index+1, len(population)):
         if self.bitString == population[other].bitString:
            self.bitString = ""
            for bit in xrange(0,85):  self.bitString += str(randint(0,1))
   

   def __str__( self ):
      return self.bitString


   def __repr__( self ):
      return self.__str__()
         
   
           
         
         
            
      
         
      
      

