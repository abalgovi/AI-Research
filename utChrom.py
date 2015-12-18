import GAChromosome
import GANetwork
import random
import math

net = GANetwork.GANetwork(3,  (7,14,52 ) )

population = []
for individual in range(20):
   population.append(GAFChromosome.GAFChromosome(  ))
   population[individual].fitness = random.uniform(0,1)
# for l in range(len(population[individual].location)):
#     population[individual].location[l] = random.randint(1,50)

#		TESTING SELECTION

temporaryPop = population[0].selection(population)
assert(len(temporaryPop) == len(population))
population = temporaryPop

#		TESTING CROSSOVER


crsOvrPop = population[0:15]
population[0].crossover(population[0:15])

for i in range(len(crsOvrPop) -1 ):
   t =  crsOvrPop[i].bitString[2:41]
   t1 = crsOvrPop[i].bitString[41:len(crsOvrPop[i].bitString)]
   t2 = crsOvrPop[i+1].bitString[2:41] 
   t3 = crsOvrPop[i+1].bitString[41:len(crsOvrPop[i+1].bitString)]
   
   crsOvrPop[i].bitString = crsOvrPop[i].bitString[0:2] + t2 +t1
   crsOvrPop[i+1].bitString = crsOvrPop[i+1].bitString[0:2] + t + t3
   
   assert(len(crsOvrPop[i].bitString) == 85)


for i in range(len(crsOvrPop)):
   print crsOvrPop[i], "\n" , population[i], "\n"
   assert( crsOvrPop[i] == population[i] )


#---------------------------{   TESTING SPARSNESS   }------------------------#
nov = [0.0] * len(population)
for c in range(len(population)):
   chrom, distances = population[c], []
   for i in range(len(population)):
      if population[i] is not chrom:
         dist = 0
         for l in range(len(chrom.location)):
            dist += math.pow(chrom.location[l] - population[i].location[l], 2)
         distances.append( math.pow(dist, 0.5) )
   distances.sort()
   nov[c] = sum(distances[0:16]) * 1/15
   if nov[c] < 0.25: nov[c] = 0.25

novelty = []
for c in range(len(population)):
   novelty.append(population[c].sparsness( population ))
   assert( novelty[c] == nov[c])


#---------------------------{     TESTING DECODE      }------------------------#

chromosome = GAChromosome.GAChromosome( 85, 6 )
print chromosome.bitString
print chromosome.decode(0)
print chromosome.decode(1)




   



