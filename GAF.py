import random
import sklearn.datasets as ds
import GAFChromosome, GANetwork
import threading, Queue, multiprocessing
from time import time, ctime
import numpy as np


#random generator
inp, output =ds.make_classification(n_samples=250,n_features=10,n_classes=4,n_informative=7)
out = list(output)

def runProcesses( pop, pack, q ):
   for individual in xrange(pack, pack+4):
      runDatasets( pop[individual], individual, 1,10,5, q )


def runDatasets( individual, num, ttlRuns, ttlEpochs, ttlDatasets, queue ):
   
   ttlCorrect = 0
   
   for run in xrange(0,ttlRuns):

      network = GANetwork.GANetwork( 3, (10,20,2) )

      for epoch in xrange(0,ttlEpochs):

         for dataset in xrange( 0, 200 ):

            inputs = list(inp[dataset])
            optVal = out[dataset]
            network.runData( inputs, optVal, individual, 1 )

      netOutput, outCounter = np.empty(0), 0

      for dataset in xrange( 200, 250 ):

         inputs = list(inp[dataset])
         optVal = out[dataset]
         netOutput = np.append( netOutput, network.runData(inputs, optVal,
                  individual, 0 ) )
      
      ttlCorrect += network.computeAcc( out[200:250], netOutput )

   individual.fitness = ttlCorrect/ttlRuns
   del network

   f = open("./results/indF" + str(num), 'a')
   f.write( "\nTEST RUN OF INDIVIDUAL " + str(num) + " " + str(individual) + ":\n" ) 
   f.write( "\n\t\t|-----------------------------------------------------------------------------------|" )
   f.write( "\n\t\t  Fitness:\t\t\t" + str(individual.fitness ) )
   f.write("\n\t\t|-----------------------------------------------------------------------------------|\n ")

   queue.put( individual )


#------------------------------------------- BEGIN GENETIC ALGORITHM -------------------------------------------#

population, preserve = [], multiprocessing.Queue()
'''f = open("./GAArchive")
lines = f.readlines()
for line in lines:
   string = 0
   while( string < len(population) and population[string].bitString[0:85] !=
         line[0:85] and len(population) < 40 ):
      string += 1
   if string == len(population): population.append(GAFChromosome.GAFChromosome(line[0:85]))
'''

for c in xrange(0,40): population.append(GAFChromosome.GAFChromosome(85))

for generation in xrange( 0, 300 ):
   
   processes = []
   print "\nSTARTED AT TIME", ctime( time() )

   for pack in xrange(0,len(population), 4):
      processes.append( multiprocessing.Process(target=runProcesses,
               args=(population, pack, preserve)))
   
   for process in processes: process.start()
   for process in processes: process.join()
   
   for copy in xrange( 0, len(population) ):  population[copy] = preserve.get()

   print "\nFINISHED TRAINING AND TESTING AT TIME", ctime( time() ), "\n"

   eliteIndividual, loser, elite = population[0], population[0].fitness, population[0].fitness
   ttlFitness = 0
   # determine the elite  individual and the least and greatest fitnesses
   for chrom in population:
      print "INDIVIDUAL", chrom, "HAD FITNESS", chrom.fitness
      ttlFitness += chrom.fitness
      if chrom.fitness > elite:     eliteIndividual, elite = chrom, chrom.fitness
      elif chrom.fitness < loser:   loser = chrom.fitness

   print "\nGREATEST FITNESS", elite
   print "LEAST FITNESS\t", loser
   print "AVERAGE FITNESS\t", ttlFitness/len(population)

   # perform selection, crossover, and mutation
   population = population[0].selection( population, ttlFitness )
   mates = random.sample(range(0,len(population)), len(population))
   for chrom in xrange(0,len(population)-1,2):
      population[mates[chrom]].crossover( population[mates[chrom+1]] )
      for mutant in xrange(chrom,chrom+2): population[mates[mutant]].mutate() 
   
   # duplicate individuals check
   for ind in xrange(len(population) - 1):
      while( population[ind].isDuplicate( ind, population ) == 1 ):
         population[ind].crossover( population, ind )
         population[ind].mutate()

   index = 0
   # elitism, insert the elite individual back into the population
   if len(population)%2 > 0:
      index = population[len(population)-1]
      population[index] = eliteIndividual
   else:
      index = random.randint(0,len(population)-1)
      population[index] = eliteIndividual
      
   print "\nElite Individual inserted at index: ", index
   print "FINISHED GENERATION AT TIME", ctime( time() ), "\n"

   print "\n\n{-----------------------------------END OF GENERATION",generation,"----------------------------------}", "\n\n"

   
    
