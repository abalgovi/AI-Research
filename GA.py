import random
import pickle
import numpy as np
import threading, Queue, multiprocessing
from time import time, ctime
import GAChromosome, GANetwork
import sklearn.datasets as ds

# The random generator
inp, output = ds.make_classification(n_samples=250,n_features=10,n_classes=4,n_informative=7)

out = list(output)
# this method trains and tests the network on the current set of inputs
def runDatasets( individual, num, ttlRuns, ttlEpochs, ttlDataSets, queue ):

   ttlLocation = np.zeros( 4 )
   
   # ttlCorrect:  the total correct for every run
   ttlCorrect = 0
   
   # loops by number of runs
   for run in xrange(0,ttlRuns):  
      
      # initialize network with 3 layers and 10 input nodes, 20 middle nodes,
      # and 2 output nodes
      network = GANetwork.GANetwork( 3, (10,20,2) )
      
      
      # loops by the number of specified epochs
      for epochs in xrange(0,ttlEpochs):
               
         # train and test the network based on the number of input values
         for dataset in xrange( 0, 200 ):

            #inputs: the inputs  provided by the random number generator
            # outCounter: counter used to track the number of outputs  
            inputs = list(inp[dataset])
            
            # optVal: the desired output value
            optVal = out[dataset]
            
            # train the network for 200 iterations on the current data set
            network.runData( inputs, optVal, individual, 1 )
      

      # netOutput: the output of the network
      netOutput, outCounter = np.empty(0), 0
      
      # test the network
      for dataset in xrange( 200, 250 ):
            
         inputs = list(inp[dataset])
         # optVal: the desired output value
         optVal = out[dataset]
         ttlLocation[optVal] += 1
            
         netOutput = np.append( netOutput, network.runData(inputs, optVal,
         individual,0 ) )
               
         # determine if the output is correct and if it is increment the
         # location value
         if network.convToBinary(optVal) == netOutput[outCounter]: 
            individual.location[optVal] += 1
         outCounter += 1
         
      # tally the total correct for this test run
      ttlCorrect += network.computeAcc( out[200:250], netOutput )
      
   
   # compute the individuals fitness by dividing by the total number of runs
   individual.fitness = ttlCorrect/ttlRuns 

   # determine the final location value
   for dim in xrange(0,ttlLocation.size):
      if ttlLocation[dim] > 0:   individual.location[dim] /= ttlLocation[dim]
      individual.location[dim] = format( individual.location[dim], '.4f')

   # save the individual's results
   f = open("./results/ind" + str(num), 'a')
   f.write( "\nTEST RUN FOR INDIVIDUAL" + str(num) + " " + str(individual) + ":\n" )
   f.write( "\t\t|-----------------------------------------------------------------------------------|")
   f.write( "\n\t\t  Average Fitness:\t\t" + str(individual.fitness) )
   f.write( "\n\t\t  All Location:\t\t\t" + str(ttlLocation) )
   f.write( "\n\t\t  Location:\t\t\t" + str(individual.location) )
   f.write( "\n\t\t|-----------------------------------------------------------------------------------|\n" )
   f.close()
      
   del network
   
   # save in the individual in the multiprocessing queue
   queue.put( individual )


# this method saves individuals who have high fitness values
def saveIndividual( individual ):
   f = open("./FITIND",'a')
   f.write( individual.bitString + "\n" )
   f.close()


# calls rundatasets with the appropriate number of runs and epochs
def runThreads( pop, pack, q ):
   for individual in xrange(pack, pack+4):
      runDatasets( pop[individual], individual, 1,10,5, q )


#-----------------------------------------   BEGIN GENETIC ALGORITHM   ---------------------------------#

population, preserve = [], multiprocessing.Queue()
'''
# read the last generation's population from pickled file
with open('lastGen.pkl', 'rb' ) as input:
   for i in xrange(0,4):   
      population.append(pickle.load(input))

# read the saved archive from the pickled file
t = open( "./archive", 'r' )
population[0].archive = pickle.load(t)
for c in xrange(1,len(population)):   
   population[c].archive = population[0].archive
t.close()
'''

# generate a brand new population
for c in xrange(0,40):  population.append( GAChromosome.GAChromosome( 85, 4 ))

# runs for specified number of generations
for generation in xrange( 0, 1 ):
   
   print "\nSTARTED AT TIME", ctime( time() )
   processes = []
   
   # call the multiprocessing methods and pass in the population as well as 4
   # individuals per iteration
   for pack in xrange(0,len(population), 4):
      processes.append( multiprocessing.Process(target=runThreads,
               args=(population, pack, preserve)))
   
   # start each process  
   for process in processes: process.start()
   
   # syncronize processes
   for process in processes: process.join()
   
   # copy over the newly updated population( after calling runDataSets ) back
   # into population
   for copy in xrange( 0, len(population) ):  population[copy] = preserve.get()
   
   print "FINSIHED AT TIME", ctime( time() )
   
   # perform crossover and mutation
   elite, loser, ttlNovelty, exCounter = 0.0, population[0].novelty, 0, 0
   fittest, leastFit, ttlFitness = 0.0, population[0].fitness, 0.0
   for chrom in population:
      chrom.sparsness( population )
      print "\nAGENT", chrom, "NOVELTY:", format(chrom.novelty, '.4'),"\tFITNESS:", format(chrom.fitness, '.4')
      print "LOCATION:", chrom.location
      ttlNovelty, ttlFitness = ttlNovelty + chrom.novelty, ttlFitness + chrom.fitness
      if chrom.novelty > 0.55:      
         exCounter += 1
         chrom.addToArchive()
         print "\n\nNEW ADDITION TO THE ARCHIVE: ", chrom
      if chrom.fitness > 0.70:       saveIndividual( chrom )
      if chrom.novelty > elite:        elite = chrom.novelty
      elif chrom.novelty < loser:      loser = chrom.novelty
      if chrom.fitness > fittest:      fittest = chrom.fitness
      elif chrom.fitness < leastFit:   leastFit = chrom.fitness

   print "\nNUMBER OF INDIVIDUALS ADDED TO THE ARCHIVE", exCounter, "\n" 
   print "NUMBER OF INDIVIDUALS IN THE ARCHIVE", len(population[0].archive)
   print "GREATEST NOVELTY", elite
   print "LEAST NOVELTY", loser
   print "AVERAGE NOVELTY", ttlNovelty/len(population)
   print "GREATEST FITNESS", fittest
   print "LEAST FITNESS", leastFit
   print "AVERAGE FITNESS", ttlFitness/len(population)

   population = population[0].selection( population, ttlNovelty )
   mates = random.sample(range(0,len(population)), len(population))
   for ind in xrange(0,len(population) -1,2 ):
      population[mates[ind]].crossover( population[mates[ind+1]] )
      for c in xrange( ind, ind+2 ): population[mates[c]].mutate()

   for ind in xrange(len(population) - 1):
      population[ind].isDuplicate( ind, population )
   
   print "\nFINSIHED GENERATION AT TIME", ctime( time() ) 
   print "\n\n\n---------------------------------------------------------------------"
   print "--------------------- END OF GENERATION:", generation,"--------------------"
   print "-------------------------------------------------------------------\n\n\n"

   # save the last generation to a pickle file  
   if generation == 1:
      with open('lastGen.pkl', 'wb') as output:
         for c in population:
            pickle.dump( c, output, pickle.HIGHEST_PROTOCOL )
      with open('archive.pkl','wb') as output:
         for c in population:
            pickle.dump( c, output, pickle.HIGHEST_PROTOCOL )
