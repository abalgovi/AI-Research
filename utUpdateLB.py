import GANetwork
import GAFChromosome
import sklearn.datasets as ds
#----------------------------{ UNIT TEST FOR UPDATE WEIGHT }-------------------#


inp, output = ds.make_classification(n_samples=250,n_features=10,n_classes=4,n_informative=7)
out = list(output)

inputs = list(inp[0][0:2])
net = GANetwork.GANetwork(3, (2,3,2))

individual = GAFChromosome.GAFChromosome(85)
net.runData( inputs, out[0], individual, 1  )




