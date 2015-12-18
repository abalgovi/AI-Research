f = open("./fitness")
lines = f.readlines()
avgGen = []
for line in lines:
   line = float( line )
   avgGen.append( float( format( line, '.4f' ) ))

print avgGen, "\n", len(avgGen)


