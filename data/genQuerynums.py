import subprocess,sys,math,random,io,os

if (len(sys.argv) < 2):
   print("Usage: " + sys.argv[0] + " <max int of distribution> <num predicates> <query type (AND(0)-OR(1))>")
   exit()

maxint = 10
num_predicates = 8
query_type = 0
if (len (sys.argv) > 1):
    maxint = int(sys.argv[1])
if (len (sys.argv) > 2):
    num_predicates = int(sys.argv[2])
if (len (sys.argv) > 3):
    query_type = int(sys.argv[3])

selectivity = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1']

for p in range(1, num_predicates+1):
    #sys.stdout.write('Following are predicate probs for %d predicates:' %(p))
    sys.stdout.write('\n')
    for s in  selectivity:
            if query_type == 0:
                probability = float(s) ** (1/float(p))
                comparewith = int(maxint - (maxint * probability))
                #print(p,s,comparewith)
                print "-d="+str(p),"-s="+str(s),"-mx="+str(comparewith)
            else:
                probability = (1 - ((1- float(s) )** (1/float(p))))
                comparewith = int(maxint - (maxint * probability))
                print(p,s,comparewith)
