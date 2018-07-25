import sys, math, random, struct

if (len(sys.argv) < 3 or len(sys.argv) > 5):
   print("Usage: " + sys.argv[0] + " <number of rows> <output file> [ <max value> = 10, <columns num> = 8 ]")
   exit()

dataRows = long(sys.argv[1])
outputPath = sys.argv[2]
maxint = 10
columnNum = 8
if (len (sys.argv) > 3):
    maxint = int(sys.argv[3])
if (len (sys.argv) > 4):
    columnNum = int(sys.argv[4])

tenthRows = dataRows / 10
cnt = 0
with open(outputPath, 'wb') as output:
    while cnt < dataRows:
        for col in range(0,columnNum):
           if col > 0:
              output.write(",")
           output.write(str(random.randint(1, maxint)))
        output.write('\n')
        cnt = cnt + 1
#         if cnt % tenthRows == 0:
#             sys.stdout.write('\rProgress: %d%%' % (cnt * 100 / dataRows))
#             sys.stdout.flush()
#     sys.stdout.write('\n')
    sys.stdout.flush()
