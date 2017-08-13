import numpy as np
import cv2
import code
import glob
import os
import tifffile
from datetime import datetime

inDir = sys.argv[1]
outDir = sys.argv[2]
l = glob.glob(inDir + "*.tif*")
l = [os.path.basename(i) for i in l]
alreadyDone = glob.glob(outDir + "*.tif*")
alreadyDone = [os.path.basename(i) for i in alreadyDone]


l = [i for i in l if i not in alreadyDone]

firstWaveWidth = 20
firstWaveAt1 = 0
firstWaveAt2 = 8
firstWaveAt3 = 18
firstWaveAt125 = 710
# create a list of first 5 frames

x= [1, 2, 125]
y = [8, 18, 710]
xvals = range(125)
startWave = np.interp(xvals, x, y)

endWaveX = [1, 2, 125]
endWaveY = [393, 400, 1080]
endWave = np.interp(xvals, endWaveX, endWaveY)

firstStack = tifffile.imread(inDir + l[0])

z, x, y = firstStack.shape
startTime = datetime.now()
for frameNumber, frame in enumerate(l):
	print frame + " / " + len(l)
	stack = np.uint8(tifffile.imread(inDir + frame))
	for ii in range(z):
		if ii == 0:
			stack[ii, :, :] = 0
		if ii > 124:
			stack[ii:, :, :] = 0
			break
		start = int(startWave[ii])
		end = int(startWave[ii] + 20)
		stack[ii, :, start:end] = 0
		start = int(endWave[ii])

		stack[ii, :, start:] = 0
		img = stack[ii, :, :]
		stack[ii, :, :] = cv2.fastNlMeansDenoising(img, 500, 7, 21)
	print datetime.now() - startTime
	tifffile.imsave(outDir + os.path.basename(frame), stack)

print "Done!"