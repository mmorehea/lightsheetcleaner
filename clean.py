import numpy as np
import cv2
import code
import glob
import os
import tifffile
import sys
from datetime import datetime

def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    '''
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    '''
    if not(0 <= lower_bound < 2**16) and lower_bound is not None:
        raise ValueError(
            '"lower_bound" must be in the range [0, 65535]')
    if not(0 <= upper_bound < 2**16) and upper_bound is not None:
        raise ValueError(
            '"upper_bound" must be in the range [0, 65535]')
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError(
            '"lower_bound" must be smaller than "upper_bound"')
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


inDir = sys.argv[1]
outDir = sys.argv[2]
l = glob.glob(inDir + "*.tif*")
l = [os.path.basename(i) for i in l]
alreadyDone = glob.glob(outDir + "*.tif*")
alreadyDone = [os.path.basename(i) for i in alreadyDone]


l = sorted([i for i in l if i not in alreadyDone])


#this will restrict the run to these frames
l = l[1680:2330]

#change this to adjust front wave width removal
firstWaveWidth = 20


firstWaveAt1 = 0
firstWaveAt2 = 8
firstWaveAt3 = 18
firstWaveAt125 = 710
# create a list of first 5 frames

# X is the frame number 
x= [1, 2, 68]
# Y is the pixel the front wave starts at
y = [1, 6, 384]
# xvals is the number of slices in each frame
xvals = range(72)
startWave = np.interp(xvals, x, y)


endWaveX = [1, 32, 72]
endWaveY = [391, 558, 776]
endWave = np.interp(xvals, endWaveX, endWaveY)

firstStack = tifffile.imread(inDir + l[0])

z, x, y = firstStack.shape
startTime = datetime.now()
for frameNumber, frame in enumerate(l):
	print str(frame) 
	print str(frameNumber) + " / " + str(len(l))
	stack = tifffile.imread(inDir + frame)
	stack = map_uint16_to_uint8(stack)
	for ii in range(z):
	
		#makes first slice totally black, disable by placing # on lines
		#if ii == 0:
		#	stack[ii, :, :] = 0
		
		#makes everything past slice 124 black
		#if ii > 124:
		#	stack[ii:, :, :] = 0
		#	break
		
		start = int(startWave[ii])
		end = int(startWave[ii] + firstWaveWidth)
		stack[ii, :, start:end] = 0
		start = int(endWave[ii])

		stack[ii, :, start:] = 0
		img = stack[ii, :, :]
		stack[ii, :, :] = cv2.fastNlMeansDenoising(img, 500, 7, 21)
	print datetime.now() - startTime
	tifffile.imsave(outDir + os.path.basename(frame), stack)

print "Done!"
