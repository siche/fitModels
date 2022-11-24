
import numpy as np
from scipy.signal import find_peaks
from typing import Union

inType = Union[list, np.ndarray]
def sin(x, amp, fre, phase, offset):
    return amp*np.sin(fre*x+phase)+offset


def getPeakWidth(x: inType,y: inType)->tuple:

    x = np.array(x) if type(x) is list else y
    y = np.array(y) if type(y) is list else y
    dNum = len(x)

    xSort = np.sort(x)
    ySort = np.sort(y)
    pickRange = np.ceil(0.1*dNum)

    peakValue = ySort[-pickRange:].mean()
    minValue = ySort[:pickRange].mean()
    peakHeight = peakValue - minValue
    peakIdx = np.where(y > peakValue)[0].mean()
    
    centerFre = x[peakIdx]
    lowerData = x[y < (peakHeight/2+minValue)]
    hfWidth = min(abs(lowerData - centerFre))

    return centerFre, hfWidth, peakHeight, minValue

def findPeaks(x: inType,y: inType, minGap: float = None, minHeight: float = None):
    step = abs(np.diff(x)[0])
    minGap = 10*step if minGap is None else minGap
    minHeight = 2*np.mean(y) if minHeight is None else minHeight
    minDis = np.ceil(minGap/step)

    peakLocs,peakProperties = find_peaks(y,distnace = minDis, height = 2*np.mean(y))
    
    return peakLocs


def findGaps(x:inType)->np.ndarray:
    """
    if there exist jumps in data
    it will return the index of jump
    """
    steps = np.diff(x)
    step = steps[0]
    gaps = steps[steps > 10*step]
    gapNum = len(gaps)

    gapLoc = np.zeros(gapNum,dtype = int)
    for i in range(gapNum):
        gapLoc[i] = np.where(steps == gaps[i])[0][0]
    return gapLoc



