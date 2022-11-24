import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model,Parameter,Parameters
from .lineShapes import (
    findPeaks, getPeakWidth, findGaps
)

def sin(x, amp:float, fre:float, phase: float = 0, offset:float = 0)->np.ndarray:
    """
    sin function with offset

    # Parameters

    amp: amplitude
    fre: frequency
    phase: phase
    offset: offset

    # Examples
    >>> x = np.linspace(0,100,201)
    >>> y = sin(x,10,0.1,1,1)
    """
    return amp*np.sin(x*fre + phase) + offset

def guassian():
    pass

class sinModel(Model):
    def __init__(self,independent_vars = ['x'], prefix = '', nan_policy = 'raise', **kwargs):
        kwargs.update({ 
            'prefix': prefix, 
            'nan_policy': nan_policy,
            'independent_vars': independent_vars})
        
        super().__init__(sin, **kwargs)
                
        amp = Parameter("amp",1,min = 0)
        fre = Parameter("fre",1,min = 0)
        phase = Parameter("phase",min = -np.pi/2, max = np.pi/2)
        offset = Parameter("offset",0)

        self.defaultParameters = Parameters()
        self.defaultParameters.add_many(
            amp, fre, phase, offset
        )

    def guess(x,y):
        """
        Guess the initial value for fitting
        """
        _,rabiTime, amp, offset = getPeakWidth(x,y)
        fre = np.pi/rabiTime
        amp = amp/2

        # estimate the initial phase
        phase = np.arcsin((y[0]-offset)/amp)-fre*x[0]
        dNum = len(x)
        isRise = x[:3] < x[3:6]      

        amp = Parameter("amp", amp, True, 0, 2*amp)
        fre = Parameter("fre", fre, True, 0, 3*fre)
        phase = Parameter("phase", phase)
        offset = Parameter("offset",offset, True, -2*abs(offset), 2*abs(offset))


    def Fit(self, x, y, isPlot = True,**kwargs):
        """
        init fit parameters can be assigned in kwargs parameters.
        it must start witn key = value
        key can be parameter
        
        in the following way:
        1. set it directly
        >>> model.Fit(x, y, amp = 1, phase = 2)

        2. set it using parameter
        >>> amp = parameter("amp",1,min = 0, max = 100)
        >>> model.Fit(x, y, amp = amp)
        """
        params = self.getParams(kwargs)
        fitResult = self.fit(y, params, x = x)
        if isPlot:
            self.plotFig(x,y,fitResult.best_fit)
        return fitResult

    def plotFig(self, x, y, yFit):
        fig, axes = plt.subplots()
        axes.plot(x, y,"ro", label = "raw data")
        axes.plot(x, yFit, label = "fit")
        axes.legend()
        fig.show()

    def testPlot(self, x ,y,**kwargs):
        params = self.getParams(kwargs)
        yFit = self.eval(params,x=x)
        self.plotFig(x,y,yFit)
    
    def getParams(self, kwargs):
        params = self.defaultParameters.copy()

        # update parameters
        for k,v in kwargs.items():
            if k in params.keys():

                newParameters = Parameters()
                if type(v) in [float, int]:
                    temp = Parameter(k,v)
                    newParameters.add_many(temp)

                elif type(v) in [list, tuple]:
                    if len(v) == 3:
                        # params[k].value,params[k].min,params[k].max = v
                        # params[k].setup_bounds()
                        temp = Parameter(k,v[0],True,v[1],v[2])
                        newParameters.add_many(temp)

                elif type(v) is Parameter:
                    newParameters.add_many(v)

                elif type(v) is Parameters:
                    newParameters.add_many(*v.values())

            if newParameters:
                params.update(newParameters)
        return params

if __name__ == "__main__":
    x = np.linspace(0,100,201)
    amp = 10
    fre = 0.1
    phase = np.pi/3
    offset = 0.1

    y = sin(x, amp, fre, phase, offset) + np.random.normal(scale = 1, size = x.size)

    model = sinModel()
    model.Fit(x,y)
    model.testPlot(x,y, amp = 10, fre = 0.1,phase = 1)


