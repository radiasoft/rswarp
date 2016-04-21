import numpy as np
import h5py as h5
from warp import *


class TransverseDiagnostic:
    """
        Setup a diagnostic to measure and record statistical quantities for 
        the transverse beam distribution.
        Can split data by turn.
        Currently will not accomodate multiple species.
    """
    def __init__(self):

        self.zpos = []

        self.xstd = []
        self.xpstd = []
        self.ystd = []
        self.ypstd = []

        self.xxpavg = []
        self.yypavg = []

        self.xemit = []
        self.yemit = []

    def record(self):
        """
            Record 6D particle coordinates
            Must derive directly from these:
            x std
            y std
            x emittance
            y emittance
            x-xp std
            y-yp std
        """

        xArray = np.array(getx())
        xpArray = np.array(getxp())
        yArray = np.array(gety())
        ypArray = np.array(getyp())
        zArray = np.array(getz())
        zpArray = np.array(getuz() / 5.344286E-22)

        self.xemit.append(np.sqrt(np.average(xArray**2) * np.average(xpArray**2) - np.average(xArray * xpArray)**2))
        self.yemit.append(np.sqrt(np.average(yArray**2) * np.average(ypArray**2) - np.average(yArray * ypArray)**2))

        self.xstd.append(np.std(xArray))
        self.ystd.append(np.std(yArray))

        self.xxpavg.append(np.average(xArray * xpArray))
        self.yypavg.append(np.average(yArray * ypArray))

        self.zpos.append(np.average(zArray))


    def derivedQuantities(self):
        """
            Normally called at end of a run.
            Derive quantities from data  taken from transverseDiag function
            x beta
            y beta
            x alpha
            y alpha
        """

        self.betax = np.array(self.xstd)**2 / np.array(self.xemit)
        self.betay = np.array(self.ystd)**2 / np.array(self.yemit)
        self.alphax = -1. * np.array(self.xxpavg) / np.array(self.xemit)
        self.alphay = -1. * np.array(self.yypavg) / np.array(self.yemit)

    def resetArrays(self):
        """
            Reset global lists.
        """

        self.zpos = []

        self.xstd = []
        self.xpstd = []
        self.ystd = []
        self.ypstd = []

        self.xxpavg = []
        self.yypavg = []

        self.xemit = []
        self.yemit = []

    def dataWrite(self, turn, fileName='beamEnvelope.h5', resetArrays=True):

        self.derivedQuantities()

        f = h5.File(fileName, 'a')

        f.create_dataset('%s/spos' % turn, data=self.zpos)

        f.create_dataset('%s/betax' % turn, data=self.betax)
        f.create_dataset('%s/betay' % turn, data=self.betay)
        f.create_dataset('%s/alphax' % turn, data=self.alphax)
        f.create_dataset('%s/alphay' % turn, data=self.alphay)

        f.create_dataset('%s/sigx' % turn, data=self.xstd)
        f.create_dataset('%s/sigy' % turn, data=self.ystd)
        f.create_dataset('%s/epsx' % turn, data=self.xemit)
        f.create_dataset('%s/epsy' % turn, data=self.yemit)

        if resetArrays:
            self.resetArrays()
