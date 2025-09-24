"""
This file is part of CDIO Tools: https://github.com/donaldlab/CASP16
Copyright (C) 2025 Bruce Donald Lab, Duke University

CDIO Tools is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License version 2 as published by the Free
Software Foundation.

You should have received a copy of the GNU General Public License along with
CDIO Tools.  If not, see <http://www.gnu.org/licenses/>.

Our lab's software relies on grants for its development, and since visibility
in the scientific literature is essential for our success, we ask that users of
CDIO Tools cite our papers. See the CITATION.cff and README.md documents in
this distribution for more information.

Contact Info:
   Bruce Donald
   Duke University
   Department of Computer Science
   Levine Science Research Center (LSRC)
   Durham
   NC 27708-0129
   USA
   e-mail: www.cs.duke.edu/brd/

<signature of Bruce Donald>, Sept 24, 2025
Bruce Donald, Professor of Computer Science
"""

from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

import cdiotools.cdios
import cdiotools.rotation


class Fit:
    def __init__(self, path, grid, solmat, soldisc):
        self.solmat = solmat
        self.soldisc = soldisc
        self.grid = grid
        matsByName = cdiotools.cdios.getSolmatsByName(path)
        matsList = list(matsByName.values())
        assert len(matsList) == 1
        self.mat = matsList[0]
        cdiosByName = cdiotools.cdios.discretizeSols(self.grid, matsByName)
        cdiosList = list(cdiosByName.values())
        assert len(cdiosList) == 1
        self.cdio = cdiosList[0]

    def modeAngDist(self):
        q1 = cdiotools.cdios.getModeFromBinghamMatrix(self.mat)
        q2 = cdiotools.cdios.getModeFromBinghamMatrix(self.solmat)
        return cdiotools.rotation.rot_angle(q1, q2)

    def feDiff(self):
        assert len(self.soldisc) == len(self.cdio)
        return cdiotools.cdios.findDiffs(self.soldisc, self.cdio, self.grid)[0]

    def anisoDiff(self, unifcdio):
        assert len(unifcdio) == len(self.cdio)
        solAniso = cdiotools.cdios.findDiffs(unifcdio, self.soldisc, self.grid)[0]
        myAniso = cdiotools.cdios.findDiffs(unifcdio, self.cdio, self.grid)[0]
        # I think these are all positive because they're not refined yet
        return myAniso - solAniso

    def modeMatchFeDiff(self):
        def modeRotFromParams(mat):
            return cdiotools.rotation.quatToScipy(cdiotools.cdios.getModeFromBinghamMatrix(mat))
        rotSolMode = modeRotFromParams(self.solmat)
        rotSelfMode = modeRotFromParams(self.mat)
        selfOntoSolInCDIOFrame = rotSolMode * rotSelfMode.inv()
        modeMatchedCDIOsByName = cdiotools.cdios.discretizeSols(self.grid, {None: self.mat}, optionalRot = selfOntoSolInCDIOFrame)
        cdiosList = list(modeMatchedCDIOsByName.values())
        assert len(cdiosList) == 1
        modeMatchedCDIO = cdiosList[0]
        assert len(self.soldisc) == len(modeMatchedCDIO)
        return cdiotools.cdios.findDiffs(self.soldisc, modeMatchedCDIO, self.grid)[0]


def main():
    argParser = ArgumentParser()
    argParser.add_argument("--gridfn", default='grids/quatsForDoS.txt', type=str, required=False)
    argParser.add_argument("--cdiodir", default='pertResultsBnB', type=str, required=False)
    argParser.add_argument("--debug", default=False, action='store_true')

    args = argParser.parse_args()
    grid = cdiotools.rotation.readQuats(args.gridfn)
    solmatsByName = cdiotools.cdios.getSolmatsByName()
    soldiscsByName = cdiotools.cdios.discretizeSols(grid, solmatsByName, None)

    cdiodirpath = Path(args.cdiodir)
    fitsByPertBySol = {}
    angDistByPertBySol = {}
    iperts = range(1, 11)
    solnames = ['1', '2']
    for ipert in iperts:
        fitsByPertBySol[ipert] = {}
        for solname in solnames:
            fn = f'pert{ipert}sol{solname}.txt'
            path = cdiodirpath / fn
            if path.exists():
                fitsByPertBySol[ipert][solname] = Fit(path, grid, solmatsByName[solname], soldiscsByName[solname])
            else:
                print(f'{path.name} not found')
                fitsByPertBySol[ipert][solname] = None

    labelsize = 9 
    #labelsize = 4 

    def boxplot(fn, ylab, func):
        fig, ax = plt.subplots()
        ax.tick_params(labelsize=labelsize)
        ax.set_ylabel(ylab, fontsize=labelsize)
        data = [np.array([func(fitsByPertBySol[i][s]) for i in iperts if fitsByPertBySol[i][s]]) for s in solnames]
        ax.boxplot(data, tick_labels = [f'Solution {s}' for s in solnames], patch_artist=True, showfliers=False, showcaps=False, showbox=False, medianprops=dict(linewidth=0), whiskerprops=dict(linewidth=0))
        for i, p in enumerate(data):
            ax.scatter(np.random.normal(i+1, 0.008, size=len(p)), p, s=14.0, c='blue', alpha=0.5, edgecolor='none', linewidth=0)
        plt.savefig(fn, dpi=300, bbox_inches='tight')
        plt.close()

    plt.rcParams["font.family"] = "serif"
    boxplot('pertFEDiff.eps', 'Free energy difference with solution (kcal/mol)', lambda x: x.feDiff())
    boxplot('pertAnisoDiff.eps', 'Anisotropicity difference with solution (kcal/mol)', lambda x: x.anisoDiff(soldiscsByName['unif']))

if __name__ == '__main__':
    main()

