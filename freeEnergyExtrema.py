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

from argparse import ArgumentParser
from pathlib import Path
import cdiotools.rotation
import cdiotools.cdios
import sys
import numpy as np


# Find kernelized CDIOs for each predictor ensemble and compute integral
# differences/divergences between them and the ground truth Bingham solutions


def main():
    argParser = ArgumentParser()
    #argParser.add_argument("--preddir", type=str, required=True)
    #argParser.add_argument("--spreads", default=[-(2**n) for n in range(3, 8)], type=int, nargs='+', required=False)
    argParser.add_argument("--spreads", default=[-(2**n) for n in [5]], type=int, nargs='+', required=False)
    #argParser.add_argument("--gridfn", default='data.qua', type=str, required=False)
    argParser.add_argument("--gridfn", default='grids/quatsForDoS.txt', type=str, required=False)
    argParser.add_argument("--nproc", default=1, type=int, required=False)
    argParser.add_argument("--debug", default=False, action='store_true')

    args = argParser.parse_args()

    grid = cdiotools.rotation.readQuats(args.gridfn)
    soldiscsByAlpha = cdiotools.cdios.discretizeByAlpha(grid, 101, 2)
    soldiscsByName = cdiotools.cdios.discretizeSolutionsByName(grid)
    alphas = soldiscsByAlpha.keys()


    for spread in args.spreads:
        print(f'Spread: {spread}\n')

        maxFEDiffs = {}
        modeMinMixFEDiffs = {}
        unifFEDiffs = {}
        for alpha, soldisc in soldiscsByAlpha.items():
            quatAtMin = grid[np.argmin(soldisc)]
            quatAtMode = grid[np.argmax(soldisc)]
            kernelAtMin = cdiotools.cdios.findKernel(quatAtMin, spread)
            kernelAtMode = cdiotools.cdios.findKernel(quatAtMode, spread)
            discKernelAtMin = cdiotools.cdios.discretizeBingham(kernelAtMin, grid)
            discKernelAtMode = cdiotools.cdios.discretizeBingham(kernelAtMode, grid)
            discKernelModeMinMix = 0.5 * (discKernelAtMin + discKernelAtMode)
            maxFEDiff, _, _ = cdiotools.cdios.findDiffs(soldisc, discKernelAtMin, grid)
            modeMinMixFEDiff, _, _ = cdiotools.cdios.findDiffs(soldisc, discKernelModeMinMix, grid)
            unifFEDiff, _, _ = cdiotools.cdios.findDiffs(soldisc, soldiscsByName['unif'], grid)
            maxFEDiffs[alpha] = maxFEDiff
            modeMinMixFEDiffs[alpha] = modeMinMixFEDiff
            unifFEDiffs[alpha] = unifFEDiff

        print('maxdFE = {')
        for alpha in alphas:
            print(f'    "{alpha}": {maxFEDiffs[alpha]},')
        print('}\n')

        print('modeMinMixdFE = {')
        for alpha in alphas:
            print(f'    "{alpha}": {modeMinMixFEDiffs[alpha]},')
        print('}\n')

        print('isodFE = {')
        for alpha in alphas:
            print(f'    "{alpha}": {unifFEDiffs[alpha]},')
        print('}\n')

    solmatsByName = cdiotools.cdios.getSolmatsByName()
    for spread in args.spreads:
        for name in ['1', '2']:
            soldisc = soldiscsByName[name]
            quatAtMin = cdiotools.cdios.findIthSmallestEigenvalEigenvec(solmatsByName[name], 0)
            quatAtMode = cdiotools.cdios.getModeFromBinghamMatrix(solmatsByName[name])
            kernelAtMin = cdiotools.cdios.findKernel(quatAtMin, spread)
            kernelAtMode = cdiotools.cdios.findKernel(quatAtMode, spread)
            discKernelAtMin = cdiotools.cdios.discretizeBingham(kernelAtMin, grid)
            discKernelAtMode = cdiotools.cdios.discretizeBingham(kernelAtMode, grid)
            discKernelModeMinMix = 0.5 * (discKernelAtMin + discKernelAtMode)
            maxFEDiff, _, _ = cdiotools.cdios.findDiffs(soldisc, discKernelAtMin, grid)
            modeFEDiff, _, _ = cdiotools.cdios.findDiffs(soldisc, discKernelAtMode, grid)
            unifFEDiff, _, _ = cdiotools.cdios.findDiffs(soldisc, soldiscsByName['unif'], grid)
            modeMinMixFEDiff, _, _ = cdiotools.cdios.findDiffs(soldisc, discKernelModeMinMix, grid)
            print(f'For spread {spread}, maximum possible free energy difference for solution {name}: {maxFEDiff}')
            print(f'For spread {spread}, free energy difference with mode-only for solution {name}: {modeFEDiff}')
            print(f'For spread {spread}, free energy difference with uniform for solution {name}: {unifFEDiff}')
            print(f'For spread {spread}, free energy difference with mode/min mixture for solution {name}: {modeMinMixFEDiff}')

    for spread in args.spreads:
        quatArbitrary = np.array([1., 0., 0., 0.])
        kernelArbitrary = cdiotools.cdios.findKernel(quatArbitrary, spread)
        discKernelArbitrary = cdiotools.cdios.discretizeBingham(kernelArbitrary, grid)
        kernelFEDiffWithUnif, _, _ = cdiotools.cdios.findDiffs(soldiscsByName['unif'], discKernelArbitrary, grid)
        print(f'For spread {spread}, free energy difference with arbitrary kernel for uniform: {kernelFEDiffWithUnif}')


if __name__ == '__main__':
    main()
