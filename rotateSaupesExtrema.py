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

import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
import csv
import itertools
import math
from joblib import Parallel, delayed
import cdiotools.ensembles
import cdiotools.rdcs
import cdiotools.rotation
import cdiotools.util

def main():
    argParser = ArgumentParser()
    argParser.add_argument("--gridfn", default='grids/quatsForDoS.txt', type=str, required=False)
    argParser.add_argument("--refC", default='pdbs/C domain_RT_rotated.pdb', type=str, required=False)
    argParser.add_argument("--refchainC", default='A', type=str, required=False)
    argParser.add_argument("--ranges", default=[24, 37, 53, 68, 95, 108, 112, 126], type=int, nargs=8, required=False)
    argParser.add_argument("--nproc", default=1, type=int, required=False)
    argParser.add_argument("--debug", default=False, action='store_true')

    args = argParser.parse_args()

    pdbParser = PDBParser(QUIET=True)

    rangesZ = args.ranges[0:4]
    rangesC = args.ranges[4:8]

    refC = pdbParser.get_structure('ref', args.refC)[0][args.refchainC]
    offsetRefC = cdiotools.ensembles.findOffset(rangesZ, rangesC, refC, {'C'}, args.debug)
    assert offsetRefC == [None, None, -71, -71]

    grid = cdiotools.rotation.readQuats(args.gridfn)
    saupeBackcalcRdcs = cdiotools.rdcs.backcalculate_rdcs(refC, offsetRefC, np.identity(3), {'C'}, cdiotools.rdcs.saupes['C'], debug=args.debug)

    def findStatsRot(index, q, altRot = None):
        if index % 5000 == 0:
            print(f'count: {index}')
        rot = R.from_quat(cdiotools.rotation.toScalarLast(q)).as_matrix()
        rotatedZSaupes = [rot.T @ s @ rot for s in cdiotools.rdcs.saupes['Z']]
        rdcsBackcalcOneRotated = cdiotools.rdcs.backcalculate_rdcs(refC, offsetRefC, np.identity(3), {'C'}, rotatedZSaupes, debug=args.debug)
        return rdcsBackcalcOneRotated.concatLinreg(cdiotools.rdcs.expRdcs, ['OLC1', 'OLC2'])['C']
        return rdcsBackcalcRefRotated.rms()['C'], rdcsBackcalcRefRotated.range()['C'], rdcsBackcalcRefRotated.rmsd(saupeBackcalcRdcs)['C']

    def findStatsRotBatch(tuples):
        return [findStatsRot(i, q) for i, q in tuples]

    stats = cdiotools.util.batchRun(findStatsRotBatch, enumerate(grid), args.nproc)

    pearsonrs = np.array([x.rvalue for x in stats])
    slopes = np.array([x.slope for x in stats])
    yintercepts = np.array([x.intercept for x in stats])

    def printSummaries(stat, summaryFn):
        index = summaryFn(stat)
        print(index, stat[index])
    def printExtrema(statArray, name):
        print(f'{name}, smallest:')
        printSummaries(statArray, np.argmin)
        print(f'{name}, largest:')
        printSummaries(statArray, np.argmax)

    printExtrema(pearsonrs, 'Pearson Correlations')
    printExtrema(slopes, 'Slopes')
    printExtrema(yintercepts, 'Y-intercepts')

if __name__ == '__main__':
    main()
