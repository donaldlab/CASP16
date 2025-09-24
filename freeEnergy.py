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
import cdiotools.ensembles
import cdiotools.rotation
import cdiotools.cdios


# Find kernelized CDIOs for each predictor ensemble and compute integral
# differences/divergences between them and the ground truth Bingham solutions


def main():
    argParser = ArgumentParser()
    argParser.add_argument("--preddir", type=str, required=True)
    argParser.add_argument("--spreads", default=[-(2**n) for n in [5]], type=int, nargs='+', required=False)
    argParser.add_argument("--gridfn", default='grids/quatsForDoS.txt', type=str, required=False)
    argParser.add_argument("--refZ", default='pdbs/1Q2N_Zdomain.pdb', type=str, required=False)
    argParser.add_argument("--refchainZ", default='A', type=str, required=False)
    argParser.add_argument("--refchainC", default='A', type=str, required=False)
    argParser.add_argument("--ranges", default=[24, 37, 53, 68, 95, 108, 112, 126], type=int, nargs=8, required=False)
    argParser.add_argument("--cdiodir", type=str, required=False)
    argParser.add_argument("--nproc", default=1, type=int, required=False)
    argParser.add_argument("--nowrite", default=False, action='store_true')
    argParser.add_argument("--debug", default=False, action='store_true')

    args = argParser.parse_args()

    cdioPath = cdiotools.cdios.setUpCDIOPath(args.cdiodir, args.nowrite)
    refRangesZAdj, rangesZ, rangesC = cdiotools.ensembles.setUpRefRanges(args.refZ, args.ranges, args.refchainZ, args.debug)

    grid = cdiotools.rotation.readQuats(args.gridfn)

    soldiscsByName = cdiotools.cdios.discretizeByAlpha(grid, 101, 2) | cdiotools.cdios.discretizeSolutionsByName(grid, ['unif'])
    
    # Functions giving file and row names related to predictor-ensemble CDIOS,
    # as well as a function wrapper that partially applies appropriate
    # arguments to findPredictorCDIO
    def cdioNamePred(subdir, spread, _):
        return f'cdio_{subdir.name}_{spread}.txt'
    def subdirToCDIO(subdir, spread, _):
        return cdiotools.ensembles.findPredictorCDIO(subdir, grid, spread, rangesZ, rangesC, refRangesZAdj, args.debug)
    def rowNamePred(subdir):
        return subdir.name

    dirpath = Path(args.preddir)
    cdiotools.cdios.writeIntegralDiffs(
            soldiscsByName,
            grid,
            args.spreads,
            dirpath.name,
            cdioNamePred,
            subdirToCDIO,
            rowNamePred,
            list(dirpath.glob("*.cleaned")),
            args.nproc,
            cdioPath,
            args.nowrite
            )

if __name__ == '__main__':
    main()
