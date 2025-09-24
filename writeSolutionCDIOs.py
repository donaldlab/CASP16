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
import cdiotools.cdios
import cdiotools.rotation

def main():
    argParser = ArgumentParser()
    #argParser.add_argument("--gridfn", default='data.qua', type=str, required=False)
    argParser.add_argument("--gridfn", default='grids/quatsForDoS.txt', type=str, required=False)
    argParser.add_argument("--cdiodir", default='cdios', type=str, required=True)
    argParser.add_argument("--pertbasename", default=None, type=str, required=False)
    argParser.add_argument("--alphas", default=False, action='store_true')
    argParser.add_argument("--debug", default=False, action='store_true')

    args = argParser.parse_args()
    cdioPath = cdiotools.cdios.setUpCDIOPath(args.cdiodir, False)
    grid = cdiotools.rotation.readQuats(args.gridfn)
    if args.pertbasename:
        for path in Path('.').glob(args.pertbasename + '*.txt'):
            pertSolmatsByName = cdiotools.cdios.getSolmatsByName(path)
            cdioSubPath = cdioPath / path.stem
            cdioSubPath.mkdir()
            cdiotools.cdios.discretizeSols(grid, pertSolmatsByName, cdioSubPath)
    elif args.alphas:
        cdiotools.cdios.discretizeByAlpha(grid, 101, 2, cdioPath)
    else:
        cdiotools.cdios.discretizeSolutions(grid, cdioPath)

if __name__ == '__main__':
    main()
