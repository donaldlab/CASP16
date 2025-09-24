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
import cdiotools.cdios
import cdiotools.rotation
import cdiotools.util
import csv


def main():
    argParser = ArgumentParser()
    #argParser.add_argument("--gridfn", default='data.qua', type=str, required=False)
    argParser.add_argument("--gridfn", default='grids/quatsForDoS.txt', type=str, required=False)
    argParser.add_argument("--out", default='solutionMixtureModes', type=str, required=False)
    argParser.add_argument("--nproc", default=1, type=int, required=False)
    argParser.add_argument("--debug", default=False, action='store_true')

    args = argParser.parse_args()

    grid = cdiotools.rotation.readQuats(args.gridfn)

    solmatsByName = cdiotools.cdios.getSolmatsByName()
    modesByName = cdiotools.cdios.findModesByName(['1', '2'], solmatsByName)
    for name, mode in modesByName.items():
        print("Name:", name)
        print("Mode:", mode)

    def modesBatch(nameSoldiscPairs):
        nameModeList = []
        for name, soldisc in nameSoldiscPairs:
            mode, _ = cdiotools.cdios.getModeFromCDIO(grid, soldisc)
            nameModeList.append((name, mode))
        return nameModeList

    soldiscsByName = cdiotools.cdios.discretizeByAlpha(grid, 101)
    nameModeList = cdiotools.util.batchRun(modesBatch, soldiscsByName.items(), args.nproc)

    with open(f'{args.out}_{Path(args.gridfn).name}.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        header = ['alpha', 'q1', 'q2', 'q3', 'q4']
        writer.writerow(header)
        for name, mode in nameModeList:
            writer.writerow([round(name, 2)] + list(mode))

if __name__ == '__main__':
    main()
