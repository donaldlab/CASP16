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
import cdiotools.rotation
import cdiotools.cdios


# Sample ground-truth Bingham solutions and then kernelize them, and also
# compute integral differences/divergences between them and the ground
# truth Bingham solutions


def main():
    argParser = ArgumentParser()
    argParser.add_argument("--spreads", default=[-(2**n) for n in [5]], type=int, nargs='+', required=False)
    argParser.add_argument("--gridfn", default='quatsForDoS.txt', type=str, required=False)
    argParser.add_argument("--cdiodir", default='cdios', type=str, required=False)
    argParser.add_argument("--nproc", default=1, type=int, required=False)
    argParser.add_argument("--nowrite", default=False, action='store_true')
    argParser.add_argument("--debug", default=False, action='store_true')

    args = argParser.parse_args()

    cdioPath = cdiotools.cdios.setUpCDIOPath(args.cdiodir, args.nowrite)

    grid = cdiotools.rotation.readQuats(args.gridfn)

    solmatsByName = cdiotools.cdios.getSolmatsByName()
    soldiscsByName = cdiotools.cdios.discretizeSols(grid, solmatsByName)
    
    # Functions giving file and row names related to CDIOs derived from
    # sampling and kernelizing ground-truth Bingham solutions, as well as a
    # function that partially applies arguments to findSampledCDIO to do
    # the actual sampling and kernelizing. These functions are to be passed
    # to writeIntegralDiffs, which will call them later as appropriate.
    def cdioNameSamp(nsamp, spread, solname):
        return f'cdio_samp_sol{solname}_{nsamp}_{spread}.txt'
    def nsampToCDIO(nsamp, spread, solname):
        #return cdiotools.cdios.findSampledCDIO(cdiotools.cdios.solmatsByName[solname], grid, nsamp, spread)
        return cdiotools.cdios.findSampledCDIO(solmatsByName[solname], grid, nsamp, spread)
    def rowNameSamp(nsamp):
        return f'{nsamp} samples'

    paramList = [2**n for n in range(0, 16)]
    cdiotools.cdios.writeIntegralDiffs(
            soldiscsByName,
            grid,
            args.spreads,
            'samp2', 
            cdioNameSamp, 
            nsampToCDIO, 
            rowNameSamp,
            paramList,
            args.nproc,
            cdioPath,
            args.nowrite
            )


if __name__ == '__main__':
    main()
