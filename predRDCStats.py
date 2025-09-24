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
import csv
import re
from Bio.PDB import PDBParser
from argparse import ArgumentParser
from pathlib import Path

import cdiotools.ensembles
import cdiotools.rdcs
import cdiotools.rotation
import cdiotools.util
import cdiotools.plotting


saupes = cdiotools.rdcs.makeSaupes()


def perStructureRDCs(predChain, refZ, refC, rangesZ, rangesC, offsetRefZ, offsetRefC, refRangesZAdj, debug = False):
    offsetPred = cdiotools.ensembles.findOffset(rangesZ, rangesC, predChain, {'Z', 'C'}, debug)
    predRangesZAdj, predRangesCAdj = cdiotools.ensembles.adjustRange(rangesZ, rangesC, offsetPred)
    rotPredZToRefZ = cdiotools.rotation.pdbToRot(refZ, predChain, refRangesZAdj, predRangesZAdj)
    rotPredZToPredC = cdiotools.rotation.pdbToRot(predChain, predChain, predRangesCAdj, predRangesZAdj)
    rotRefCToPredC = rotPredZToRefZ @ rotPredZToPredC @ rotPredZToRefZ.T
    return cdiotools.rdcs.backcalculate_rdcs(refC, offsetRefC, rotRefCToPredC, {'C'}, saupes['Z'], debug=debug)


def main():
    argParser = ArgumentParser()
    argParser.add_argument("--preddir", default=None, type=str, required=True)
    argParser.add_argument("--preddir2", default=None, type=str, required=False)
    argParser.add_argument("--refZ", default='pdbs/1Q2N_Zdomain.pdb', type=str, required=False)
    argParser.add_argument("--refC", default='pdbs/C domain_RT_rotated.pdb', type=str, required=False)
    argParser.add_argument("--refchainZ", default='A', type=str, required=False)
    argParser.add_argument("--refchainC", default='A', type=str, required=False)
    argParser.add_argument("--ranges", default=[24, 37, 53, 68, 95, 108, 112, 126], type=int, nargs=8, required=False)
    argParser.add_argument("--rdcsdir", default='rdcs', type=str, required=False)
    argParser.add_argument("--plotsdir", default='rmbplots', type=str, required=False)
    argParser.add_argument("--nproc", default=1, type=int, required=False)
    argParser.add_argument("--noplot", default=False, action='store_true')
    argParser.add_argument("--assumerdcs", default=False, action='store_true')
    argParser.add_argument("--debug", default=False, action='store_true')

    args = argParser.parse_args()

    pdbParser = PDBParser(QUIET=True)

    rangesZ = args.ranges[0:4]
    rangesC = args.ranges[4:8]

    refZ = pdbParser.get_structure('X', args.refZ)[0][args.refchainZ]
    refC = pdbParser.get_structure('X', args.refC)[0][args.refchainC]
    offsetRefZ = cdiotools.ensembles.findOffset(rangesZ, rangesC, refZ, {'Z'}, args.debug)
    offsetRefC = cdiotools.ensembles.findOffset(rangesZ, rangesC, refC, {'C'}, args.debug)
    assert offsetRefZ == [-1,-13, None, None]
    assert offsetRefC == [None, None, -71, -71]
    refRangesZAdj, _ = cdiotools.ensembles.adjustRange(rangesZ, None, offsetRefZ)

    expRdcs = cdiotools.rdcs.getExpRdcs()

    rdcsDirPath = Path(args.rdcsdir)
    if not rdcsDirPath.exists():
        rdcsDirPath.mkdir()

    plotsPath = Path(args.plotsdir)
    if not plotsPath.exists():
        plotsPath.mkdir()

    for saupe, solutionName in [('C', 'OLC'), ('sol1', '1'), ('sol2', '2')]:
        rdcsPath = rdcsDirPath / (solutionName + '_rdcs.csv')
        if rdcsPath.exists():
            rdcs = cdiotools.rdcs.Rdcs(rdcsPath)
        else:
            assert not args.assumerdcs
            rdcs = cdiotools.rdcs.backcalculate_rdcs(refC, offsetRefC, np.identity(3), {'C'}, saupes[saupe], debug=args.debug)
            rdcs.writeToFile(rdcsPath)
        linreg = rdcs.concatLinreg(expRdcs, ['OLC1', 'OLC2'])['C']
        outPath = plotsPath / f'{solutionName}_rmb_err.png'
        if not args.noplot:
            cdiotools.plotting.rmbPlot(rdcs, expRdcs, outPath, linreg, solutionName)

    def backcalcEnsemble(subdir):
        rdcsPath = rdcsDirPath / (subdir.name + '_rdcs.csv')
        print(rdcsPath)
        if rdcsPath.exists():
            rdcs = cdiotools.rdcs.Rdcs(rdcsPath)
        else:
            assert not args.assumerdcs
            def sumRDCsBatch(popfilelines):
                weightsum = 0.0
                rdcs = cdiotools.rdcs.Rdcs()
                for popfileline in popfilelines:
                    predChain, weight = cdiotools.ensembles.weightedChain(subdir, popfileline, pdbParser)
                    rdcs += weight * perStructureRDCs(predChain, refZ, refC, rangesZ, rangesC, offsetRefZ, offsetRefC, refRangesZAdj, debug=args.debug)
                    weightsum += weight
                return rdcs, weightsum

            populationsPath = subdir / Path('populations.txt')
            if not populationsPath.exists():
                raise RuntimeError(f'Error: populations.txt not found for {subdir}')
            with open(populationsPath) as popfile:
                popfilelines = popfile.readlines()
            weightsum = 0.0
            rdcs = cdiotools.rdcs.Rdcs()
            for chunkRDCs, chunkWeights in cdiotools.util.batchGen(sumRDCsBatch, popfilelines, args.nproc):
                rdcs += chunkRDCs
                weightsum += chunkWeights
            #print(weightsum, '... should be 1.0')
            assert np.isclose(weightsum, 1.0, atol=1e-02)
            rdcs.writeToFile(rdcsPath)
        return subdir, rdcs

    predRDCsList = [backcalcEnsemble(subdir) for subdir in Path(args.preddir).glob("*.cleaned")]
    predRDCsList.sort(key = lambda pair: pair[0]) # Sort by ensemble ID

    with open('rdcStats.csv', 'w') as rdcStatsFile, open('qAndRdip.tex', 'w') as qAndRdipFile:
        rdcStats = csv.writer(rdcStatsFile)
        rdcStats.writerow(['Predictor Number','Pearson Correlation','Slope','Y Intercept','Q','Rdip simple','Rdip Z','Rdip C'])
        qAndRdipFile.write(r'    \textbf{Predictor ID} & \textbf{$Q$} & \textbf{$R_{\textrm{dip}}$} \\ \hline' + '\n')
        for subdir, rdcs in predRDCsList:
            linreg = rdcs.concatLinreg(expRdcs, ['OLC1', 'OLC2'])['C']
            q = rdcs.concatQ(expRdcs, ['OLC1', 'OLC2'])['C']
            rdipSimple = rdcs.concatRdipSimple(expRdcs, ['OLC1', 'OLC2'])['C']
            saupeListZ = [saupes['Z'][i] for i in [4, 5]]
            saupeListC = [saupes['C'][i] for i in [4, 5]]
            rdipZ = rdcs.concatRdip(expRdcs, ['OLC1', 'OLC2'], saupeListZ)['C']
            rdipC = rdcs.concatRdip(expRdcs, ['OLC1', 'OLC2'], saupeListC)['C']
            outPath = plotsPath / f'{subdir.name}_rmb_err.png'
            if not args.noplot:
                cdiotools.plotting.rmbPlot(rdcs, expRdcs, outPath, linreg)
            print(subdir.name)
            predNum = re.fullmatch('T1200TS([0-9]{3}).cleaned', subdir.name).group(1)
            rdcStats.writerow([predNum, linreg.rvalue, linreg.slope, linreg.intercept, q, rdipSimple, rdipZ, rdipC])
            qAndRdipFile.write(f'    {predNum} & {round(q, 2)} & {round(rdipC, 2)} \\\\ \\hline\n')

if __name__ == '__main__':
    main()
