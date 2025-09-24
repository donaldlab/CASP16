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
from Bio import PDB
import numpy as np

import cdiotools.rotation
import cdiotools.pdbutils
import cdiotools.cdios


ZH2seq = ['N', 'E', 'E', 'Q', 'R', 'N', 'A', 'F', 'I', 'Q', 'S', 'L', 'K', 'D']
ZH3seq = ['Q', 'S', 'A', 'N', 'L', 'L', 'A', 'E', 'A', 'K', 'K', 'L', 'N', 'D', 'A', 'Q']
CH2seq = ['E', 'E', 'Q', 'R', 'N', 'G', 'F', 'I', 'Q', 'S', 'L', 'K', 'D', 'D']
CH3seq = ['S', 'K', 'E', 'I', 'L', 'A', 'E', 'A', 'K', 'K', 'L', 'N', 'D', 'A', 'Q']


#Here, We are finding the offsets that we can use on helix range, which we use them on frame calculation.
#We are trying to match the sequence to the referece structure by trying different offset data sets onto the predictor's sequence
def findOffset(rangesZ, rangesC, chain, domains, debug = False):
    def checkOffset(offset):
        seq = []
        indices = []
        if 'Z' in domains:
            seq += ZH2seq + ZH3seq
            indices += (list(range(rangesZ[0] + offset[0], rangesZ[1] + offset[0] + 1))
                + list(range(rangesZ[2] + offset[1], rangesZ[3] + offset[1] + 1)))
        if 'C' in domains:
            seq += CH2seq + CH3seq
            indices += (list(range(rangesC[0] + offset[2], rangesC[1] + offset[2] + 1))
                + list(range(rangesC[2] + offset[3], rangesC[3] + offset[3] + 1)))
        
        allmatch = True
        for i, expect in enumerate(seq):
            if indices[i] not in chain:
                allmatch = False
                if debug:
                    print('Chain does not contain expected residues for this offset trial')
                break
            predID = cdiotools.pdbutils.oneLetter[chain[indices[i]].get_resname()]
            if predID != seq[i]:
                allmatch = False
                #if debug:
                #    print(f'first non-match: offset: {offset}; i: {i}; indices[i]: {indices[i]}; got: {predID}; expected {seq[i]}')
                break
        return allmatch

    #If we are only considering one domain(Reference testing)
    if domains == {'Z'}:
        offsets = [ 
                [-1, -13, None, None],
                [12, 17, None, None] # 2LR2
                ]
    elif domains == {'C'}:
        offsets = [ [None, None, -71, -71] ]
    elif domains == {'Z', 'C'}:
        offsets = [
                [0, 0, 0, 0],
                [-1, -1, -1, -1],
                [12, 12, 12, 12],
                [-1, 4, 4, 4],
                [0, 5, 5, 5] # for pooledRef output
                #[12, 17, 17, 17]
                ]
    else:
        raise RuntimeError('Unexpected domains list')

    for otry in offsets:
        if checkOffset(otry):
            return otry
    
    print(chain, flush=True)
    for res in chain:
        print(f'{res.get_id()[1]}: {res.get_resname()}', flush=True)
    raise RuntimeError('Offset not found')


def adjustRange(rangeA, rangeB, offset):
    rangeAAdj = None if not rangeA else [rangeA[0] + offset[0], rangeA[1] + offset[0], rangeA[2] + offset[1], rangeA[3] + offset[1]]
    rangeBAdj = None if not rangeB else [rangeB[0] + offset[2], rangeB[1] + offset[2], rangeB[2] + offset[3], rangeB[3] + offset[3]]
    return rangeAAdj, rangeBAdj


def getPredChain(pdbpath, parser, debug = False):
    if debug:
        print('About to read', pdbpath, flush=True)
    predStruct = parser.get_structure('X', pdbpath)[0]
    chains = list(predStruct.get_chains())
    if len(chains) == 1:
        predictor = chains[0]
    else:
        raise RuntimeError(f'More than one chain in predictor structure {pdbpath}')
    return predictor


# TODO: Factor out common parts of next two funcs
def genStructurePaths(subdir, debug = False):
    populationsPath = subdir / Path('populations.txt')
    if not populationsPath.exists():
        raise RuntimeError(f'Error: populations.txt not found for {subdir}')
    with open(populationsPath) as pdblist:
        for pdbnameline in pdblist:
            filepath = Path(pdbnameline.split()[0])
            pdbpath = subdir / filepath
            yield pdbpath


def weightedChain(subdir, popfileline, parser):
    filepath = Path(popfileline.split()[0])
    #filepath = filepathOrig if not args.reprotonated else filepathOrig.with_stem(filepathOrig.stem + '_his_strip_reduce')
    weight = float(popfileline.split()[1])
    pdbpath = subdir / filepath
    predictor = getPredChain(pdbpath, parser)
    return predictor, weight


def genWeightedChains(subdir, parser, debug = False):
    populationsPath = subdir / Path('populations.txt')
    if not populationsPath.exists():
        raise RuntimeError(f'Error: populations.txt not found for {subdir}')
    with open(populationsPath) as pdblist:
        #Looping over discrete strucutres
        for pdbnameline in pdblist:
            yield weightedChain(subdir, pdbnameline, parser)


#Generate a CDIO on predictor's ensemble
def findPredictorCDIO(subdir, grid, spread, rangesZ, rangesC, refRangesZAdj, rotation = None, debug = False):
    parser = PDB.PDBParser(QUIET=True)
    densities = np.zeros(len(grid))
    totWeight = 0.0
    # TODO This is where everything integral-related should be parallelized, NOT the outer layers.
    for predictor, weight in genWeightedChains(subdir, parser, debug):
        offsetPred = findOffset(rangesZ, rangesC, predictor, {'Z', 'C'}, debug)
        predRangesZAdj, predRangesCAdj = adjustRange(rangesZ, rangesC, offsetPred)
        
        #Finding the relative orientation between the Predictor's Z domain and C domain
        rotmatPredZToPredC = cdiotools.rotation.pdbToRot(predictor, predictor, predRangesCAdj, predRangesZAdj, wrtRot=True)
        quatPredZToPredC = cdiotools.rotation.rotmat2quat(rotmatPredZToPredC, rotation, debug)
        kernel = cdiotools.cdios.findKernel(quatPredZToPredC, spread)
        densities += weight * cdiotools.cdios.discretizeBingham(kernel, grid)
        totWeight += weight
    densities /= totWeight
    assert np.isclose(cdiotools.cdios.integrate(densities, grid), 1.0)
    return densities


def setUpRefRanges(refZfn, ranges, refchainZ, debug = False):
    pdbParser = PDB.PDBParser(QUIET=True)
    rangesZ = ranges[0:4]
    rangesC = ranges[4:8]
    refZ = pdbParser.get_structure('ref', refZfn)[0][refchainZ]
    offsetRefZ = findOffset(rangesZ, rangesC, refZ, {'Z'}, debug)
    assert offsetRefZ == [-1,-13, None, None]
    refRangesZAdj, _ = adjustRange(rangesZ, None, offsetRefZ)
    return refRangesZAdj, rangesZ, rangesC


# Calculate difference in experimental C-domain Saupe tensor and a C-domain
# Saupe tensor calculated by averaging appropriately-rotated ZLBT-frame Saupe
# tensors for each structure in a given predictor ensemble. Calculations follow
# Yan et al. 2005.
# Inputs:
#   correct: an experimentally-determined C-domain Saupe tensor or OLC
#   equivalent
#   estimated: a Saupe tensor calculated for a predictor ensemble, by
#   averaging. 
# Return value: Tuple of statistics following tables in Yan et al. 2005.
# Note: The two Saupe tensors should derive either from the same experimental
# alignment condition or from the same OLC.
def saupeDiff(correct, estimated):
    eigs = [np.linalg.eigh(s) for s in [correct, estimated]]
    for eig in eigs:
        assert eig[0][0] <= eig[0][1] and eig[0][1] <= eig[0][2]
    def ds(eig):
        return (
                0.5 * eig[0][2],
                (1.0 / 3.0) * (eig[0][1] - eig[0][0])
                )
    def axisForIndex(eig, i):
        vec = eig[1][:, i]
        return vec / np.linalg.norm(vec)
    def xax(eig):
        return axisForIndex(eig, 1)
    def yax(eig):
        return axisForIndex(eig, 0)
    def zax(eig):
        return axisForIndex(eig, 2)
    def axdot(axfn):
        return np.dot(axfn(eigs[0]), axfn(eigs[1]))
    def axangle(axfn):
        return np.arccos(axdot(axfn))
    perc = 100 * (1 - (2 * axangle(xax) / np.pi) * (1 - axdot(zax)))
    dsCorrect = ds(eigs[0])
    dsEstimated = ds(eigs[1])
    dadiff = 100 * (dsCorrect[0] - dsEstimated[0]) / dsCorrect[0]
    drdiff = 100 * (dsCorrect[1] - dsEstimated[1]) / dsCorrect[1]
    return dadiff, drdiff, axangle(zax), axangle(xax), axangle(yax), perc


# For each structure in a predictor ensemble, rotate the
# experimentally-determined ZLBT-frame Saupe tensor, according to the
# inter-domain orientation for that structure, and find the population-weighted
# average of the resulting C-domain frame Saupe tensors.
# Inputs:
#   subdir: directory containing populations.txt file for given ensemble
#   refZ: BioPython chain for the reference Z domain from which the ZLBT Saupe
#       tensor was derived
#   parser: a BioPython PDB parser
#   rangesZ, rangesC: initial guesses for beginning and end residues of helices
#       II and III for ZLBT and C domains of the predictor structure, before
#   refRangesAdj: correct beginning and end residues of reference ZLBT
#       structure
# Return value: Row of statistics comparing these Saupe tensors to
# experimentally-derived
def findAvgSaupes(subdir, refZ, parser, rangesZ, rangesC, refRangesZAdj, debug = False):
    avgSaupesC = [np.zeros((3, 3)) for i in range(len(saupes['Z']))]
    for predictor, weight in genWeightedChains(subdir, parser, debug):
        offsetPred = findOffset(rangesZ, rangesC, predictor, {'Z', 'C'}, debug)
        predRangesZAdj, predRangesCAdj = adjustRange(rangesZ, rangesC, offsetPred)
        
        rotPredZToPredC = cdiotools.rotation.pdbToRot(predictor, predictor, predRangesCAdj, predRangesZAdj)
        rotPredZToRefZ = cdiotools.rotation.pdbToRot(refZ, predictor, refRangesZAdj, predRangesZAdj)
        rotRefCToPredC = rotPredZToRefZ @ rotPredZToPredC @ rotPredZToRefZ.T
        for i, saupeZ in enumerate(saupes['Z']):
            avgSaupesC[i] += weight * (rotRefCToPredC.T @ saupeZ @ rotRefCToPredC)
    if debug:
        print(f'{subdir} average C Saupe tensors, OLC 1 & 2:')
        print(avgSaupesC[4])
        print(avgSaupesC[5])
    row = [subdir]
    for i, saupeC in enumerate(saupes['C']):
        row += saupeDiff(saupeC, avgSaupesC[i])
    return row


def fixLabels(data, labels):
    newdata = {}
    newlabels = []
    for oldlabel in labels:
        newlabel = oldlabel.replace("T1200TS", "").replace(".cleaned", "")
        newdata[newlabel] = data[oldlabel]
        newlabels.append(newlabel)
    return newdata, newlabels
