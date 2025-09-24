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

from Bio.PDB import PDBParser
from argparse import ArgumentParser
import numpy as np
import random

import cdiotools.rdcs
import cdiotools.ensembles


def vectorForOLC(tensor):
    Sxx, Syy, Szz, Sxy, Sxz, Syz = tensor
    r2 = np.sqrt(2)
    return np.array([Sxx, Syy, Szz, r2 * Sxy, r2 * Sxz, r2 * Syz])


def tensorFromVector(vector):
    r2i = 1.0 / np.sqrt(2)
    return [vector[0], vector[1], vector[2], r2i * vector[3], r2i * vector[4], r2i * vector[5]]


def printSaupes(saupeDict, label):
    print()
    print(label)
    for cond, saupe in saupeDict.items():
        print(f'{cond}: {saupe}')


def printCommonRDCs(commonRDCs):
    print('commonRDCs:')
    for rdcs in commonRDCs:
        print(rdcs)


def printOLCSets(rdcsOLCList):
    print('OLC data sets:')
    for i, dataset in enumerate(rdcsOLCList):
        print(f'{i + 1}: {dataset}')


def printTensorsOLC(tensorsOLC):
    print('tensorsOLC:')
    for i, tensor in enumerate(tensorsOLC):
        print(f'{i + 1}: {tensor}')


def findSaupes(ref, offset, domain, rdcs, debug=False):
    condnames = cdiotools.rdcs.alignment_conditions[0:4]
    saupesAll = cdiotools.rdcs.findSaupes(ref, offset, rdcs, domain)
    return {condname: saupesAll[condname] for condname in condnames}


def findOLC(saupes, rdcs, domain, Vh = None, debug = False):
    condnames = cdiotools.rdcs.alignment_conditions[0:4]
    matrixForOLC = np.row_stack(tuple(vectorForOLC(saupes[cond]) for cond in condnames))
    if Vh is None:
        U, W, Vh = np.linalg.svd(matrixForOLC.T, full_matrices=False)
        #print('W:', W)
    #assert all([((a == b).all() for a, b in zip(commonRDCs[i], commonRDCs2[i])) for i in range(4)])
    commonRDCs = cdiotools.rdcs.get_common_rdcs(domain, rdcs)
    rdcsOLCMat = Vh @ np.row_stack(tuple(commonRDCs))
    rdcsOLCList = [dataset for dataset in rdcsOLCMat]
    saupeVecsOLC = Vh @ matrixForOLC
    print('domain:', domain, '; Vh:\n', Vh)
    print('domain:', domain, '; orig rdcs:\n', np.row_stack(tuple(commonRDCs)))
    print('domain:', domain, '; olc rdcs mat:\n', rdcsOLCMat)
    print('domain:', domain, '; matrixForOLC:\n', matrixForOLC)
    print('domain:', domain, '; saupeVecsOLC:\n', saupeVecsOLC)
    tensorsOLC = [tensorFromVector(vec) for vec in saupeVecsOLC]
    print('domain:', domain, '; tensorsOLC:\n', tensorsOLC)
    print('domain:', domain, '; error:\n', Vh @ np.array([[.2], [.2], [.2], [.2]]))
    olcSaupeList = [cdiotools.rdcs.construct_saupe_matrix(tensor, debug) for tensor in [[0] * 6] * 4 + tensorsOLC]
    return rdcsOLCList, olcSaupeList, Vh


def findOLCFromBothRDC(saupes, rdcs, debug = False):
    condnames = cdiotools.rdcs.alignment_conditions[0:4]
    commonRDCsZ = cdiotools.rdcs.get_common_rdcs('Z', rdcs)
    commonRDCsC = cdiotools.rdcs.get_common_rdcs('C', rdcs)
    commonRDCs = [np.concatenate((commonRDCsZ[icond], commonRDCsC[icond])) for icond in range(4)]

    #commonRDCsMat = np.row_stack(tuple(commonRDCs))
    commonRDCsMat = np.row_stack(tuple(commonRDCsZ))
    U, W, Vh = np.linalg.svd(commonRDCsMat.T, full_matrices=False)
    #print('V:\n', Vh.T)
    #print('W:', W)
    #printCommonRDCs(commonRDCs)
    rdcsOLCMat = Vh @ np.row_stack(tuple(commonRDCs))
    rdcsOLCList = [dataset for dataset in rdcsOLCMat]

    domains = ['Z', 'C']
    numObservedZ = len(cdiotools.rdcs.get_common_residues('Z'))
    numObservedC = len(cdiotools.rdcs.get_common_residues('C'))
    rdcOLCsPerDomain = {'Z': [l[0:numObservedZ] for l in rdcsOLCList], 'C': [l[numObservedZ:numObservedZ+numObservedC] for l in rdcsOLCList]}
    #printOLCSets(rdcsOLCList)

    tensorMat = {domain: np.row_stack(tuple(vectorForOLC(saupes[domain][cond]) for cond in condnames)) for domain in domains}
    saupeVecsOLC = {domain: Vh @ tensorMat[domain] for domain in domains}

    tensorsOLC = {domain: [tensorFromVector(vec) for vec in saupeVecsOLC[domain]] for domain in domains}
    #printTensorsOLC(tensorsOLC)
    olcSaupes = {domain: [cdiotools.rdcs.construct_saupe_matrix(tensor, debug) for tensor in [[0] * 6] * 4 + tensorsOLC[domain]] for domain in domains}
    return rdcOLCsPerDomain, olcSaupes


def main():
    argParser = ArgumentParser()
    argParser.add_argument("--refZ", default='pdbs/1Q2N_Zdomain.pdb', type=str, required=False)
    argParser.add_argument("--refC", default='pdbs/C domain_RT_rotated.pdb', type=str, required=False)
    argParser.add_argument("--refchainZ", default='A', type=str, required=False)
    argParser.add_argument("--refchainC", default='A', type=str, required=False)
    argParser.add_argument("--ranges", default=[24, 37, 53, 68, 95, 108, 112, 126], type=int, nargs=8, required=False)
    argParser.add_argument("--nperturbations", default=None, type=int, required=False)
    argParser.add_argument("--seed", default=None, type=int, required=False)
    argParser.add_argument("--newmethod", default=False, action='store_true')
    argParser.add_argument("--debug", default=False, action='store_true')

    args = argParser.parse_args()

    pdbParser = PDBParser(QUIET=True)

    rangesZ = args.ranges[0:4]
    rangesC = args.ranges[4:8]

    refZ = pdbParser.get_structure('ref', args.refZ)[0][args.refchainZ]
    refC = pdbParser.get_structure('ref', args.refC)[0][args.refchainC]
    offsetRefZ = cdiotools.ensembles.findOffset(rangesZ, rangesC, refZ, {'Z'}, args.debug)
    offsetRefC = cdiotools.ensembles.findOffset(rangesZ, rangesC, refC, {'C'}, args.debug)
    assert offsetRefZ == [-1,-13, None, None]
    assert offsetRefC == [None, None, -71, -71]
    offset = offsetRefZ[0:2] + offsetRefC[2:4]
    assert offset == [-1, -13, -71, -71]

    print(f'Seed: {args.seed}')
    rng = np.random.default_rng(args.seed)

    if args.nperturbations:
        assert args.nperturbations > 0
        perturbed = True
    else:
        perturbed = False

    for ipert in range(args.nperturbations if perturbed else 1):
        expRdcs = cdiotools.rdcs.getExpRdcs()

        if perturbed:
            rdcs = cdiotools.rdcs.add_gaussian_noise_to_rdcs(expRdcs, stddev=0.2, rng=rng)
            print()
            print()
            print(f'Perturbation #{ipert + 1}:')
            print(f'Perturbed RDCs:')
            for domain in ['Z', 'C']:
                for icond in range(4):
                    print(f'domain {domain}, condition {icond}:', ' '.join([str(x) for x in rdcs.rdcs[domain][icond]]))
            print()
            print('First two OLC Saupe tensors in format required for input into BnBmain.m:')

        else:
            rdcs = expRdcs


        #print("****************************RDCS:******************************")
        #print(get_common_rdcs('C', rdcs))
        #print("***************************************************************")

        saupesZ = findSaupes(refZ, offsetRefZ, 'Z', rdcs)
        saupesC = findSaupes(refC, offsetRefC, 'C', rdcs)

        if not perturbed:
            print("\nSaupe Tensors, Z:")
            for cond, tensor in saupesZ.items():
                print(f"{cond}: {tensor}")
            
            print("\nSaupe Tensors, C:")
            for cond, tensor in saupesC.items():
                print(f"{cond}: {tensor}")

        def backcalcBoth(saupesByDomain, olcDataSet):
            #back calculate olc rdc
            olcBackcalc = cdiotools.rdcs.backcalcBoth({'Z':refZ, 'C':refC}, offset, saupesByDomain)
            rmsds = olcBackcalc.rmsd(olcDataSet)
            qs = olcBackcalc.q(olcDataSet)
            print('RMSD, Z:', rmsds['Z'])
            print('RMSD, C:', rmsds['C'])
            print('Q, Z:', qs['Z'])
            print('Q, C:', qs['C'])

        def saupeVecForBnB(s):
            return [s[0,0], s[1,1], s[0,1], s[0,2], s[1,2]]

        # This does the SVD on the RDCs themselves, not the Saupe tensors, and for both domains together
        if args.newmethod:
            olcSets, olcSaupes = findOLCFromBothRDC({'Z': saupesZ, 'C': saupesC}, rdcs)
            for icond in [4,5]:
                print('[', end='')
                strings = []
                for domain in ['Z', 'C']:
                    strings.append(','.join([str(x) for x in saupeVecForBnB(olcSaupes[domain][icond])]))
                print(';'.join(strings), '] ', sep='', end='')
            print()

            #print('olcSaupesZ:')
            #for saupe in olcSaupes['Z']:
            #    print(saupe)
            #print('olcSaupesC:')
            #for saupe in olcSaupes['C']:
            #    print(saupe)

            if not perturbed:
                olcDataSet = cdiotools.rdcs.Rdcs()
                olcDataSet.rdcs['Z'][4:8] = olcSets['Z']
                olcDataSet.rdcs['C'][4:8] = olcSets['C']
                backcalcBoth(olcSaupes, olcDataSet)

        # This is the way it was done for Qi et al 2018: SVD is performed on the vectorized Z domain Saupe tensors
        else:
            olcSetsZOld, olcSaupesZOld, VhZOld = findOLC(saupesZ, rdcs, 'Z')
            # SVD is NOT repeated for the C tensor; the weights from the Z tensor SVD are re-used
            olcSetsCOld, olcSaupesCOld, _ = findOLC(saupesC, rdcs, 'C', Vh = VhZOld)
            fennarioNumber = [1, 2, 2, 2, 3, 3, 3, 4, 4, 4][ipert]
            print(f'sbatch --partition=grisman -A grisman --nodelist=fennario-0{fennarioNumber} -c 4 --mem=300G --job-name=pert{ipert+1} -o out_6_1_4_pert{ipert+1}.txt -e err_6_1_4_pert{ipert+1}.txt batch_search.sh "BnBmain(', end='')
            for saupes in [olcSaupesZOld, olcSaupesCOld]:
                print('[', end='')
                print(';'.join([','.join([str(x) for x in saupeVecForBnB(saupes[icond])]) for icond in [4,5]]), end='')
                print("]',", end='')
            print(f"'pert{ipert+1}')" + '"')

            #print('olcSaupesZOld:')
            #for saupe in olcSaupesZOld:
            #    print(saupe)
            #print('olcSaupesCOld:')
            #for saupe in olcSaupesCOld:
            #    print(saupe)
            if not perturbed:
                olcDataSetOld = cdiotools.rdcs.Rdcs()
                olcDataSetOld.rdcs['Z'][4:8] = olcSetsZOld
                olcDataSetOld.rdcs['C'][4:8] = olcSetsCOld
                backcalcBoth({'Z': olcSaupesZOld, 'C': olcSaupesCOld}, olcDataSetOld)

if __name__ == '__main__':
    main()
