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

import csv
import random
import numpy as np
from pathlib import Path
import re

import cdiotools.util
import cdiotools.rotation


#Find free energy difference via calculating the probability density differnce 
#Equations:
#\Delta F = \Delta U - T \Delta S
#\Delta U = -RT \int \left( P_2(\theta) - P_1(\theta) \right) \ln P_1(\theta) \, d\theta
#\Delta S = R \int \left( P_2(\theta) \ln P_2(\theta) - P_1(\theta) \ln P_1(\theta) \right) d\theta
def findDiffs(truepdf, badpdf, grid):
    R = 1.9891e-3
    T = 298.15
    feIntegral = 0
    klIntegral = 0
    jsIntegral = 0
    volumeElement = cdiotools.rotation.VSO3 / len(grid)

    for i, _ in enumerate(grid):
        p1 = truepdf[i]
        p2 = badpdf[i]
        lnp1 = np.log(p1)
        lnp2 = np.log(p2)
        feIntegral += p2 * (lnp2 - lnp1)
        klIntegral += p1 * (lnp1 - lnp2)
        M = 0.5 * (p1+p2)
        lnM = np.log(M)
        jsIntegral += 0.5 * (p1*(lnp1 - lnM) + p2*(lnp2 - lnM))

    feIntegral *= R * T * volumeElement
    klIntegral *= volumeElement
    jsIntegral *= volumeElement

    return [feIntegral, klIntegral, jsIntegral]


def findAndWriteCDIO(cdioFn, param, spread, solname, cdioPath, cdioNameFn):
    cdio = None
    foundFile = False
    cdiofilepath = cdioPath / cdioNameFn(param, spread, solname) if cdioPath else None
    if cdiofilepath and cdiofilepath.is_file():
        foundFile = True
        with open(cdiofilepath, 'r') as cdioInFile:
            print(f'found file {cdiofilepath}, reading', flush=True)
            cdio = np.array([float(line) for line in cdioInFile])
    else:
        print(f'Not found: {cdiofilepath}', flush=True)
    if cdio is None:
        print(f'computing cdio for {param}, {spread}', flush=True)
        cdio = cdioFn(param, spread, solname)
    if cdio is None:
        raise RuntimeError(f'Error finding or computing cdio for {param}, {spread}')
    if cdioPath and not foundFile:
        with open(cdiofilepath, 'w') as cdioOutFile:
            print(f'writing cdio to file {cdiofilepath}', flush=True)
            cdioOutFile.write('\n'.join(str(d) for d in cdio))
    return cdio


# Write csv files with integral divergences (free energy, KL divergence, JS divergence)
# Inputs:
#   soldiscs: discretized CDIOs of ground-truth solutions
#   grid: list of quaternions representing grid over SO(3)
#   spreads: list of minor-eigenvalues to use as kernel parameters
#   csvPrefix: string to prepend to output csv files
#   cdioNameFn: function that returns the name for a CDIO file, given a CDIO
#       parameter and a kernel spread parameter
#   cdioFn: function that returns a discretized CDIO, given a CDIO parameter
#      and a kernel spread parameter
#   rowNameFn: function that returns a label for a row of output csv files,
#      given a CDIO parameter
#   paramList: a list of CDIO parameters
#   nproc: number of processors to use
#   cdioPath: path to directory to save CDIO files to
# Return value: None
# Notes:
#   The term "CDIO parameter" in the above is deliberately vague. In the case
#   of predictor ensembles, it means the subdirectories for each ensemble. In
#   the case of kernelized sampled solutions, it means number of samples
#   crossed with the different solutions in solmats.
def writeIntegralDiffs(soldiscsByName, grid, spreads, csvPrefix, cdioNameFn, cdioFn, rowNameFn, paramList, nproc, cdioPath, nowrite = False, debug = False):
    with (
            (open(f'{csvPrefix}_freeEnergyDiffs.csv', 'w') if not nowrite else nullcontext()) as feFile,
            (open(f'{csvPrefix}_kullbackLeiblerDivs.csv', 'w') if not nowrite else nullcontext()) as klFile,
            (open(f'{csvPrefix}_jensenShannonDivs.csv', 'w') if not nowrite else nullcontext()) as jsFile
            ):

        def startFile(file, diffName):
            if file:
                writer = csv.writer(file)
                row = ['ensemble'] + [f'{diffName}, sol {solname}, spread {s}' for solname in soldiscsByName for s in spreads]
                writer.writerow(row)
                return writer
            else:
                return None

        feWriter = startFile(feFile, 'dFE')
        klWriter = startFile(klFile, 'KL')
        jsWriter = startFile(jsFile, 'JS')

        def findDiffsBatch(paramSpreadNameDiscTuples):
            diffs = []
            for param, spread, name, disc in paramSpreadNameDiscTuples:
                cdio = findAndWriteCDIO(cdioFn, param, spread, name, cdioPath, cdioNameFn)
                diffs.append(findDiffs(disc, cdio, grid))
            return diffs

        # Order here needs to match how the column names are defined above
        paramSpreadNameDiscTuples = [(param, spread, name, disc) for param in paramList for name, disc in soldiscsByName.items() for spread in spreads]
        byParam = {param: [] for param in paramList}
        for paramSpreadNameDiscTuple, diffs in zip(paramSpreadNameDiscTuples, cdiotools.util.batchRun(findDiffsBatch, paramSpreadNameDiscTuples, nproc)):
            param, spread, name, disc = paramSpreadNameDiscTuple
            byParam[param].append(diffs)

        def writeLine(param, diffsRow, file, writer, index):
            if writer:
                row = [rowNameFn(param)] + [diff[index] for diff in diffsRow]
                writer.writerow(row)
                file.flush()

        for param, diffList in byParam.items():
            writeLine(param, diffList, feFile, feWriter, 0)
            writeLine(param, diffList, klFile, klWriter, 1)
            writeLine(param, diffList, jsFile, jsWriter, 2)
            print(f'Found integrals for {rowNameFn(param)}')


def bingParamsFromBnB(path):
    with open(path, 'r') as file:
        for line in file:
            if line == 'Paramters of the best-fit Bignham distribution(Ls, quaL, quaR):\n':
                #floatpattern = r"([-+]?(?:\d*\.*\d+))" 
                floatpattern = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)" # from https://docs.python.org/3/library/re.html#simulating-scanf
                LsPattern   = r'      Ls: \[' + r' '.join([floatpattern for i in range(3)]) + r'\]'
                quaLPattern = r'    quaL: \[' + r' '.join([floatpattern for i in range(4)]) + r'\]'
                quaRPattern = r'    quaR: \[' + r' '.join([floatpattern for i in range(4)]) + r'\]'
                LsString = next(file).rstrip()
                quaLString = next(file).rstrip()
                quaRString = next(file).rstrip()
                LsMatch = re.fullmatch(LsPattern, LsString)
                quaLMatch = re.fullmatch(quaLPattern, quaLString)
                quaRMatch = re.fullmatch(quaRPattern, quaRString)
                assert LsMatch.lastindex == 3, str(LsMatch)
                assert quaLMatch.lastindex == 4, str(quaLMatch)
                assert quaRMatch.lastindex == 4, str(quaRMatch)
                yield findBingMat(
                        [float(LsMatch.group(i)) for i in range(1, 4)],
                        [float(quaLMatch.group(i)) for i in range(1, 5)],
                        [float(quaRMatch.group(i)) for i in range(1, 5)]
                        )


def findBingMat(conc, qL, qR):
    #Yang's Rotation is an additional rotation applied to the bingham matrix, this is due to before everything 
    #is calculated in the PDB frame, but we want everything in the Z domain frame
    yangRotation = np.array([
        [1, 0, 0, 0],
        [0, 0.7179424849229702, 0.5975391483281293, 0.3589865224229396],
        [0, -0.5116735783191485, 0.09961914423048843, 0.8527017172796637],
        [0, 0.4734073365676751, -0.7948943550037976, 0.37993093283318946]
        ])
    ourRotation = np.array([
        [1, 0, 0, 0],
        [0, 0.7063401, 0.6154996, 0.34861308],
        [0, -0.52957416, 0.13351427, 0.8372648],
        [0, 0.46912575, -0.7765637, 0.42055896]
        ])
    def findEigenvecMat(qL, qR):
        a, b, c, d = qL
        p, q, r, s = qR
        ML = np.array([
            [a, -b, -c, -d],
            [b, a, -d, c],
            [c, d, a, -b],
            [d, -c, b, a]
            ])
        MR = np.array([
            [p, -q, -r, -s],
            [q, p, s, -r],
            [r, -s, p, q],
            [s, r, -q, p]
            ])
        #print("ML:\n", ML)
        #print("MR:\n", MR)
        #print("ML @ MR:\n", ML @ MR)
        return ML @ MR
    M = findEigenvecMat(qL, qR)
    conc0 = -np.sum(conc)
    concFull = np.insert(conc, 0, conc0)
    #L is the Bingham distribution parameter
    L = np.diag(concFull)

    #print("L:\n", L)
    #return yangRotation @ M.T @ L @ M @ yangRotation.T

    #print("rot.MT:\n", ourRotation @ M.T)
    #print("M.rotT:\n", M @ ourRotation.T)

    # MT are column eigenvectors
    # then rot.MT puts those vecs in the correct frame
    # then (rot.MT)T = M.rotT is like M but in the correct frame
    # We know (M.rotT)T = rot.MT
    # Therefore: MTLM in the correct frame is rot.MT.L.M.rotT

    return ourRotation @ M.T @ L @ M @ ourRotation.T


#deprecated
solmats = [
        findBingMat([-2.4688, -1.2312, 1.3312], [-0.7943, 0.5497, 0.1107, -0.2340], [-0.8208, 0.4998, -0.1839, -0.2065]),
        findBingMat([-3.5062, -0.4312, 1.2688], [-0.5775, -0.8059, 0.1249, 0.0379], [-1.1092e-17, 0.0340, 0.6142, 0.7884]),
        np.zeros((4, 4))
        ]
solmatnames = ['1', '2', 'unif']


def getSolmatsByName(path = None):
    if path:
        tmpSolmats = [bing for bing in bingParamsFromBnB(path)]
        tmpSolmatnames = [path.stem + '_' + str(i) for i in range(1, len(tmpSolmats) + 1)]
    else:
        tmpSolmats = solmats
        tmpSolmatnames = solmatnames
    return {name: mat for mat, name in zip(tmpSolmats, tmpSolmatnames)}


# exp(q^T X q); calculate the probability density for the Bingham distribution
# optionalRot represents a rotation of the CDIO itself, so when evaluting
# we need to apply the inverse rotation to the grid points themselves.
def evalBinghamExp(parameterMat, quat, optionalRot = None):
    if optionalRot:
        quat = cdiotools.rotation.rotateQuatByScipy(quat, optionalRot.inv())
    return np.exp(quat @ parameterMat @ quat)


# Integrate a given function on a regular grid over SO(3).
def integrate(pdf, grid, debug = False):

    assert len(pdf) == len(grid)
    assert not np.any(np.isnan(pdf))
    assert not np.any(np.isinf(pdf))
    
    # Compute volume element and the integral
    volumeElement = cdiotools.rotation.VSO3 / len(grid)
    integral = pdf.sum() * volumeElement
    
    if debug:
        expected_integral = pdf.mean() * cdiotools.rotation.VSO3 
        if not np.isclose(integral, expected_integral):
            raise AssertionError(f"Integration mismatch: integral={integral}, expected={expected_integral}")
    
    return integral


# Calculate probability densities of a given Bingham distribution on the
# points of a regular grid.
def discretizeBingham(paramMat, grid, optionalRot = None):
    unnormalized = np.array([evalBinghamExp(paramMat, quat, optionalRot) for quat in grid])
    normalizationConstant = integrate(unnormalized, grid)
    densities = unnormalized / normalizationConstant
    return densities


# Discretizes all hard-coded solution matrices over a regular grid, returns a
# list. deprecated in favor of discretizeSols.
def discretizeSolutions(grid, cdioPath = None):
    soldiscs = []
    for i, solmat in enumerate(solmats):
        soldisc = discretizeBingham(solmat, grid)
        soldiscs.append(soldisc)
        if cdioPath:
            with open(cdioPath / f'cdio_sol{solmatnames[i]}.txt', 'w') as cdioFile:
                cdioFile.write('\n'.join(str(d) for d in soldisc))
    return soldiscs


# Discretizes all hard-coded solution matrices over a regular grid
def discretizeSols(grid, solmatsByNameToDisc, cdioPath = None, optionalRot = None):
    soldiscsByName = {}
    for name, solmat in solmatsByNameToDisc.items():
        soldisc = discretizeBingham(solmat, grid, optionalRot)
        soldiscsByName[name] = soldisc
        if cdioPath:
            with open(cdioPath / f'cdio_{name}.txt', 'w') as cdioFile:
                cdioFile.write('\n'.join(str(d) for d in soldisc))
    return soldiscsByName


# Need to update to just accept a solmats by name dict Currently this allows
# for only a subset of solmatByName to be requested through solnames parameter
# Actually I think this can be deprecated and replaced by discretizeSols.
def discretizeSolutionsByName(grid, solnames=solmatnames):
    solmatsByName = getSolmatsByName()
    return {name: discretizeBingham(solmatsByName[name], grid) 
            for name in solnames}


# Like discretizeBingham, but for when we need the evaluations on a
# set of points that is not a regular grid
def discretizeBinghamAtPoints(paramMat, quats, normalizationConstant):
    unnormalized = np.array([evalBinghamExp(paramMat, quat) for quat in quats])
    return unnormalized / normalizationConstant


def findAlphas(num, digits):
    return np.round(np.linspace(0.0, 1.0, num), digits)


def findAlphaString(alpha, digits):
    return f'{alpha:.{digits}f}'


def discretizeByAlpha(grid, num, digits, cdioPath = None):
    soldiscs = discretizeSolutionsByName(grid, ['1', '2'])
    alphas = findAlphas(num, digits)
    byAlpha = {findAlphaString(alpha, digits): alpha * soldiscs['1'] + (1.0 - alpha) * soldiscs['2'] for alpha in alphas}
    if cdioPath:
        for alpha, soldisc in byAlpha.items():
            with open(cdioPath / f'cdio_alpha{alpha}.txt', 'w') as cdioFile:
                cdioFile.write('\n'.join(str(d) for d in soldisc))
    return byAlpha


def findNormalizationConstant(paramMat, grid):
    unnormalized = np.array([evalBinghamExp(paramMat, quat) for quat in grid])
    return integrate(unnormalized, grid)


def solutionsAtPointsByName(points, grid, solnames, solmatsByName = getSolmatsByName()):
    def solDensAtPoints(solname):
        sol = solmatsByName[solname]
        normalizationConstant = findNormalizationConstant(sol, grid)
        return discretizeBinghamAtPoints(sol, points, normalizationConstant)
    return {name: solDensAtPoints(name) for name in solnames}


def findWeightsByName(grid, solnames):
    soldiscs = discretizeSolutions(grid)
    assert len(soldiscs) == len(solmatnames)
    soldiscByName = {name: disc for name, disc in zip(solmatnames, soldiscs)}
    weightsByName = {}
    for solname in solnames:
        soldisc = soldiscByName[solname]
        #weights = [dens * cdiotools.rotation.VSO3 / len(soldisc) for dens in soldisc]
        weights = soldisc * cdiotools.rotation.VSO3 / len(soldisc)
        assert np.isclose(sum(weights), 1.0)
        weightsByName[solname] = weights
    return weightsByName


# np.eigh returns eigenvectors in ascending order
def findIthSmallestEigenvalEigenvec(paramMat, index):
    eigenvalues, eigenvectors = np.linalg.eigh(paramMat)
    return eigenvectors[:, index]


def getModeFromBinghamMatrix(binghamMatrix):
    return findIthSmallestEigenvalEigenvec(binghamMatrix, 3)


def getModeFromCDIO(grid, densities):
    if len(grid) != len(densities):
        raise ValueError("The length of grid should be the same as the length of the densities")
    
    #Find the largest density in densities
    max_index = np.argmax(densities)
    mode_grid_point = grid[max_index]
    max_density = densities[max_index]
    return mode_grid_point, max_density


def findModesByName(names, solmatsByName = getSolmatsByName()):
    return {name: getModeFromBinghamMatrix(solmatsByName[name]) for name in names}


#Kernelize predictor's discrete strucutre, where the quaternion is the mode 
def findKernel(mode_quat, lambda_val, debug = False):
    if not isinstance(mode_quat, np.ndarray) or mode_quat.shape != (4,) or not np.isclose(np.linalg.norm(mode_quat), 1.0, rtol = 3e-04):
        print(mode_quat)
        raise ValueError("mode_quat must be a rotation quaternion in the form of a length-4 numpy array (it should also be in scalar-first format).")
    vvT = np.outer(mode_quat, mode_quat)
    return lambda_val * (np.identity(4) - vvT)


def findSRSQuats(paramMat, grid, nsamp):
    discCDIO = discretizeBingham(paramMat, grid)
    maxDens = max(discCDIO)
    samplequats = []
    shuffledOrder = random.sample(range(len(grid)), len(grid))
    complete = False
    for i in shuffledOrder:
        test = random.uniform(0, maxDens)
        if test <= discCDIO[i]:
            samplequats.append(grid[i])
        if len(samplequats) == nsamp:
            complete = True
            break
    return samplequats if complete else None


# Sample through experimentally determined CDIO
def findSampledCDIO(paramMat, grid, nsamp, spread):
    densities = np.zeros(len(grid))
    samplequats = random.sample(grid, nsamp)
    solSampled = discretizeBingham(paramMat, samplequats)
    for i, quat in enumerate(samplequats):
        kernel = findKernel(quat, spread)
        densities += solSampled[i] * discretizeBingham(kernel, grid)
    densities /= solSampled.sum()
    assert np.isclose(integrate(densities, grid), 1.0)
    return densities


def writeKernelCDIOs(grid, spreads, cdioPath):
    for spread in args.spreads:
        cdio = discretizeBingham(findKernel(np.array([.5]*4), spread), grid)
        with open(cdioPath / f'cdio_kernel_{spread}.txt', 'w') as cdioFile:
            cdioFile.write('\n'.join(str(d) for d in cdio))


def setUpCDIOPath(cdiodirname, nowrite):
    cdioPath = None if nowrite else Path(cdiodirname)
    if cdioPath.is_file():
        raise RuntimeError(f'{cdioPath} is a file, not a directory')
    elif not cdioPath.is_dir():
        print(f'{cdioPath} does not exist; creating')
        cdioPath.mkdir()
    return cdioPath
