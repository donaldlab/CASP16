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
from scipy.spatial.transform import Rotation as R

globalDebug = False

# volume of SO(3) space
VSO3 = np.pi ** 2

#read quaternions
#the quaternions were generated using the Yershova et al. algorithm which allows a uniform sampling across SO(3)
def readQuats(path):
    quats = []
    with open(path, 'r') as quatstream:
        for line in quatstream:
            quats.append(np.array([float(x) for x in line.split()]))
    return quats


def toScalarFirst(q):
    return np.array([q[3], q[0], q[1], q[2]])


def toScalarLast(q):
    return np.array([q[1], q[2], q[3], q[0]])


def rotateQuatByScipy(q, spRot):
    qRot = R.from_quat(toScalarLast(q))
    rotated = spRot * qRot
    return toScalarFirst(rotated.as_quat())


def quatToScipy(q):
    return R.from_quat(toScalarLast(q))


def rangeToFramePlusCenter(chain, ranges4):

    # Protein spatial information extraction
    def extract_backbone_coords(chain, start_residue, end_residue):
        coords = []
        for i in range(start_residue, end_residue + 1):
            residue = chain[i]
            for atom_name in ["N", "CA", "C"]:
                atom = residue[atom_name]
                if atom.is_disordered():
                    atom.disordered_select('A')
                coords.append(atom.get_coord())
        return np.array(coords)

    # Define Coordinate Frames 
    def define_coordinate_frame(chain, V2, V23, Cterm_II, Nterm_III, Cterm_III):
        towardNtermofIII = chain[Nterm_III]["CA"].get_coord() - chain[Cterm_III]["CA"].get_coord()
        if np.dot(V2, towardNtermofIII) >= 0:
            z_prime = V2
        else:
            z_prime = -V2
        
        towardIIfromIII = chain[Cterm_II]["CA"].get_coord() - chain[Nterm_III]["CA"].get_coord()
        zp_cross_V23 = np.cross(z_prime, V23)
        
        if np.dot(zp_cross_V23, towardIIfromIII) >= 0:
            y_prime = zp_cross_V23
        else:
            y_prime = -zp_cross_V23
        
        x_prime = np.cross(y_prime, z_prime)
        ref_prime_frame = np.column_stack((x_prime, y_prime, z_prime))
        
        coords = np.column_stack((chain[Nterm_III]["CA"].get_coord(), chain[Cterm_III]["CA"].get_coord()))
        coordsInrefPrimeFrame = ref_prime_frame.T @ coords
        assert coordsInrefPrimeFrame[2, 0] >= coordsInrefPrimeFrame[2, 1]
        
        return ref_prime_frame

    # Extracting the back bone atom coordinates
    coordsH2 = extract_backbone_coords(chain, ranges4[0], ranges4[1])
    coordsH3 = extract_backbone_coords(chain, ranges4[2], ranges4[3])

    # Combine and center coordinates
    coordsH2H3 = np.vstack((coordsH2, coordsH3))
    centered_H2H3 = coordsH2H3 - np.mean(coordsH2H3, axis=0)
    centered_H2 = coordsH2 - np.mean(coordsH2, axis=0)

    # Perform SVD on centered coordinates
    _, _, VT_H2H3 = np.linalg.svd(centered_H2H3)
    _, _, VT_H2 = np.linalg.svd(centered_H2)
    basis3_H2H3 = VT_H2H3.T[:, 2]
    basis1_H2 = VT_H2.T[:, 0]
    return define_coordinate_frame(chain, basis1_H2, basis3_H2H3, ranges4[1], ranges4[2], ranges4[3]), np.mean(coordsH2H3, axis=0)


def rangeToFrame(chain, ranges4):
    rv, _ = rangeToFramePlusCenter(chain, ranges4)
    return rv


# Here, 'ref' refers to the chain we want to rotate on to, and 'rot' is the
# chain being rotated onto the other. So if we want to find the rotation that
# takes the ZLBT domain onto the C domain, ZLBT needs to be 'rot' and C is
# 'ref'.
def pdbToRot(ref_chain, rot_chain, refRanges, rotRanges, wrtRot = False, debug = False):
    refFrame = rangeToFrame(ref_chain, refRanges)
    rotFrame = rangeToFrame(rot_chain, rotRanges)
    if debug:
        print(refFrame)
        print(rotFrame)
        print(refFrame.T)
        print(rotFrame.T)
    if wrtRot:
        rotation = rotFrame.T @ refFrame
    else:
        rotation = refFrame @ rotFrame.T
    return rotation


def rot_angle(q1, q2):
    q1norm = np.linalg.norm(q1)
    q2norm = np.linalg.norm(q2)
    assert np.isclose(q1norm, 1.0, rtol = 3e-04)
    assert np.isclose(q2norm, 1.0, rtol = 3e-04)
    q1 /= q1norm
    q2 /= q2norm
    dot_product = np.dot(q1, q2)
    theta = 2 * np.arccos(np.clip(np.abs(dot_product), 0.0, 1.0))
    return theta


#This function is going to read the 3x3 rotation matrix and convert it into quaternions
def rotmat2quat(rotation_matrix, optionalRotation = None, debug = False):
    
    if not isinstance(rotation_matrix, np.ndarray) or rotation_matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must be a 3x3 numpy array")
    
    spRot = R.from_matrix(rotation_matrix)
    if optionalRotation:
        spRot = optionalRotation * spRot
    rot_quat = toScalarFirst(spRot.as_quat())
    if debug:
        print("Quaternion:", rot_quat)
    
    return rot_quat


def rotateChain(chain, rotation):
    rotated = chain.copy()
    for atom in rotated.get_atoms():
        atom.set_coord(rotation.apply(atom.get_coord()))
    return rotated


def writePymolAxes(stream, label, axisSuffix, color, center, frame):
    def axisEndpoint(axis):
        length = 20
        return length * axis + center

    def oneCyl(axis, col):
        colsubstr = f'{col[0]}, {col[1]}, {col[2]}, '
        coords = np.concatenate((center, axisEndpoint(axis)))
        return f'CYLINDER, {np.array2string(coords, separator = ", ").rstrip("]").lstrip("[")}, 0.3, {colsubstr} {colsubstr}'

    x, y, z = frame[:,0], frame[:,1], frame[:,2]

    print(f'axisobj_{label} = [{oneCyl(x, color)} {oneCyl(y, color)} {oneCyl(z, color)}]', file = stream)
    print(f'cyl_text(axisobj_{label}, plain, {np.array2string(axisEndpoint(x), separator = ", ")}, \'x{axisSuffix}\', 0.2, color={color}, axes=[[3,0,0],[0,3,0],[0,0,3]])', file = stream)
    print(f'cyl_text(axisobj_{label}, plain, {np.array2string(axisEndpoint(y), separator = ", ")}, \'y{axisSuffix}\', 0.2, color={color}, axes=[[3,0,0],[0,3,0],[0,0,3]])', file = stream)
    print(f'cyl_text(axisobj_{label}, plain, {np.array2string(axisEndpoint(z), separator = ", ")}, \'z{axisSuffix}\', 0.2, color={color}, axes=[[3,0,0],[0,3,0],[0,0,3]])', file = stream)
    print(f'cmd.load_cgo(axisobj_{label}, \'{label}\')', file = stream)
