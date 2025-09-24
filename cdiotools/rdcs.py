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
import scipy
import os
import pandas as pd

import cdiotools.pdbutils

globalDebug = False


#4npdFasta = 'ADNKFNKEQQNAFYEILHLPNLTEEQRNGFIQSLKDDPSVSKEILAEAKKLNDAQAPK'
#1q2nFasta = 'VDNKFNKEQQNAFYEILHLPNLNEEQRNAFIQSLKDDPSQSANLLAEAKKLNDAQAPK'
zlbtc = 'MVDNKFNKEQQNAFYEILHLPNLNEEQRNAFIQSLKDYIDTNNDGAYEGDELQSANLLAEAKKLNDAQAPKADNKFNKEQQNAFYEILHLPNLTEEQRNGFIQSLKDDPSVSKEILAEAKKLNDAQAPK'#Fasta Res. No


scaling_factor = 1e4

# Physical constants for NH bond
mu0 = 4 * np.pi * 10**-7  # unit: N/A^2
gammaH = 2.67513 * 10**8  # unit: Hz/T
gammaN = 2.7116 * 10**7   # unit: Hz/T
rNH = 1.02 * 10**-10      # unit: m
h_bar = 6.626068 * 10**-34 / (2 * np.pi)  # unit: m^2*kg/s

# Calculate the coupling constant K for NH bond
KNH = -((mu0 * gammaH * gammaN * h_bar) / (4 * np.pi**2 * rNH**3))


# Alignment conditions for Z and C domains
alignment_conditions = [
    "ZLBT_C/Dy",    # Set 1
    "ZLBT_C/Tb",    # Set 2
    "NHis-ZLBT_C/Dy", # Set 3
    "NHis-ZLBT_C/Tb",  # Set 4
    "OLC1",
    "OLC2",
    #"OLC3",
    #"OLC4"
    ]


def findicondList(conditionList):
    icondList = []
    for icond, cond in enumerate(alignment_conditions):
        if cond in conditionList:
            icondList.append(icond)
    return icondList


# Function to construct Saupe matrices.
#Saupe Matrices/Tensor represents the relative orientation between the magnetic field and the the average position of the vectors
def construct_saupe_matrix(tensor, debug = False):
    """
    Constructs the Saupe matrix from a given tensor.
    """
    Sxx, Syy, Szz, Sxy, Sxz, Syz = tensor
    SaupeMatrixTmp = np.array([
        [Sxx, Sxy, Sxz],
        [Sxy, Syy, Syz],
        [Sxz, Syz, Szz]
    ])
    if not np.isclose(Sxx + Syy + Szz, 0.0, atol=1e-04):
        raise RuntimeError(f'Sxx + Syy + Szz = {Sxx + Syy + Szz}; should be zero')
    #if debug: print("Saupe Matrix:\n", SaupeMatrixTmp)
    return SaupeMatrixTmp


def axAndRhComponents(saupeMatrix):
    assert saupeMatrix[0, 1] == saupeMatrix[1, 0]
    assert saupeMatrix[0, 2] == saupeMatrix[2, 0]
    assert saupeMatrix[1, 2] == saupeMatrix[2, 1]
    Sxx, Syy, Szz = np.linalg.eigvalsh(saupeMatrix)
    # Note that the order is important for axiality and rhombicity, but
    # elsewhere in the code Sxx, Syy, Szz aren't necessarily ordered
    assert Sxx <= Syy
    assert Syy <= Szz
    ax = (Szz - (Sxx + Syy) / 2.0) / 3.0
    rh = (Sxx - Syy) / 3.0
    return ax, rh


def makeSaupes():
    tensorMap = {}

    # Saupe tensor sets for Z and C domains
    #saupe_tensor_sets_Z = [
    tensorMap['Z'] = [
        [4.28001, -7.67796, 3.39795, 6.55967, 0.40121, 1.06127],  # Set 1
        [3.52548, -8.35964, 4.83416, 13.0823, -0.85906, -0.72809],  # Set 2
        [7.34644, -6.49645, -0.84999, 5.89684, 2.51085, 1.58517],  # Set 3
        [3.97532, -7.91062, 3.93530, 13.3048, -0.08414, -1.39206],   # Set 4
        [-8.6551, 14.9983, -6.34323, -20.5640, -0.41839, 0.36701], # OLC 1
        #[8.6551, -14.9983, 6.34323, 20.5640, 0.41839, -0.36701], # OLC 1
        [5.04792, -2.43413, -2.61378, -1.55826, 2.56783, 2.28739], # OLC 2
        [0] * 6,
        [0] * 6
    ]

    #saupe_tensor_sets_C = [
    tensorMap['C'] = [
        [0.358836, -0.89213, 0.533293, -0.259814, -0.213286, -0.379667],  # Set 1
        [0.058455, -0.71456, 0.656105, -0.430284, 0.0690889, -0.052648],  # Set 2
        [0.442151, -0.92183, 0.479674, -0.233014, 0.0074228, -0.530056],  # Set 3
        [0.331236, -0.80239, 0.471152, -0.544461, 0.0135016, -0.280945],  # Set 4
        [-0.521156, 1.57063, -1.04947, 0.771357, 0.0267095, 0.524178], # OLC 1
        [0.357427, -0.54729, 0.18986, 0.036935, -0.0785261, -0.454419], # OLC 2
        [0] * 6,
        [0] * 6
    ]

    # Yang's BnBmain.m code involves calculating effective C-domain Saupe
    # tensors to go along with each experimental Saupe tensor, in this case one
    # for OLC1 and one for OLC2. This is done for each candidate Bingham
    # solution.

    tensorMap['sol1'] = [
        [0] * 6,
        [0] * 6,
        [0] * 6,
        [0] * 6,
        [-0.5248, 1.5649, -1.0401, 0.7462, 0.1530, 0.4778], # OLC 1
        [0.3453, -0.5593, 0.2140, 0.0080, 0.0039, -0.4714], # OLC 2
        [0] * 6,
        [0] * 6
    ]

    tensorMap['sol2'] = [
        [0] * 6,
        [0] * 6,
        [0] * 6,
        [0] * 6,
        [-0.5235, 1.5737, -1.0502, 0.7696, 0.0428, 0.5190], # OLC 1
        [0.3636, -0.5543, 0.1907, -0.0169, 0.0343, -0.4478], # OLC 2
        [0] * 6,
        [0] * 6
    ]

    def tensorListToMatrixList(tensorList):
        return [construct_saupe_matrix(tensor, globalDebug) for tensor in tensorList]

    return {name: tensorListToMatrixList(tensorList) for name, tensorList in tensorMap.items()}


# deprecated
saupes = makeSaupes()


# load experimental data from csv files
# untested, for use later wtih Gly6 data
def load_experimental_data(file_paths):
    observed_residues = {'Z':[], 'C':[]}
    experimental_data = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: file not found - {file_path}")
            continue
        base_name = os.path.basename(file_path)
        alignment_condition = base_name.replace("RDCsinFormat_", "").replace("_mathematica.csv", "").replace("_", "/")

        # Extract only "Dy" or "Tb"
        alignment_type = alignment_condition.split("/")[-1]  # This keeps only "Dy" or "Tb"

        if "NHis" in alignment_condition:
            condition = f"NHis-ZLBT_C/{alignment_type}"
        else:
            condition = f"ZLBT_C/{alignment_type}"

        #Read csv
        df = pd.read_csv(file_path)

        required_columns = ['DomainName', 'ResiNo', 'NH RDC']
        if not all (col in df.columns for col in required_columns):
            print(f"Warning: missing required columns in {file_path}")
            continue

        #df['ResiNo'] = df['ResiNo'].astype(int)
        #df['NH RDC'] = pd.to_numeric(df['NH RDC'], errors = 'coerce')

        df = df.dropna(subset = ['NH RDC'])
        #+1 is due to the experimental sequence is 1 off from the reference sequence.
        experimental_data[condition] = {int(row['ResiNo']): row['NH RDC'] for _, row in df.iterrows()}

        #observed_residues.setdefault(condition, {'Z':[], 'C':[]})
        Z_OFFSET = 1  # Z domain needs +1
        C_OFFSET = 1  # C domain needs an appropriate offset

        # Define linker residues to exclude
        # Other than wt linker: KADNKF, I also added the residue next "N", which is # 76 to match with the old code
        # we also dont consider residue 128 given it is not really on the c domain, and the JMB paper did not use residue 82
        #residue #82, 113, 117 was only detected by NHis-Tb
        LINKER_RESIDUES = {70, 71, 72, 73, 74, 75, 76, 82, 128}

        # Process observed residues for Z and C domains
        grouped_residues = df.groupby('DomainName')['ResiNo'].apply(list).to_dict()
        for domain in ['Z', 'C']:
            if domain in grouped_residues:
                adjusted_residues = [
                    (resi + Z_OFFSET if domain == 'Z' else resi + C_OFFSET)
                    for resi in grouped_residues[domain] if resi not in LINKER_RESIDUES
                ]
                observed_residues[domain].append(adjusted_residues)
                
    print("Loaded Experimental Data:")
    print("\n=== Experimental Data ===")
    for condition, residues in experimental_data.items():
        print(f" Condition: {condition}")
        for resi_no, rdc_value in residues.items():
            print(f"  - Residue {resi_no}: {rdc_value}")
        print("-" * 40)

    print("\n=== Observed Residues ===")
    for domain, conditions in observed_residues.items():
        print(f" Domain: {domain}")
        for i, residues in enumerate(conditions):
            print(f"  - Condition {i+1}: {residues}")
        print("-" * 40)

    return experimental_data, observed_residues


#base_path = os.path.expanduser("~/Desktop/CDIO Project/Yang Qi's stuff/Qi's Data/")
#file_paths = [
#    os.path.join(base_path, "RDCsinFormat_ZLBT_Dy_mathematica.csv"),
#    os.path.join(base_path, "RDCsinFormat_ZLBT_Tb_mathematica.csv"),
#    os.path.join(base_path, "RDCsinFormat_ZLBTNHis_Dy_mathematica.csv"),
#    os.path.join(base_path, "RDCsinFormat_ZLBTNHis_Tb_mathematica.csv")
#]
#experimental_data, observed_residues = load_experimental_data(file_paths)


experimental_error = 0.2 #Hz


experimental_data = {
    "ZLBT_C/Dy": {
        19: 0.8441, 21: 6.7231, 22: -6.5455, 64: 7.1962, 66: 5.9593, 67: -0.0028,
        68: 6.4673, 70: -2.9005, 71: 1.1789, 72: -0.3495, 75: 0.276, 76: -0.3814,
        79: -0.3866, 80: -0.0624, 81: 0.6801, 83: -0.2527, 84: -0.3737, 88: 0.5844,
        89: 0.7404, 91: 0.15, 92: 0.1671, 93: 1.0032, 94: -0.4393, 95: -0.2525,
        97: 0.1525, 98: -0.0453, 99: -0.1374, 100: 0.3577, 103: -0.2423, 104: 0.2746,
        105: -0.3379, 106: -0.1411, 107: 0.0046, 109: -0.2597, 110: -0.5433, 111: -0.7837,
        112: -0.1753, 114: -0.222, 116: 0.5046, 118: -0.4182, 119: 0.284, 120: 0.6711,
        121: -0.6838, 122: -0.4073, 124: -0.0167, 125: -0.2001, 126: 0.4959, 128: -0.2212
    },

    "ZLBT_C/Tb": {
        19: -3.8494, 21: 15.7635, 22: -8.1829, 64: 11.6468, 66: 7.8901, 67: 3.6104,
        68: 12.4897, 70: 1.6852, 71: 3.5956, 72: -0.4318, 75: -0.0047, 76: -0.0289,
        79: -0.7805, 80: -0.3474, 81: 0.1908, 83: -0.4265, 84: -0.4936, 88: 0.3451,
        89: 0.6706, 91: -0.3573, 92: -0.0858, 93: 0.271, 97: 0.0403, 98: -0.2663,
        99: -0.2895, 100: 0.2872, 103: -0.096, 104: 0.1134, 105: -0.006, 106: -0.5145,
        107: 0.1943, 109: 0.2429, 110: -0.5221, 111: -0.5723, 112: -0.1676, 114: -0.4916,
        116: 0.4674, 118: -0.3683, 119: 0.2031, 120: 0.5552, 121: -0.7762, 122: -0.3677,
        124: 0.2508, 125: -0.4516, 126: -0.0515, 128: -0.0546
    },
    "NHis-ZLBT_C/Dy": {
        19: 5.4041, 21: 5.0571, 22: -7.4286, 64: 5.0223, 66: 6.1579, 67: -1.3906,
        68: 6.9156, 70: -3.2344, 71: 1.0061, 72: -0.0417, 75: 0.5732, 76: -0.0298,
        79: -0.5197, 80: -0.4853, 81: 0.331, 83: -0.4867, 84: -0.8459, 88: 0.4035,
        89: 0.5164, 91: -0.0308, 92: -0.2175, 93: 0.7121, 97: 0.8281, 98: -0.3651,
        99: -0.5477, 100: 0.1372, 103: -0.4687, 104: 0.0364, 105: -0.1742, 107: -0.0423,
        109: -0.3406, 110: -0.2861, 111: -0.798, 112: -0.1392, 114: -0.478, 116: 0.2599,
        117: -0.8307, 118: -0.7114, 119: 0.1514, 120: 0.1906, 121: -0.854, 122: -0.547,
        124: -0.0262, 125: -0.5399, 126: 0.7025, 128: -0.088
    },
    "NHis-ZLBT_C/Tb": {
        19: -4.3421, 21: 16.788, 22: -8.2766, 64: 10.1252, 66: 7.9158, 67: 2.6963,
        68: 13.2276, 70: 1.7952, 71: 3.5817, 75: 0.0406, 76: -0.1459, 79: -0.4905,
        80: -0.4673, 81: -0.1872, 82: -1.0192, 83: -0.7412, 84: -1.0094, 88: 0.0268,
        89: 0.6137, 91: -0.5529, 92: -0.1739, 93: 0.6958, 97: 0.6031, 98: -0.5435,
        99: -0.7269, 100: 0.1554, 103: -0.2758, 104: 0.0286, 105: -0.4358, 109: 0.18,
        110: -0.5094, 111: -0.9186, 112: -0.0411, 113: -1.0105, 114: -0.4611,
        116: -0.0161, 117: -1.0948, 118: -0.6794, 119: -0.0761, 120: 0.1623,
        121: -0.8329, 122: -0.5721, 124: -0.0218, 125: -0.5145, 126: 0.1694, 128: 0.1784
    },
    "OLC1": {
        19: -2.87486, 21: 24.1115, 22: -15.005, 64: 17.685, 66: 13.9591, 67: 3.38584, 68: 20.4631,
        79: 1.09594, 80: 0.684661, 81: -0.367249, 83: 0.972134, 84: 1.34326, 88: -0.581083, 89: -1.23526, 
        91: 0.510894, 92: 0.169576, 93: -1.2055, 97: -0.729225, 98: 0.634963, 99: 0.857193, 100: -0.450202, 
        103: 0.475559, 104: -0.201538, 105: 0.455177, 109: -0.0469204, 110: 0.929685, 111: 1.47275, 112: 0.239776, 
        114: 0.826058, 116: -0.551395, 118: 1.03616, 119: -0.234418, 120: -0.753061, 121: 1.52632, 122: 0.910756, 
        124: -0.124739, 125: 0.846961, 126: -0.493731
    },
    "OLC2": {
        19: 7.40972, 21: -4.15772, 22: -2.88064, 64: -0.722305, 66: 1.84541, 67: -3.20708, 68: -0.488222,
        79: -0.136222, 80: -0.175986, 81: 0.452517, 83: -0.122002, 84: -0.36018, 88: 0.373835, 89: 0.230597, 
        91: 0.296371, 92: -0.0609308, 93: 0.581884, 97: 0.55902, 98: -0.0749571, 99: -0.194435, 100: 0.0690292, 
        103: -0.351858, 104: 0.0572807, 105: -0.111397, 109: -0.495454, 110: -0.062727, 111: -0.42825, 
        112: -0.096008, 114: -0.164617, 116: 0.200269, 118: -0.395724, 119: 0.157061, 120: 0.104163, 121: -0.403624, 
        122: -0.283803, 124: -0.106965, 125: -0.209908, 126: 0.700569
    },
    "OLC3": {},
    "OLC4": {}
}


#Experimental Datas
observed_residues = {
    'C': 
    # Goes with ZLBT-C
    [
        [x + 71 for x in [9, 10, 11, 13, 14, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 46, 48, 49, 50, 51, 52, 54, 55, 56]],  # Set 1 Dy
        [x + 71 for x in [9, 10, 11, 13, 14, 18, 19, 21, 22, 23, 27, 28, 29, 30, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 46, 48, 49, 50, 51, 52, 54, 55, 56]],  # Set 2 Tb
        [x + 71 for x in [9, 10, 11, 13, 14, 18, 19, 21, 22, 23, 27, 28, 29, 30, 33, 34, 35, 37, 39, 40, 41, 42, 44, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56]],  # Set 3 Nhis Dy
        [x + 71 for x in [9, 10, 11, 13, 14, 18, 19, 21, 22, 23, 27, 28, 29, 30, 33, 34, 35, 39, 40, 41, 42, 44, 46, 48, 49, 50, 51, 52, 54, 55, 56]],  # Set 4  NHis Tb
        # The following two observed residue lists represent the residues that the above four lists have in common, for OLC calculation
        [x + 1 for x in [79, 80, 81, 83, 84, 88, 89, 91, 92, 93, 97, 98, 99, 100, 103, 104, 105, 109, 110, 111, 112, 114, 116, 118, 119, 120, 121, 122, 124, 125, 126]], # OLC1
        [x + 1 for x in [79, 80, 81, 83, 84, 88, 89, 91, 92, 93, 97, 98, 99, 100, 103, 104, 105, 109, 110, 111, 112, 114, 116, 118, 119, 120, 121, 122, 124, 125, 126]], #OLC2
        #[x + 1 for x in [79, 80, 81, 83, 84, 88, 89, 91, 92, 93, 97, 98, 99, 100, 103, 104, 105, 109, 110, 111, 112, 114, 116, 118, 119, 120, 121, 122, 124, 125, 126]], 
        #[x + 1 for x in [79, 80, 81, 83, 84, 88, 89, 91, 92, 93, 97, 98, 99, 100, 103, 104, 105, 109, 110, 111, 112, 114, 116, 118, 119, 120, 121, 122, 124, 125, 126]]

    ],
    'Z':
    # Goes with 2lr2, which is 1 sequence more than the experimental sequences at the beginning.
    [
        [20, 22, 23, 65, 67, 68, 69],
        [20, 22, 23, 65, 67, 68, 69],
        [20, 22, 23, 65, 67, 68, 69],
        [20, 22, 23, 65, 67, 68, 69],
        [20, 22, 23, 65, 67, 68, 69],
        [20, 22, 23, 65, 67, 68, 69],
        #[20, 22, 23, 65, 67, 68, 69],
        #[20, 22, 23, 65, 67, 68, 69]
    ]
}


def get_common_indices(domain, icond, common_res):
    return [i for i, resi in enumerate(observed_residues[domain][icond]) if resi in common_res]


#Common residues
def get_common_residues(domain):
    observed_residues = cdiotools.rdcs.observed_residues[domain]
    common_residues = set(observed_residues[0])
    for cond_residues in observed_residues[1:]:
        common_residues.intersection_update(cond_residues)
    #print(f"common residues for {domain}: {sorted(common_residues)}")
    return sorted(common_residues)


def get_common_rdcs(domain, rdcs):
    common_res = get_common_residues(domain)
    common_rdcs = []
    for icond in range(4):  # Only first 4 conditions
        common_indices = get_common_indices(domain, icond, common_res)
        common_rdcs.append(np.array([rdcs.rdcs[domain][icond][i] for i in common_indices]))
    return common_rdcs  # Returns a list now


def rmsdfunc(v1, v2):
    if len(v1) != len(v2):
        print('v1: ', v1)
        print('v2: ', v2)
        raise RuntimeError("Mismatch in vector lengths in rmsdfunc")
    if len(v1) == 0:
        return 0
    return np.sqrt(((v1 - v2)**2).mean())


def rsquaredfunc(y, x):
    assert len(y) == len(x)
    
    print("y", y)
    print("x", x)
    if len(y) == 0:
        return 0
    y_mean = np.mean(y)
    print("y", y)
    print("y_mean:", y_mean)
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot ==0:
        print (f"No variance in data. Returning R^2 = 0. y: {y}")
        return 0
    r2 = 1 - (ss_res / ss_tot)
    return r2


# As in all of these, y are RDCs to be tested (e.g., backcalculated); x are the
# correct RDCs (e.g., experimental)
def linregfunc(y, x):
    return scipy.stats.linregress(x, y)


# As in all of these, y are RDCs to be tested (e.g., backcalculated); x are the
# correct RDCs (e.g., experimental)
def qfunc(y, x):
    assert len(y) == len(x)
    if len(y) == 0:
        return 0
    ss_res = np.sum((y - x) ** 2)
    ss_x = np.sum(x ** 2)
    return float('NaN') if ss_x == 0 else np.sqrt(ss_res / ss_x) # Correct
    #return float('NaN') if ss_x == 0 else np.sqrt(ss_res / len(y)) / np.std(x, ddof=1) # what Yang actually did


# As in all of these, y are RDCs to be tested (e.g., backcalculated); x are the
# correct RDCs (e.g., experimental)
def rdipfunc(y, x, saupe):
    assert len(y) == len(x)
    ss_res = np.sum((y - x) ** 2)
    ax, rh = axAndRhComponents(saupe)
    rhombicity = rh / ax
    return np.sqrt((5 * ss_res / len(y)) / (2 * ax**2 * (4 + 3 * rhombicity**2)))


# As in all of these, y are RDCs to be tested (e.g., backcalculated); x are the
# correct RDCs (e.g., experimental)
def rdipfuncList(yList, xList, saupeList):
    def forOne(y, x, saupe):
        assert len(y) == len(x)
        ss_res = np.sum((y - x) ** 2)
        ax, rh = axAndRhComponents(saupe)
        rhombicity = rh / ax
        return (5 * ss_res / len(y)) / (2 * ax**2 * (4 + 3 * rhombicity**2))

    return np.sqrt(np.sum([forOne(y, x, s) for y, x, s in zip(yList, xList, saupeList)]))


def rdipfuncSimple(y, x):
    ss_res = np.sum((y - x) ** 2)
    ss_x = np.sum(x ** 2)
    assert ss_x > 0
    return np.sqrt(ss_res / (2 * ss_x))


# This is not a standard quantity from the literature; this is just Q with
# back-calculated RDCs in the denominator instead of experimental
def qfuncAlt(y, x):
    assert len(y) == len(x)
    if len(y) == 0:
        return 0
    ss_res = np.sum((y - x) ** 2)
    ss_y = np.sum(y ** 2)
    return np.sqrt(ss_res / ss_y)


class NoHydrogens(Exception):
    pass


class NoC(Exception):
    pass


class NoZ(Exception):
    pass


#Determine RDC calculation, population weighing, and combination class
class Rdcs:
    def __init__(self, filename = None, debug = False):
        self.rdcs = {}
        self.rdcs['Z'] = []
        self.rdcs['C'] = []
        if filename and filename.exists():
            with open(filename, 'r') as rdcfile:
                rdcscsv = csv.reader(rdcfile)
                for domain in ['Z', 'C']:
                    for condname in alignment_conditions:
                        row = next(rdcscsv)
                        assert row[0] == domain
                        assert row[1] == condname
                        self.rdcs[domain].append(np.array([float(x) for x in row[2:]]))
        elif filename:
            raise RuntimeError(f'Filename {filename} not found to read CSV values from')
        else:
            for domain in ['Z', 'C']:
                for observed in observed_residues[domain]:
                    self.rdcs[domain].append(np.zeros(len(observed)))

    def __str__(self):
        rvlist = []
        for domain, rdcsByCond in self.rdcs.items():
            rvlist.append(domain)
            for rdcsPerCond in rdcsByCond:
                rvlist.append(str(rdcsPerCond))
        return '\n'.join(rvlist)

    def __rmul__(self, other):
        result = Rdcs()
        for domain, rdcVecs in self.rdcs.items():
            for i, _ in enumerate(rdcVecs):
                result.rdcs[domain][i] = other * self.rdcs[domain][i]
        return result

    def __add__(self, other):
        result = Rdcs()
        for domain, rdcVecs in self.rdcs.items():
            for i, _ in enumerate(rdcVecs):
                result.rdcs[domain][i] = other.rdcs[domain][i] + self.rdcs[domain][i]
        return result

    def __eq__(self, other):
        for domain, rdcVecs in self.rdcs.items():
            for i, _ in enumerate(rdcVecs):
                if not (self.rdcs[domain][i] == other.rdcs[domain][i]).all():
                    return False
        return True

    def summaryStatsOne(self, func, debug = False):
        rv = {}
        rv['Z'] = []
        rv['C'] = []
        for domain, rdcVecs in self.rdcs.items():
            for icond, _ in enumerate(rdcVecs):
                rv[domain].append(func(self.rdcs[domain][icond]))
        return rv

    def range(self, debug = False):
        return self.summaryStatsOne(lambda v: v.max() - v.min(), debug)

    def summaryStats(self, other, func, debug = False):
        rv = {}
        rv['Z'] = []
        rv['C'] = []
        def statOne(domain, icond):
            return func(self.rdcs[domain][icond], other.rdcs[domain][icond])

        for domain, rdcVecs in self.rdcs.items():
            for icond, _ in enumerate(rdcVecs):
                rv[domain].append(statOne(domain, icond))

        return rv
        
    #Here, we are defining a rmsd class method, with a return of the rmsd dict keyed on the domains
    def rmsd(self, other, debug = False):
        return self.summaryStats(other, rmsdfunc, debug)
    
    def rms(self, debug = False):
        zeros = Rdcs()
        return self.rmsd(zeros, debug)

    def rsquared(self, other, debug = False):
        return self.summaryStats(other, rsquaredfunc, debug)

    def linreg(self, other, debug = False):
        return self.summaryStats(other, linregfunc, debug)
    
    def q(self, other, debug = False):
        return self.summaryStats(other, qfunc, debug)
    
    def qAlt(self, other, debug = False):
        return self.summaryStats(other, qfuncAlt, debug)

    def summaryconcat(self, other, conditionList, func, debug = False):
        rv = {}
        icondList = findicondList(conditionList)
        def concatLists(rdcs, domain):
            concat = np.empty(0)
            for icond in icondList:
                concat = np.concatenate((concat, rdcs.rdcs[domain][icond]))
            return concat
        for domain in self.rdcs:
            a = concatLists(self, domain)
            b = concatLists(other, domain)
            rv[domain] = func(a, b)
        return rv
        
    #putting OLC RMSDs (OLC1, OLC2 together in the same list)
    def concatRmsd(self, other, conditionList, debug = False):
        return self.summaryconcat(other, conditionList, rmsdfunc, debug)
    
    def concatLinreg(self, other, conditionList, debug = False):
        return self.summaryconcat(other, conditionList, linregfunc, debug)

    def concatQ(self, other, conditionList, debug = False):
        return self.summaryconcat(other, conditionList, qfunc, debug)

    def concatrsquared(self, other, conditionList, debug = False):
        return self.summaryconcat(other, conditionList, rsquaredfunc, debug)

    def concatRange(self, other, conditionList, debug = False):
        return self.summaryconcat(other, conditionList, lambda v: v.max() - v.min(), debug)

    def concatRdipSimple(self, other, conditionList, debug = False):
        return self.summaryconcat(other, conditionList, rdipfuncSimple, debug)

    def concatRdip(self, other, conditionList, saupeList, debug = False):
        icondList = findicondList(conditionList)
        rv = {}
        for domain in self.rdcs:
            yList = [self.rdcs[domain][icond] for icond in icondList]
            xList = [other.rdcs[domain][icond] for icond in icondList]
            rv[domain] = rdipfuncList(yList, xList, saupeList)
        return rv

    def writeToFile(self, filename):
        with open(filename, 'w') as rdcsfile:
            rdcscsv = csv.writer(rdcsfile)
            for domain, rdcsByCond in self.rdcs.items():
                for icond, cond in enumerate(alignment_conditions):
                    row = [domain, cond] + list(rdcsByCond[icond])
                    rdcscsv.writerow(row)


def getExpRdcs():
    tmpExpRdcs = Rdcs()
    for domain in ['Z', 'C']:
        #Edward chenged alignment_conditions from [0:6] to [0:4] for the actual experimental data
        for icond, condname in enumerate(alignment_conditions[0:6]):
            # The -1 is just a hard-coded offset to adjust for various numbering conventions.
            tmpExpRdcs.rdcs[domain][icond] = np.array([
                experimental_data[condname][i-1]
                for i in observed_residues[domain][icond]
            ])
    return tmpExpRdcs


def add_gaussian_noise_to_rdcs(rdcs_obj, stddev, rng):
    noisy_rdcs = Rdcs()
    for domain in ['Z', 'C']:
        for i in range(len(rdcs_obj.rdcs[domain])):
            original = rdcs_obj.rdcs[domain][i]
            # Add zero-centered Gaussian noise to each RDC value
            noise = rng.normal(loc=0.0, scale=stddev, size=original.shape)
            noisy_rdcs.rdcs[domain][i] = original + noise
    return noisy_rdcs


# deprecated
expRdcs = getExpRdcs()


#Define RDC Offsets, 
def findRdcOffsets(offset, domain, icond):
    try:
        selected_residues = observed_residues[domain][icond]
        rdcOffsets = np.zeros(len(selected_residues), dtype=int)
    except IndexError as e:
        print('domain:', domain)
        print('icond:', icond)
        raise e
    #For Z domain we are going to determine off sets for each of 7 RDC readouts
    if domain == 'Z':
        if offset == [0, 0, 0, 0]:
            # Predictor used FASTA Residue Number
            pass
        elif offset == [-1, -1, -1, -1]:
            # Predictor started numbering at valine
            rdcOffsets -= 1
        elif offset == [12, 12, 12, 12]:
            # Predictor started numbering at Histidine tag
            rdcOffsets += 12
        elif offset == [-1, 4, 4, 4]:
            # Predictor used alternative numbering (no deletions in LBT)
            rdcOffsets += [-1, -1, -1, 4, 4, 4, 4]
        elif offset[0:2] == [-1, -13]:
            # Used for testing only (with 1Q2N)
            rdcOffsets += [-1, -1, -1, -13, -13, -13, -13]
        elif offset[0:2] == [None, None]:
            raise NoZ("Offset appears to be for C domain, not Z domain")
        else:
            raise RuntimeError("Unaccounted-for offsets")
    
    #For C domain, all the off sets are uniformed.
    elif domain == 'C':
        if offset == [0, 0, 0, 0]:
            # Predictor used FASTA Residue Number
            pass
        elif offset == [-1, -1, -1, -1]:
            # Predictor started numbering at valine
            rdcOffsets -= 1
        elif offset == [12, 12, 12, 12]:
            # Predictor started numbering at Histidine tag
            rdcOffsets += 12
        elif offset == [-1, 4, 4, 4]:
            # Predictor used alternative numbering (no deletions in LBT)
            rdcOffsets += 4
        elif offset[2:4] == [-71, -71]:
            rdcOffsets += -71
        elif offset[2:4] == [None, None]:
            raise NoC("1Q2N has no C domain")
        else:
            raise RuntimeError("Unaccounted-for offsets")
    else:
        raise RuntimeError("Bad domain identifier")
    return rdcOffsets


def findWantedResidues(icond, offset, domain, debug = False):
    selected_residues = observed_residues[domain][icond]
    if domain == "C":
        #skip residue 128 because it's flexable and not really part of the C domain
        selected_residues = [res for res in selected_residues if res != 128]
    rdcOffsets = findRdcOffsets(offset, domain, icond)

    result = np.array(selected_residues, dtype=int) + rdcOffsets

    return result


def findVectors(chain, icond, offset, domain, debug = False):
    vectors = []

    residuesWithRDCsIndices = findWantedResidues(icond, offset, domain, debug)

    for i, resnum in enumerate(residuesWithRDCsIndices):
        res = chain[int(resnum)]
        residue_id = res.id[1]  # Residue sequence number

        # The -1 accounts for the zlbtc array being zero-indexed
        assert cdiotools.pdbutils.oneLetter[res.get_resname()] == zlbtc[observed_residues[domain][icond][i]-1]
        if domain =='Z':
            if i == 0:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'L'
            if i == 1:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'N'
            if i == 2:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'L'
            if i == 3:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'N'
            if i == 4:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'A'
            if i == 5:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'Q'
            if i == 6:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'A'
        if domain =='C':
            if i == 0:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'Q'
            if i == 1:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'Q'
            if i == 2:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'N'
            if i == 3:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'F'
            if i == 4:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'Y'
            if i == 5:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'H'
            if i == 6:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'L'
            if i == 7:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'N'
            if i == 8:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'L'
            if i == 9:
                assert cdiotools.pdbutils.oneLetter[res.get_resname()] == 'T'
        if 'N' in res and ('H' in res or 'HN' in res):
            N_atom = res['N']
            if N_atom.is_disordered():
                N_atom.disordered_select('A')
            H_atom = res['H'] if 'H' in res else res['HN']
            if H_atom.is_disordered():
                H_atom.disordered_select('A')
            vector = H_atom.get_coord() - N_atom.get_coord()
            norm = np.linalg.norm(vector)
            if norm != 0:
                normalized_vector = vector / norm
                vectors.append(normalized_vector)

                if debug:
                    print(f"Normalized & Rotated vector for residue {residue_id}: {normalized_vector}")
            else:
                raise RuntimeError("Error: zero-length NH vector found")
        else:
            print(f'Error: residue {resnum} did not contain both N and H atoms')
            raise NoHydrogens
    return vectors


def findSaupes(chain, offset, rdcs, domain):
    saupeDict = {}
    for icond, condname in enumerate(alignment_conditions[0:4]):
        allVectors = findVectors(chain, icond, offset, domain)
        assert len(allVectors) == len(rdcs.rdcs[domain][icond])

        def TRow(v):
            x, y, z = v
            return y**2 - x**2, z**2 - x**2, 2*x*y, 2*x*z, 2*y*z
        T = np.row_stack([TRow(v) for v in allVectors])
        Dred = (rdcs.rdcs[domain][icond] / (KNH / 2.0)).T 

        S = scaling_factor * np.linalg.lstsq(T, Dred, rcond=None)[0] # Calls dgelsd in LAPACK, which uses SVD

        Sxx = -S[0] - S[1]  # Sxx = -(Syy + Szz) due to traceless property
        Syy = S[0]
        Szz = S[1]
        Sxy = S[2]
        Sxz = S[3]
        Syz = S[4]

        # Construct and append the full Saupe matrix to the list
        saupe_matrix = [Sxx, Syy, Szz, Sxy, Sxz, Syz]
        saupeDict[condname] = saupe_matrix

    return saupeDict


# Function to compute dipolar coupling
def backcalculateRdcsPerDomain(chain, rotation, offset, domain, saupelist, icond, debug = False):
    vectors = findVectors(chain, icond, offset, domain)
    vectorsArray = np.column_stack(vectors)
    vectorsRotated = rotation @ vectorsArray
    vectorsList = [x for x in vectorsRotated.T]
    VTSV = []
    
    for vec in vectorsList:
        #coupling = vec @ saupematrixlist[icond] @ vec / scaling_factor
        coupling = vec @ saupelist[icond] @ vec / scaling_factor
        VTSV.append(float(coupling))
        if debug:
            print("dipolar couplings:\n", VTSV)
    D = np.array([float(KNH * vtsv / 2.) for vtsv in VTSV])
    if debug:
        print("D:\n", D)
    return D


def backcalcBoth(chainByDomain, offset, saupesByDomain, rotationMatrix = np.identity(3), debug = False):
    rdcs = Rdcs(debug=debug)
    for domain, saupelist in saupesByDomain.items():
        chain = chainByDomain[domain]
        for icond, _ in enumerate(rdcs.rdcs[domain]):
            try:
                rdcs.rdcs[domain][icond] = backcalculateRdcsPerDomain(chain, rotationMatrix, offset, domain, saupelist, icond, debug)
            except IndexError as e:
                print(rdcs)
                raise e
    return rdcs


# Deprecated in favor of backcalcBoth
def backcalculate_rdcs(predictor, offset, rotationMatrix, domains, saupelist, debug = False):
    rdcs = Rdcs(debug=debug)
    for domain in domains:
        for icond, _ in enumerate(rdcs.rdcs[domain]):
            try:
                rdcs.rdcs[domain][icond] = backcalculateRdcsPerDomain(predictor, rotationMatrix, offset, domain, saupelist, icond, debug)
            except IndexError as e:
                print(rdcs)
                raise e
    return rdcs
