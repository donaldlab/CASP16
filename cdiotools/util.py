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

import math
import itertools
from joblib import Parallel, delayed


def chunkify(argiterable, nproc):
    arglist = list(argiterable)
    chunksize = math.ceil(len(arglist) / nproc)
    chunks = []
    for iproc in range(nproc):
        chunks.append(arglist[iproc * chunksize : (iproc + 1) * chunksize])
    return chunks


def batchRun(batchFunc, argiterable, nproc):
    chunks = chunkify(argiterable, nproc)
    listlist = Parallel(n_jobs = nproc)(delayed(batchFunc)(chunk) for chunk in chunks)
    return list(itertools.chain(*listlist)) if any(listlist) else None


def batchGen(batchFunc, argiterable, nproc):
    chunks = chunkify(argiterable, nproc)
    for result in Parallel(n_jobs = nproc, return_as="generator")(delayed(batchFunc)(chunk) for chunk in chunks):
        yield result
