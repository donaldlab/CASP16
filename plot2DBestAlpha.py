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

import matplotlib.pyplot as plt
from adjustText import adjust_text
from argparse import ArgumentParser
import csv
import math
import re

import cdiotools.plotting
import cdiotools.cdios
import cdiotools.ensembles


plt.rcParams["font.family"] = "serif"

if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument("--xfile", type=str, default="T1200_cleaned_rot_freeEnergyDiffs.csv")
    parser.add_argument("--yfile", type=str, default="angDistByAlpha_-32.csv")
    parser.add_argument("--alphasfile", type=str, default="alphas.csv")
    parser.add_argument("--categfile", type=str, default="bestGuessCategories.csv")
    parser.add_argument("--xpredcol", type=str, default="ensemble")
    parser.add_argument("--ypredcol", type=str, default="ensemble")
    parser.add_argument("--xlabel", type=str, default="Mode-matched free energy difference between prediction and correct solution (kcal/mol)")
    parser.add_argument("--ylabel", type=str, default="Angular distance to correct mode (degrees)")
    parser.add_argument("--title", type=str, required=False)
    parser.add_argument("--legend", default=False, action='store_true')
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    excludesRMSD = ['088', '139', '002', '084', '003', '004', '005', '100']
    excludesMixtures = ['264', '148']
    excludes = excludesRMSD + excludesMixtures

    alphasByPred = {}
    with open(args.alphasfile, 'r', newline='') as alphafile:
        alphaReader = csv.DictReader(alphafile)
        for row in alphaReader:
            alphasByPred[row['Ensemble']] = cdiotools.cdios.findAlphaString(float(row['best alpha']), 2)

    def rowToPrednum(rowname):
        return re.fullmatch('T1200TS([0-9]{3}).cleaned', rowname).group(1)

    # Read the CSV file
    def readfile(fn, colNameFunc, colNameLabels):
        data = {}
        labels = []
        with open(fn, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = row[colNameLabels]
                predNum = rowToPrednum(label)
                if predNum in alphasByPred:
                    bestalpha = alphasByPred[rowToPrednum(label)]

                    data[label] = float(row[colNameFunc(bestalpha)])
                    labels.append(label)
                else:
                    assert predNum in excludesRMSD

        return data, labels

    xdata, xlabels = readfile(args.xfile, lambda a: 'dFE, sol ' + a + ', spread -32', args.xpredcol)
    ydataRadians, ylabels = readfile(args.yfile, lambda a: a, args.ypredcol)
    ydata = {label: math.degrees(radians) for label, radians in ydataRadians.items()}
    print(xdata)
    print(ydata)

    xdata, xlabels = cdiotools.ensembles.fixLabels(xdata, xlabels)
    ydata, ylabels = cdiotools.ensembles.fixLabels(ydata, ylabels)

    def findCommon(l1, l2):
        labels = []
        for s in l1:
            if s in l2 and s not in excludes:
                labels.append(s)
        assert len(labels) == 35 - len(excludes), len(labels)
        return labels

    labels = findCommon(xlabels, ylabels)
    print(labels)

    xdata = cdiotools.plotting.filterAndOrderData(xdata, labels)
    ydata = cdiotools.plotting.filterAndOrderData(ydata, labels)
    print('xdata:')
    print(xdata)
    print('ydata:')
    print(ydata, flush=True)

    predColorList, colorMapInv = cdiotools.plotting.findPredColors(args.categfile, labels)
    cdiotools.plotting.plot2d(
            args.output, 
            xdata, 
            ydata, 
            labels, 
            args.xlabel, 
            args.ylabel, 
            args.title, 
            idealx=0, 
            idealy=0, 
            colors= predColorList,
            xlim = [0, None],
            ylim = [0, None],
            yticks = [0, 30, 60, 90, 120, 150, 180],
            legend = args.legend,
            colorMapInv = colorMapInv,
            labelsize = 12
            )
