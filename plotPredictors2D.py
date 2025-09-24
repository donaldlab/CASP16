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
import cdiotools.plotting
import cdiotools.ensembles
import csv

# Example usage:

#python plotPredictors2D.py --xfile saxsResults.csv --yfile chebyshev.csv --output isoVsChiSq.png --xlog --xcol "chi_square" --ycol "Anisotropicity" --xpredcol "submission ID" --ypredcol "Ensemble" --xlabel "SAXS Chi-squared for I(q) (log scale)" --ylabel "Anisotropicity (kcal/mol)" --exclude --xhi 100 --categfile categories.csv --idealy 0.09 --idealy2 0.42 --idealx 1

#python plotPredictors2D.py --xfile saxsResults.csv --yfile chebyshev.csv --output chebyshevVsChiSq.png --xlog --xcol "chi_square" --ycol "Chebyshev distance" --xpredcol "submission ID" --ypredcol "Ensemble" --xlabel "SAXS Chi-squared for I(q) (log scale)" --ylabel "Chebyshev distance for NMR-related quantities" --exclude --xhi 100 --categfile bestGuessCategories.csv  --idealx 1 --idealy 0

plt.rcParams["font.family"] = "serif"

if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument("--xfile", type=str, required=True)
    parser.add_argument("--yfile", type=str, required=True)
    parser.add_argument("--categfile", type=str, default=None, required=False)
    parser.add_argument("--xcol", type=str, required=True)
    parser.add_argument("--ycol", type=str, required=True)
    parser.add_argument("--xpredcol", type=str, required=True)
    parser.add_argument("--ypredcol", type=str, required=True)
    parser.add_argument("--xlabel", type=str, required=True)
    parser.add_argument("--ylabel", type=str, required=True)
    parser.add_argument("--title", type=str, required=False)
    parser.add_argument("--idealx", type=float, default=None, required=False)
    parser.add_argument("--idealx2", type=float, default=None, required=False)
    parser.add_argument("--idealy", type=float, default=None, required=False)
    parser.add_argument("--idealy2", type=float, default=None, required=False)
    parser.add_argument("--xlog", default=False, action='store_true')
    parser.add_argument("--xhi", type=float, default=None, required=False)
    parser.add_argument("--exclude", default=False, action='store_true')
    parser.add_argument("--legend", default=False, action='store_true')
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    # Read the CSV file
    def readfile(fn, colNameData, colNameLabels):
        data = {}
        labels = []
        with open(fn, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = row[colNameLabels]
                data[label] = float(row[colNameData])
                labels.append(label)
        return data, labels

    xdata, xlabels = readfile(args.xfile, args.xcol, args.xpredcol)
    ydata, ylabels = readfile(args.yfile, args.ycol, args.ypredcol)
    print(xdata)
    print(ydata)

    xdata, xlabels = cdiotools.ensembles.fixLabels(xdata, xlabels)
    ydata, ylabels = cdiotools.ensembles.fixLabels(ydata, ylabels)
    excludesRMSD = ['088', '139', '002', '084', '003', '004', '005', '100']
    excludesMixtures = ['264', '148']
    excludes = excludesRMSD + excludesMixtures

    def findCommon(l1, l2):
        labels = []
        for s in l1:
            if s in l2 and not (s in excludes and args.exclude):
                labels.append(s)
        assert len(labels) == 35 - (len(excludes) if args.exclude else 0)
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
    # Generate the plot
    cdiotools.plotting.plot2d(
            args.output, 
            xdata, 
            ydata, 
            labels, 
            args.xlabel, 
            args.ylabel, 
            args.title, 
            idealx=args.idealx, 
            idealx2=args.idealx2, 
            idealy=args.idealy, 
            idealy2=args.idealy2, 
            colors=predColorList,
            xlim = [0, args.xhi] if args.xhi else None,
            xlog = args.xlog,
            legend = args.legend,
            colorMapInv = colorMapInv
            )
