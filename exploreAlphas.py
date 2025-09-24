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
from argparse import ArgumentParser
import csv
import re
import matplotlib.pyplot as plt
from contextlib import nullcontext

#import cdiotools.extrema
import cdiotools.cdios
import cdiotools.plotting

plt.rcParams["font.family"] = "serif"

alphas = cdiotools.cdios.findAlphas(101, 2)
alphaStrings = [cdiotools.cdios.findAlphaString(alpha, 2) for alpha in alphas]


def rowToPrednum(rowname):
    return re.fullmatch('T1200TS([0-9]{3}).cleaned', rowname).group(1)


def main():
    argparser = ArgumentParser()
    argparser.add_argument("--festats", default='T1200_cleaned_freeEnergyDiffs.csv', type=str, required=False)
    argparser.add_argument("--plotname", default='freeEnergyAlphas', type=str, required=False)
    argparser.add_argument("--histname", default='alphasHist', type=str, required=False)
    argparser.add_argument("--out", default='alphas', type=str, required=False)
    argparser.add_argument("--excludes", default=False, action='store_true')
    argparser.add_argument("--nowrite", default=False, action='store_true')
    argparser.add_argument("--debug", default=False, action='store_true')

    args = argparser.parse_args()

    exclude = ['088', '139', '002', '084', '003', '004', '005', '100']

    bestdFEs = []
    bestalphas = []
    prednums = []
    with open(args.festats, newline='') as festatsfile:
        freeEnergyStatsReader = csv.DictReader(festatsfile)
        for row in freeEnergyStatsReader:
            prednum = rowToPrednum(row['ensemble'])
            if prednum in exclude and args.excludes:
                continue
            fediff = []
            for alpha in alphaStrings:
                fediff.append(float(row[f'dFE, sol {alpha}, spread -32']))
            bestdFEs.append(np.min(fediff))
            bestalphas.append(alphas[np.argmin(fediff)])
            prednums.append(prednum)
            
    if not args.nowrite:
        cdiotools.plotting.plot2d(
                args.plotname + ('Excludes' if args.excludes else ''), 
                bestalphas, 
                bestdFEs, 
                prednums, 
                'Alpha with smallest free energy difference for ensemble', 
                'Free energy difference (kcal/mol)', 
                'Alpha values and best free energy differences', 
                colors = ['blue' if l not in exclude else 'red' for l in prednums],
                xlim = [0, 1.05], 
                ylim = [0, 1.7]
                )

    with open(args.out + '.csv', 'w') if not args.nowrite else nullcontext() as outfile:
        if not args.nowrite:
            writer = csv.writer(outfile)
            writer.writerow(['Ensemble', 'best dFE (kcal/mol)', 'best alpha'])
        for prednum, bestalpha, bestdFE in zip(prednums, bestalphas, bestdFEs):
            print(f'{prednum}: {bestdFE} kcal/mol at {bestalpha}')
            if not args.nowrite:
                writer.writerow([prednum, bestdFE, bestalpha])

    prednumsFiltered = []
    bestalphasFiltered = []
    for prednum, bestalpha in zip(prednums, bestalphas):
        if prednum not in exclude:
            prednumsFiltered.append(prednum)
            bestalphasFiltered.append(bestalpha)

    if not args.nowrite:
        print('Creating new figure', flush=True)
        plt.figure(figsize=(5,3))
        print('Plotting histograms', flush=True)
        plt.hist(bestalphas, range=(0,1), bins=10, color='red')
        plt.hist(bestalphasFiltered, range=(0,1), bins=10, color='blue')
        #plt.title('Frequency of alphas with best free energy difference for each predictor')
        plt.xlabel('α parameter value')
        plt.ylabel('Frequency (count)')
        ax = plt.gca()
        alphaticks = np.round(np.linspace(0, 1, 6), 1)
        alphalabels = [f'{alpha:.1f}' for alpha in alphaticks]
        alphalabels[0] += '\n(Solution 2)'
        alphalabels[5] += '\n(Solution 1)'
        ax.set_xticks(alphaticks)
        ax.set_xticklabels(alphalabels)
        ax.locator_params(axis='y', integer=True)
        print('Saving figure', flush=True)
        plt.savefig(args.histname + ('Excludes' if args.excludes else '') + '.eps', bbox_inches='tight')
        print('Closing figure', flush=True)
        plt.close()

if __name__ == '__main__':
    main()
