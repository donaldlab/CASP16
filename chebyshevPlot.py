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
from pathlib import Path
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from adjustText import adjust_text
import re

import cdiotools.cdios
import cdiotools.extrema


alphas = [cdiotools.cdios.findAlphaString(alpha, 2) for alpha in cdiotools.cdios.findAlphas(101, 2)]
exclude = ['088', '139', '002', '084', '003', '004', '005', '100']
#vectornames = ['Pearson Correlation', 'Slope', 'Y-intercept', 'Mode angular distance', 'Free energy difference', 'Isotropicity', 'Zero-moded free energy difference']
vectornames = ['Pearson Correlation', 'Slope', 'Y-intercept', 'Anisotropicity']


def normalize(value, ideal, maxDistance):
    normed = np.abs(value - ideal) / maxDistance
    assert normed <= 1.0
    return normed


class PredStats:

    def __init__(self, prednum):
        self.prednum = prednum

    def normalized(self):
        normed = PredStats(self.prednum)
        normed.pearsonr = normalize(self.pearsonr, 1.0, 2.0)

        # From python rotateSaupesExtrema.py --gridfn grids/biggrid.qua
        normed.slope = normalize(self.slope, 1.0, 15.53) # 15.48 is 1.0 - -14.53 (ideal minus the min possible)
        normed.yintercept = normalize(self.yintercept, 0.0, 8.80) # This is 0 - -8.80 (ideal minus min possible)

        normed.modedist = normalize(self.modedist, 0.0, np.pi)

        minDFEAlpha = min(self.fediff, key=self.fediff.get)
        minDFE = self.fediff[minDFEAlpha]
        assert minDFE == min(self.fediff.values())
        normed.fediff = normalize(minDFE, 0.0, cdiotools.extrema.maxdFE[minDFEAlpha])

        minAniso = np.min(list(cdiotools.extrema.isodFE.values()))
        maxAniso = np.max(list(cdiotools.extrema.isodFE.values()))
        if self.iso > maxAniso:
            normed.iso = normalize(self.iso, maxAniso, 2.50 - maxAniso)
        elif self.iso < minAniso:
            normed.iso = normalize(self.iso, minAniso, 2.50 - maxAniso)
        else:
            assert self.iso <= maxAniso and self.iso >= minAniso
            normed.iso = 0
        #print(f'{self.prednum}: self.iso: {self.iso}; normed.iso: {normed.iso}; minAniso: {minAniso}; maxAniso: {maxAniso}')

        #minModeMinMixDFEAlpha = min(self.moderot, key=self.moderot.get)
        #minModeMinMixDFE = self.moderot[minModeMinMixDFEAlpha]
        #assert minModeMinMixDFE == min(self.moderot.values())
        #normed.moderot = normalize(minModeMinMixDFE, 0.0, cdiotools.extrema.modeMinMixdFE[minModeMinMixDFEAlpha])
        #normed.moderot = normalize(self.moderot[minDFEAlpha], 0.0, cdiotools.extrema.modeMinMixdFE[minDFEAlpha])

        return normed

    def chebyshev(self):
        normed = self.normalized()
        #vector = np.array([normed.pearsonr, normed.slope, normed.yintercept, normed.modedist, normed.fediff, normed.iso, normed.moderot])
        vector = np.array([normed.pearsonr, normed.slope, normed.yintercept, normed.iso])
        imax = vector.argmax()
        vmax = vector.max()
        #print(self.prednum, vectornames[imax], round(vmax, 2))
        return self.prednum, vmax, vectornames[imax], *vector, self.pearsonr, self.slope, self.yintercept, self.iso


def plotChebyshev(plotname, chebyshevs):
    labels = [x[0] for x in chebyshevs]
    vals = [x[1] for x in chebyshevs]
    plt.figure(figsize=(1, 10))
    ax = plt.subplot(111)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_ylim([1.0, 0.0])
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.scatter(np.zeros(len(vals)), vals, c=['red' if l in exclude else 'blue' for l in labels])
    texts = []
    for label, val, *_ in chebyshevs:
        texts.append(ax.annotate(label, (0.012, val-0.004), size=6))
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
        target_x = np.zeros(len(vals)),
        target_y = vals,
        only_move={'text': 'y'},
        time_lim=5.0
    )
    plt.savefig(plotname + '.png', dpi=300)
    plt.close()


def rowToPrednum(rowname):
    return re.fullmatch('T1200TS([0-9]{3}).cleaned', rowname).group(1)


def main():
    argparser = ArgumentParser()
    argparser.add_argument("--rdcstats", default='currentResults7may25/rdcStats.csv', type=str, required=False)
    argparser.add_argument("--angstats", default='computedStats/angular_distance_-32_2.csv', type=str, required=False)
    argparser.add_argument("--festats", default='currentResults7may25/T1200_cleaned_freeEnergyDiffs.csv', type=str, required=False)
    argparser.add_argument("--plotname", default='chebyshev', type=str, required=False)
    argparser.add_argument("--out", default='chebyshev', type=str, required=False)
    argparser.add_argument("--exclude", default=False, action='store_true')
    argparser.add_argument("--nowrite", default=False, action='store_true')
    argparser.add_argument("--debug", default=False, action='store_true')

    args = argparser.parse_args()

    stats = {}
    with open(args.rdcstats, newline='') as rdcstatsfile:
        rdcStatsReader = csv.DictReader(rdcstatsfile)
        for row in rdcStatsReader:
            prednum = row['Predictor Number']
            if prednum in exclude and args.exclude:
                continue
            predStats = PredStats(prednum)
            predStats.pearsonr = float(row['Pearson Correlation'])
            predStats.slope = float(row['Slope'])
            predStats.yintercept = float(row['Y Intercept'])
            stats[prednum] = predStats

    with open(args.angstats, newline='') as modediststatsfile:
        modedistStatsReader = csv.DictReader(modediststatsfile)
        for row in modedistStatsReader:
            prednum = rowToPrednum(row['Predictor'])
            if prednum in exclude and args.exclude:
                continue
            modedist1 = float(row['Angular Distance, Solution 1'])
            modedist2 = float(row['Angular Distance, Solution 2'])
            stats[prednum].modedist = min(modedist1, modedist2)

    with open(args.festats, newline='') as festatsfile:
        freeEnergyStatsReader = csv.DictReader(festatsfile)
        for row in freeEnergyStatsReader:
            prednum = rowToPrednum(row['ensemble'])
            if prednum in exclude and args.exclude:
                continue
            stats[prednum].fediff = {}
            for alpha in alphas:
                stats[prednum].fediff[alpha] = float(row[f'dFE, sol {alpha}, spread -32'])
            stats[prednum].iso = float(row['dFE, sol unif, spread -32'])

    for prednum, stat in stats.items():
        print(f'    {prednum} & {stat.pearsonr:.4f} & {stat.slope:.4f} & {stat.yintercept:.4f} & {stat.iso:.4f} \\\\')

    allslopes = [stats[prednum].slope for prednum in stats]
    allyintercept = [stats[prednum].yintercept for prednum in stats]
    allpearsonr = [stats[prednum].pearsonr for prednum in stats]
    allaniso = [stats[prednum].iso for prednum in stats]

    print('Ranges:')
    print('    Slope:         ', min(allslopes), max(allslopes))
    print('    Pearson R:     ', min(allpearsonr), max(allpearsonr))
    print('    Y-intercept:   ', min(allyintercept), max(allyintercept))
    print('    Anisotropicity:', min(allaniso), max(allaniso))

    #with open(args.moderotstats, newline='') as festatsfile:
    #    freeEnergyStatsReader = csv.DictReader(festatsfile)
    #    for row in freeEnergyStatsReader:
    #        prednum = rowToPrednum(row['ensemble'])
    #        if prednum in exclude and args.exclude:
    #            continue
    #        stats[prednum].moderot = {}
    #        for alpha in alphas:
    #            stats[prednum].moderot[alpha] = float(row[f'dFE, sol {alpha}, spread -32'])

    chebyshevs = [predstats.chebyshev() for predstats in stats.values()]

    chebyshevs.sort(key = lambda x: x[1])

    if not args.nowrite:
        with open(args.out + '.csv', 'w') as outcsv:
            writer = csv.writer(outcsv)
            header = ['Ensemble', 'Chebyshev distance', 'Limiting quantity'] + [v + ' (normalized)' for v in vectornames] + vectornames
            writer.writerow(header)
            for tup in chebyshevs:
                writer.writerow(tup)
        plotChebyshev(args.plotname + ('Excludes' if args.exclude else ''), chebyshevs)


if __name__ == '__main__':
    main()
