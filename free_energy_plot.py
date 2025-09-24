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

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import csv
import itertools
import os
import sys
from pathlib import Path

def plot_deltaenergy_vs_deltaspreads(spreads, free_energy_diffs, subdir):
    try:
        # Ensure spreads and free_energy_diffs lengths are the same
        if len(spreads) != len(free_energy_diffs):
            raise ValueError("The length of spreads and free energy diffs must be the same.")

        # Sanitize subdir to be used as filename
        sanitized_subdir = str(subdir).replace('/', '_').replace('\\', '_').replace(' ', '_')
        
        # Create and save the plot
        plt.figure(figsize=(10, 6))
        plt.ylim(0,6)
        plt.xlim(0, 10)
        plt.plot(range(len(spreads)), free_energy_diffs, marker='o', linestyle='-', color='b', label='Free Energy Difference')
        
        tick_positions = list(range(0, 11))
        custom_x_labels = [f"$2^{{{n}}}$" for n in tick_positions] 
        plt.xticks(tick_positions, custom_x_labels)
        
        plt.xlabel("log2 of -eigenvalue")
        plt.ylabel("Free Energy Difference (kcal/mol)")
        plt.title(f"Free Energy Difference vs Spread Parameter for {sanitized_subdir}")
        plt.grid(True)
        plt.legend()

        plot_filename = f'{sanitized_subdir}_Free_energy_difference_plot.png'
        plt.savefig(plot_filename)
        print(f"Plot saved to: {plot_filename}")  # Confirm saving
        plt.close()

    except Exception as e:
        print(f"An error occurred while plotting: {e}")
        

def read_and_plot_from_csv(csv_filename, typestr, colrange):
    def name(s):
        if s == 'FE':
            return 'Free energy difference'
        elif s == 'KL':
            return 'Kullback-Leibler divergence'
        elif s == 'JS':
            return 'Jensen-Shannon divergence'
        else:
            return ''

    def units(s):
        if s == 'FE':
            return 'kcal/mol'
        elif s == 'JS' or s == 'KL':
            return 'nats'
        else:
            return ''

    combined_data = {}
    
    try:
        maxy = 0.0
        with open(csv_filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)[colrange[0]:colrange[1]]
            print(f"Headers: {headers}")

            # Extract spread values from headers
            spreads = []
            for header in headers:
                if 'spread' in header.lower():
                    spread_value = float(header.split()[-1])
                    spreads.append(spread_value)
            
            log2spreads = [int(np.round(np.log2(-x))) for x in spreads]
            print(f"Spreads extracted: {spreads}")

            # Create individual plots
            for row in csvreader:
                subdir = row[0]
                free_energy_diffs = [float(value) if value != 'nan' else np.nan for value in row[colrange[0]:colrange[1]]]
                print(f"Processing subdir: {subdir}, free energy diffs: {free_energy_diffs}")

                #plot_deltaenergy_vs_deltaspreads(spreads, free_energy_diffs, subdir)
                combined_data[subdir] = free_energy_diffs
                maxy = max(maxy, max([x for x in free_energy_diffs if np.isfinite(x)]))

        # Create combined plot for all data
        plt.figure(figsize=(24, 12))
        plt.ylim(0, maxy)
        plt.xlim(min(log2spreads), max(log2spreads)+2)

        # Define colors and linestyles

        color_cycle = itertools.cycle([
            'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'olive', 'navy', 'teal', 'lime', 
            'gold', 'indigo', 'cyan', 'darkred', 'darkgreen', 'deepskyblue'
        ])
        linestyle_cycle = itertools.cycle([
            '-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (5, 1)), (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (3, 10, 1, 10))
        ])

        # Plotting all free energy differences on the same figure
        for subdir, free_energy_diffs in combined_data.items():
            print(free_energy_diffs)
            sanitized_subdir = str(subdir).replace('/', '_').replace('\\', '_').replace(' ', '_')
            plt.plot(
                range(min(log2spreads), max(log2spreads)+1), 
                free_energy_diffs, 
                marker='o', 
                linestyle=next(linestyle_cycle),
                color=next(color_cycle), 
                label=f'{name(typestr)} ({sanitized_subdir})'
            )

        tick_positions = list(range(min(log2spreads), max(log2spreads)+1))
        custom_x_labels = [f"$2^{{{n}}}$" for n in tick_positions] 
        plt.xticks(tick_positions, custom_x_labels)

        plt.xlabel("Kernalization Parameter")
        plt.ylabel(f'{name(typestr)} ({units(typestr)})')
        plt.title(f'{name(typestr)} vs Spread Parameter')
        plt.grid(True)
        plt.legend()

        # Output directory for combined plot
        output_dir = "./plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        combined_plot_filename = os.path.join(output_dir, f'{Path(csv_filename).stem}_plot.png')
        plt.savefig(combined_plot_filename)
        plt.close()

        # Confirm file save for combined plot
        if os.path.exists(combined_plot_filename):
            print(f"Combined plot saved successfully to: {combined_plot_filename}")
        else:
            print(f"Failed to save combined plot: {combined_plot_filename}")

    except Exception as e:
        print(f"An error occurred while reading the CSV or plotting the graph: {e}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("csv_filename", type=str)
    parser.add_argument("--type", default="", type=str, required=False)
    parser.add_argument("--colrange", nargs=2, type=int, required=False)

    args = parser.parse_args()

    if not os.path.isfile(args.csv_filename):
        print(f"Error: The file '{csv_filename}' does not exist.")
    else:
        read_and_plot_from_csv(args.csv_filename, args.type, args.colrange)
