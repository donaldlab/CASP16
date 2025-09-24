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
import matplotlib.pyplot as plt
import csv
from adjustText import adjust_text
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

import cdiotools.rdcs


#Graph: plot OLC1 and 2 Back Calculated RDC vs Experimental Values
def rmbPlot(rdcs, expRdcs, outPath, combined_reg, solution_name = None, debug = False):

    plt.rcParams["font.family"] = "serif"

    backcalculated_rdc_olc1 = rdcs.rdcs['C'][4]
    backcalculated_rdc_olc2 = rdcs.rdcs['C'][5]
    experimental_rdc_olc1 = expRdcs.rdcs['C'][4]
    experimental_rdc_olc2 = expRdcs.rdcs['C'][5]

    if len(backcalculated_rdc_olc1) != len(experimental_rdc_olc1) or len(backcalculated_rdc_olc2) != len(experimental_rdc_olc2):
        raise ValueError(f"Back-calculated RDC values for OLC1 or OLC2 and experimental RDC values have different lengths")

    all_predicted_rdc = np.concatenate([backcalculated_rdc_olc1, backcalculated_rdc_olc2])
    y_min, y_max = all_predicted_rdc.min(), all_predicted_rdc.max()
    y_buffer = (y_max - y_min) * 0.02  
    y_min -= y_buffer
    y_max += y_buffer
    fig = plt.figure(figsize=(10,6))
    grid = fig.add_gridspec(1, 2, width_ratios=[4, 1]) 
    ax = fig.add_subplot(grid[0, 0])

    alpha = 0.7

    def onescatter(exp, back, col, number):
        label = f'OLC{number} Back-Calculated RDC Data Points'
        plt.scatter(exp, back, color = col, alpha=alpha, label = label)
        #_, _, bars = plt.errorbar(exp, back, xerr=1.96 * cdiotools.rdcs.experimental_error, ecolor = col, fmt='none')
        _, _, bars = plt.errorbar(exp, back, xerr=cdiotools.rdcs.experimental_error, ecolor = col, fmt='none')
        for bar in bars:
            bar.set_alpha(alpha)

    onescatter(experimental_rdc_olc1, backcalculated_rdc_olc1, 'b', 1)
    onescatter(experimental_rdc_olc2, backcalculated_rdc_olc2, 'y', 2)
    
    ax.set_xlim(-2, 2)
    if solution_name:
        ax.set_ylim(-2, 2)
    else:   
        ax.set_ylim(y_min , y_max)
    
    if solution_name:
        ax.set_aspect('equal', adjustable='box')
        
    plt.plot([-2, 2], [-2, 2], linestyle='--', color='r', label='Ideal Agreement (y=x)')
    
    plt.plot(
        [-2, 2],
        [combined_reg.slope * -2 + combined_reg.intercept, combined_reg.slope * 2 + combined_reg.intercept],
        linestyle='--', color='darkgreen', label='Best-Fit Line'
    )
    
    plt.xlabel('Experimental NH RDC (Hz)', fontsize = 14)
    plt.ylabel('Backcalculated NH RDC (Hz)', fontsize = 14)
    plt.grid(True)
    
    ax_text = fig.add_subplot(grid[0, 1])
    ax_text.axis('off')
    legend_elements = [
        Line2D([0], [0], color="b", marker="o", linestyle="", label="OLC1 Data Points"),
        Line2D([0], [0], color="y", marker="o", linestyle="", label="OLC2 Data Points"),
        Line2D([0], [0], color="r", linestyle="--", label="Ideal Agreement (y=x)"),
        Line2D([0], [0], color="darkgreen", linestyle="--", label="Best-Fit Line")
    ]
    ax_text.legend(handles=legend_elements, loc="upper center", fontsize=14, frameon=True, markerscale=1, bbox_to_anchor=(0.0, 0.9))
    
    combined_box_test = (
        f"Pearson Correlation: {combined_reg.rvalue:.2f}\n".replace("-", "−")+
        f"Slope: {combined_reg.slope:.2f}\n".replace("-", "−")+
        f"Y Intercept: {combined_reg.intercept:.2f}".replace("-", "−")
    )
    
    ax_text.text(
        0.9, 0.34, combined_box_test , fontsize=14, color="darkgreen", linespacing=1.4,
        bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.5"),
        verticalalignment="top", horizontalalignment="right", transform=ax_text.transAxes
    )
    
    blue_box_text = (
        "Ideal Pearson Correlation: 1.0\n"
        "Ideal Slope: 1.0\n"
        "Ideal Y Intercept: 0.0"
    )
    
    ax_text.text(0.9, 0.53, blue_box_text, fontsize=14, color='red', linespacing=1.4,
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'),
            verticalalignment='top', horizontalalignment='right', transform=ax_text.transAxes)
    
    plt.tight_layout()
    print('Saving', outPath)
    plt.savefig(outPath, format = 'png')
    plt.close()


def plot2d(output_file, xdata, ydata, labels, xlabel, ylabel, title, idealx = None, idealx2 = None, idealy = None, idealy2 = None, colors = None, xlim = None, ylim = None, xlog = False, yticks = None, legend = False, colorMapInv = None, labelsize = None):
    
    fig, ax = plt.subplots(figsize=(10,6))

    prettyMeaningMap = {"MD": "Molecular dynamics", "DL": "Deep learning (DL)", "DL-MSA": "DL with custom MSA", "OTHER": "Other"}
    for color, meaning in colorMapInv.items():
        xdataForColor = [xdata[i] for i in range(len(colors)) if colors[i] == color]
        ydataForColor = [ydata[i] for i in range(len(colors)) if colors[i] == color]
        print("color: ", color)
        scatter = ax.scatter(xdataForColor, ydataForColor, edgecolor='k', s=100, alpha=0.7, c=color, label=prettyMeaningMap[meaning])

    if legend:
        ax.legend()

    if xlog:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks([1, 10, 100])

    if yticks is not None:
        ax.set_yticks(yticks)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    elif np.all(np.array(xdata) >= 0):
        ax.set_xlim([1, None])
    else:
        ax.set_xlim([None, None])

    if ylim is not None:
        ax.set_ylim(ylim)
    elif np.all(np.array(ydata) >= 0):
        ax.set_ylim([0, None])
    else:
        ax.set_ylim([None, None])

    if idealx is not None:
        # Add the "ideal" point
        ax.scatter([idealx], [idealy], color='darkorange', edgecolor='black', s=150, zorder=5)
        if idealx2 is not None:
            ax.scatter([idealx2], [idealy], color='darkorange', edgecolor='black', s=150, zorder=5)
            ax.plot([idealx, idealx2], [idealy, idealy], color='darkorange', linewidth=6, linestyle='solid')
        if idealy2 is not None:
            ax.scatter([idealx], [idealy2], color='darkorange', edgecolor='black', s=150, zorder=5)
            ax.plot([idealx, idealx], [idealy, idealy2], color='darkorange', linewidth=8, linestyle='solid')
        # Add text near the "ideal" point
        idealTextX = idealx + 0.05 if xlog else idealx + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015
        idealTextY = idealy + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
        plt.text(idealTextX, idealTextY, "Ideal", fontsize=9, color='darkorange', ha='left', va='center', weight='bold')
    
    # Add axis labels and title
    plt.xlabel(xlabel, fontsize=labelsize if labelsize else 14)
    plt.ylabel(ylabel, fontsize=labelsize if labelsize else 14)
    if title:
        plt.title(title, fontsize=14, pad=15)
    
    # Add grid for better visualization
    ax.grid(alpha=0.3)
    
    # Add labels with dynamic adjustment
    texts = []
    for x, y, label in zip(xdata, ydata, labels):
        texts.append(plt.text(x, y, label, fontsize=9, alpha=0.7))
    
    # Adjust text positions to avoid overlap
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
        force_points=0.3,
        force_text=0.5,
        expand_points=(1.2, 1.2),
        expand_text=(1.2, 1.2),
        only_move={'text': 'xy', 'points': 'xy'}
    )
    
    # Save the plot to a file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def findPredColors(categfile, labels):
    predCateg = {}
    if categfile:
        with open(categfile, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                predCateg[row['predictor']] = row['category']
    colorMap = {'MD': 'blue', 'DL': 'green', 'UNK': 'gray', 'DL-MSA': 'greenyellow', 'OTHER': 'lightblue'}
    predColorMap = {pred: colorMap[categ] for pred, categ in predCateg.items()}
    predColorList = [predColorMap[pred] if pred in predColorMap else 'black' for pred in labels]
    colorMapInv = {color: meaning for meaning, color in colorMap.items() if meaning != "UNK"}
    return predColorList, colorMapInv


def filterAndOrderData(data, labels):
    newdata = []
    print('data:', data)
    for label in labels:
        print('label:', label)
        newdata.append(data[label])
    return newdata
