import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(os.path.realpath(__file__))
plots_folder = os.path.dirname(curr_dir)
stats_folder = os.path.dirname(plots_folder)

many_gaps_dir_all = os.path.join(stats_folder, 'evaluation', 'realistic', 'all')
many_gaps_dir_area = os.path.join(stats_folder, 'evaluation', 'realistic', 'area')

dirs = [many_gaps_dir_all, many_gaps_dir_area]

types = ['linear', 'gti', 'impute', 'imputeR']

def create_closeness_plot(type:str, y_label:str, title:str):
    sparsification_rates = ['All', 'Area']
    for index, directory in enumerate(dirs):
        linear_interpolation = [0, 0]
        gti = [0, 0]
        dgvti = [0, 0]
        dgvtiR = [0, 0]

        csv_files = Path(directory).rglob('*.csv')
        csv_files = [str(file) for file in csv_files]

        for csv in csv_files:
            if ('linear' in csv):
                linear_interpolation[index] = pd.read_csv(csv)[type].mean() * 1000
            elif ('gti' in csv):
                gti[index] = pd.read_csv(csv)[type].mean() * 1000
            elif ('imputeR' in csv):
                dgvtiR[index] = pd.read_csv(csv)[type].mean() * 1000
            else:
                dgvti[index] = pd.read_csv(csv)[type].mean() * 1000

        # Patterns for the bars
        patterns = ['--', '..', 'xx', 'o']
        fig, ax1 = plt.subplots()
        width = 0.2

        # Create the figure and axis
        x = np.arange(len(sparsification_rates))

        # Bar plot for dynamic time warping distance
        ax1.bar(x - 1.5 * width, linear_interpolation, width, label='Linear Interpolation', hatch=patterns[0])
        ax1.bar(x - 0.5 * width, gti, width, label='GTI', hatch=patterns[1])
        ax1.bar(x + 0.5 * width, dgvti, width, label='DGIVT', hatch=patterns[2])
        ax1.bar(x + 1.5 * width, dgvtiR, width, label='DGIVTR', hatch=patterns[3])

        # Set labels and title
        ax1.set_xlabel('Sparsification rate')
        ax1.set_ylabel('')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sparsification_rates)
        ax1.legend(loc='upper left')
        # Show and save the plot
        area = 'All' if 'all' in directory else 'Area'
        plt.title(f'{area} - {title}')
        plt.tight_layout()

        fig_type = 'area' if 'area' in directory else 'all'

        type = type.replace(' ', '_').lower()
        plt.savefig(f'./{fig_type}_many_gap_{type}.png')

def create_type_plots():
    create_closeness_plot('DTW', 'Dynamic Time Warping (in meters)', 'Many Gaps - Dynamic Time Warping')
    create_closeness_plot('Frechet Distance', 'Frechet Distance (in meters)', 'Many Gaps - Frechet Distance')


create_type_plots()