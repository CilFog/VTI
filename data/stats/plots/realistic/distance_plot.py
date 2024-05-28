import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(os.path.realpath(__file__))
plots_folder = os.path.dirname(curr_dir)
stats_folder = os.path.dirname(plots_folder)

many_gaps_dir_all = os.path.join(stats_folder, 'evaluation', 'all', 'realistic')
many_gaps_dir_area = os.path.join(stats_folder, 'evaluation', 'area', 'realistic')

dirs = [many_gaps_dir_area]

types = ['linear', 'gti', 'impute']

def create_closeness_plot(type:str, y_label:str, title:str):
    sparsification_rates = ['Area']
    for index, directory in enumerate(dirs):
        linear_interpolation = [0]
        gti = [0]
        dgvti = [0]

        csv_files = Path(directory).rglob('*.csv')
        csv_files = [str(file) for file in csv_files]
        
        for csv in csv_files:
            df = pd.read_csv(csv)
            if 'linear' in csv:
                linear_interpolation[index] = df[type].mean() * 1000
            elif 'gti' in csv:
                gti[index] = df[type].mean() * 1000
            else:
                dgvti[index] = df[type].mean() * 1000

        # Patterns for the bars
        patterns = ['--', '..', 'xx', 'o']
        fig, ax1 = plt.subplots()
        width = 0.2
        space = 0.5  # Space between bars

        # Create the figure and axis
        x = np.arange(len(sparsification_rates))

        # Bar plot for dynamic time warping distance
        ax1.bar(x - width - space, linear_interpolation, width, label='Linear Interpolation', hatch=patterns[0])
        ax1.bar(x, gti, width, label='GTI', hatch=patterns[1])
        ax1.bar(x + width + space, dgvti, width, label='DGIVT', hatch=patterns[2])

        # Set labels and title
        ax1.set_xlabel('Sparsification rate', fontsize=16)
        ax1.set_ylabel(f'{y_label}', fontsize=16)
        ax1.set_xticks(x)  
        ax1.set_xticklabels(sparsification_rates)
        ax1.legend(loc='upper left')
        # Show and save the plot
        plt.title(f'Area - {title}', fontsize=16)
        plt.tight_layout()

        plt.savefig(f'./area_realistic_{type.lower()}.png')

def create_type_plots():
    create_closeness_plot('DTW', 'Dynamic Time Warping (in meters)', 'Realistic - Dynamic Time Warping')
    create_closeness_plot('Frechet Distance', 'Frechet Distance (in meters)', 'Realistic - Frechet Distance')


create_type_plots()