# This script postprocesses raw data from experimental campaigns conducted on a composite material 
# made of lunar regolith simulant (EAC-1a) and PEEK. 
# The objective is to analyze mechanical behavior based on data from compression and bending tests. 
#
# A full-factorial L12 Design of Experiments (DoE) approach was applied to account for variations 
# in processing parameters during manufacturing. Each combination of parameters was tested using 
# six identical specimens. 
#
# Data format:
# - Test results: Filenames follow the pattern BDoE00.txt (bending) and CDoE00.txt (compression), 
#   where "00" represents the specimen number. Columns include: 'Time', 'Displ', 'Load'.
# - Specimen dimensions: Filenames BDoE_dim.txt (bending) and CDoE_dim.txt (compression). 
#   Columns: 'depth', 'width' (bending), 'diameter', 'height' (compression).
#
# Calculations:
# - Strength: Calculated at the maximum load sustained by each specimen.
# - Modulus: An adaptive algorithm identifies the linear region of the stress-strain curve 
#   using rolling linear regression. This method dynamically adjusts to variations in curve trends 
#   and ensures robustness by requiring slope consistency (variation ≤10%) across overlapping windows.
#
# V1.0 - 03/12/2024 - Roberto Torre
# Spaceship EAC - European Astronaut Centre / European Space Agency, Cologne, DE

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --->>>> Flag to determine the type of test: 'bending' or 'compression'
test_type = 'bending'  # Set test type here

# --->>>> Directory containing the data files
folder = 'data'

# --->>>> Set whether intermediate graphs for slopes should be displayed
# This helps visualizing how the modulus is calculated
slope_graphs_on = 'no' 

######################################
######################################
# This section loads the data and calculates the strenghts.

# Load specimen dimensions based on the test type
if test_type == 'compression':
    file_name = "CDoE_dim.txt"
elif test_type == 'bending':
    file_name = "BDoE_dim.txt"
else:
    raise ValueError("Unknown test type. Please specify 'compression' or 'bending'.")

file_path = os.path.join(folder, file_name)

# Read and preprocess specimen dimensions
with open(file_path, 'r') as file:
    data = file.read()

# Remove introductory and closing rows
lines = data.split('\n')

# Split data into a list of lists and convert to floats
data_organized = [row.split(';') for row in lines]
numeric_data = [[float(value.replace(',', '.')) for value in row] for row in data_organized]

# Create a DataFrame from the dimensions data
if test_type == 'compression':
    df0 = pd.DataFrame(numeric_data, columns=['diameter', 'height'])
elif test_type == 'bending':
    df0 = pd.DataFrame(numeric_data, columns=['depth', 'width'])
else:
    raise ValueError("Unknown test type. Please specify 'compression' or 'bending'.")

# Dictionary to store DataFrames for each specimen
dataframes = {}

# List to track the maximum stress values
max_stress_values = []

# Number of specimens tested (including spares and those to discard)
specimens = 73  # Fixed value for both test types

# Process each specimen data file
for file_number in range(1, specimens):
    # Construct file name
    if test_type == 'compression':
        file_name = f"CDoE{file_number}.txt"
    elif test_type == 'bending':
        file_name = f"BDoE{file_number}.txt"
    else:
        raise ValueError("Unknown test type. Please specify 'compression' or 'bending'.")
    file_path = os.path.join(folder, file_name)

    # Read and preprocess raw test data
    with open(file_path, 'r') as file:
        data = file.read()

    # Extract relevant data (removing headers and footers)
    lines = data.split('\n')
    trimmed_data = lines[2:-1]

    # Organize data into a list of lists and convert to floats
    organized_data = [row.split(';') for row in trimmed_data]
    numeric_data = [[float(value.replace(',', '.')) for value in row] for row in organized_data]

    # Create DataFrame for test data (adjust columns based on test type)
    if test_type == 'compression':
        df = pd.DataFrame(numeric_data, columns=['Time', 'Displ', 'Load'])
    elif test_type == 'bending':
        df = pd.DataFrame(numeric_data, columns=['Time', 'Load', 'Displ'])
    else:
        raise ValueError("Unknown test type. Please specify 'compression' or 'bending'.")

    # Offset columns so they start from zero
    for col in ['Time', 'Load', 'Displ']:
        df[col] = df[col] - df[col].iloc[0]

    # Filter data where 'Load' >= 1 to exclude noise
    df = df[df['Load'] >= 1].copy()

    # Recalculate offsets after filtering
    for col in ['Time', 'Load', 'Displ']:
        df[col] = df[col] - df[col].iloc[0]

    # Compute Stress and Strain based on the test type
    if test_type == 'compression':
        df['Stress'] = (df['Load'] + 1) / (3.14 * df0.loc[file_number - 1, 'diameter'] ** 2 / 4)
        df['Strain'] = df['Displ'] / df0.loc[file_number - 1, 'height']
    elif test_type == 'bending':
        df['Stress'] = (df['Load'] + 1) * (3 * 100) / (2 * df0.loc[file_number - 1, 'width'] * df0.loc[file_number - 1, 'depth'] ** 2)
        df['Strain'] = df['Displ'] * (6 * df0.loc[file_number - 1, 'depth'] / 10000)

    # Store the processed DataFrame
    dataframes[file_number] = df

    # Find and store the maximum stress value
    max_stress = df['Stress'].max()
    max_stress_values.append(max_stress)

# Group maximum stress values into runs
runs = [
    [max_stress_values[i - 1] for i in indices]
    for indices in [
        [1, 13, 25, 37, 49, 61], [2, 14, 26, 38, 50, 62], [3, 15, 27, 39, 51, 63],
        [4, 16, 28, 40, 52, 64], [5, 17, 29, 41, 53, 65], [6, 18, 30, 42, 54, 66],
        [7, 19, 31, 43, 55, 67], [8, 20, 32, 44, 56, 68], [9, 21, 33, 45, 57, 69],
        [10, 22, 34, 46, 58, 70], [11, 23, 35, 47, 59, 71], [12, 24, 36, 48, 60, 72],
    ]
]

# Calculate mean and standard deviation for each run
mean_values = [np.mean(run) for run in runs]
std_devs = [np.std(run) for run in runs]
run_labels = [f"Run {i + 1}" for i in range(len(runs))]

# Create a DataFrame to store strength data
columns = [
    'Run',
    'Mean Strength [MPa]',
    'Standard Deviation [MPa]',
] + [f'Value {i}' for i in range(1, 7)]
strength_data = pd.DataFrame(columns=columns)

# Populate strength data DataFrame
for i, (mean, std, run) in enumerate(zip(mean_values, std_devs, runs)):
    row = [run_labels[i], mean, std] + run
    strength_data.loc[i] = row

# Save results to Excel
output_file = 'Cstre.xlsx' if test_type == 'compression' else 'Bstre.xlsx'
strength_data.to_excel(output_file, index=False)

# Identify and remove the most distant outlier from the mean for each run
for i, row in strength_data.iterrows():
    # Select strength values for calculation (excluding descriptive columns)
    data = row[3:9].astype(float)  # Convert strength values to float

    # Calculate the mean and standard deviation for the current run
    mean_value = data.mean()
    std_dev = data.std()

    # Find the index of the value with the largest deviation from the mean
    max_deviation_index = np.abs(data - mean_value).idxmax()

    # Set the outlier value to NaN
    strength_data.at[i, max_deviation_index] = np.nan

    # Recalculate the mean and standard deviation without the outlier
    updated_data = data.drop(max_deviation_index)
    strength_data.at[i, 'Mean strength [MPa]'] = updated_data.mean()
    strength_data.at[i, 'Standard Deviation [MPa]'] = updated_data.std()

# Rearrange data for plotting (excluding descriptive and statistical columns)
data_for_plot = strength_data.iloc[:, 3:9].values.T  # Transpose of strength values

# Extract updated statistics for plotting
mean_values_ = strength_data['Mean strength [MPa]'].values
std_devs_ = strength_data['Standard Deviation [MPa]'].values
runs_labels = strength_data['Run'].values

# Save the updated DataFrame (without outliers) to an Excel file
output_file = 'Cstre_.xlsx' if test_type == 'compression' else 'Bstre_.xlsx'
strength_data.to_excel(output_file, index=False)

# Plot

runs_labels = list(range(1, 13))  # Run labels from 1 to 12

# Plotting parameters
plt.figure(figsize=(7.5, 6.25))  # Set figure size in inches (width, height)
blue_bar_width = 0.35 * 0.75  # Narrower bar width for "Before Outlier Removal" (25% thinner)
orange_bar_width = 0.35 * 1.25  # Wider bar width for "After Outlier Removal" (25% thicker)
gap = 0.1  # Horizontal gap between bars of the same group
x_pos = np.arange(len(runs_labels))  # Bar positions on the x-axis

# Plot original data (before outlier removal)
plt.bar(
    x_pos, mean_values, blue_bar_width, yerr=std_devs, capsize=5,
    label='Before Outlier Removal', color='blue'
)

# Plot data after outlier removal with an offset for separation
plt.bar(
    x_pos + blue_bar_width + gap, mean_values_, orange_bar_width, yerr=std_devs_, capsize=5,
    label='After Outlier Removal', color='orange'
)

# Set custom x-axis tick positions and labels
positions = x_pos + (blue_bar_width + gap + orange_bar_width) / 2
plt.xticks(positions, runs_labels, fontsize=12)

# Adjust x-axis limits to reduce unnecessary empty space
plt.xlim(-0.3, len(runs_labels) - 0.2)

# Add vertical red dotted lines for visual grouping
plt.axvline(x=3.75, color='red', linestyle='--', linewidth=1.5)  # Divider between 4 and 5
plt.axvline(x=7.75, color='red', linestyle='--', linewidth=1.5)  # Divider between 8 and 9

# Add group labels ("PEEK X wt%") above respective sections
if test_type == 'compression':
    plt.text(1.75, plt.ylim()[1] * 0.92, 'PEEK 5 wt%', fontsize=10, ha='center',
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.text(5.75, plt.ylim()[1] * 0.92, 'PEEK 10 wt%', fontsize=10, ha='center',
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.text(9.75, plt.ylim()[1] * 0.92, 'PEEK 15 wt%', fontsize=10, ha='center',
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
elif test_type == 'bending':
    plt.text(1.75, plt.ylim()[1] * 1.00, 'PEEK 5 wt%', fontsize=10, ha='center',
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.text(5.75, plt.ylim()[1] * 1.00, 'PEEK 10 wt%', fontsize=10, ha='center',
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.text(9.75, plt.ylim()[1] * 1.00, 'PEEK 15 wt%', fontsize=10, ha='center',
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
else:
    print("Unknown test type. Please specify 'compression' or 'bending'.")

# Add axis labels, title, and legend
plt.xlabel('DoE run no.', fontsize=14)
plt.yticks(fontsize=12)
plt.legend(bbox_to_anchor=(0.0, 1.0), loc='upper left')  # Position legend at the top left

# Adjust y-axis label and limits based on test type
if test_type == 'compression':
    plt.ylabel('Compression Strength [MPa]', fontsize=14)
    plt.ylim(0, 30)  # Set y-axis range for compression
    plt.savefig('DoE_Cstre_comb.pdf', bbox_inches='tight')  # Save plot as PDF
elif test_type == 'bending':
    plt.ylabel('Bending Strength [MPa]', fontsize=14)
    plt.ylim(0, 20)  # Set y-axis range for bending
    plt.savefig('DoE_Bstre_comb.pdf', bbox_inches='tight')  # Save plot as PDF
else:
    print("Unknown test type. Please specify 'compression' or 'bending'.")

# Display the plot
plt.show()

######################################
######################################
# This section calculates the moduli.

# Determine test type and set parameters accordingly
if test_type == 'compression':
    threshold = 1              # Stress threshold; disregard values below it
    window_size = 75           # Window size for rolling regression
    tolerance = 0.10           # Tolerance for slope variation
elif test_type == 'bending':
    threshold = 0.1            # Stress threshold; disregard values below it
    window_size = 75           # Window size for rolling regression
    tolerance = 0.10           # Tolerance for slope variation
else:
    raise ValueError("Unknown test type. Please specify 'compression' or 'bending'.")

# Define groups of indices for processing
groups_of_i = [
    [1, 13, 25, 37, 49, 61],
    [2, 14, 26, 38, 50, 62],
    [3, 15, 27, 39, 51, 63],
    [4, 16, 28, 40, 52, 64],
    [5, 17, 29, 41, 53, 65],
    [6, 18, 30, 42, 54, 66],
    [7, 19, 31, 43, 55, 67],
    [8, 20, 32, 44, 56, 68],
    [9, 21, 33, 45, 57, 69],
    [10, 22, 34, 46, 58, 70],
    [11, 23, 35, 47, 59, 71],
    [12, 24, 36, 48, 60, 72]
]

# Initialize a list to store results
data = []

# Helper function to identify the longest region with constant slope
def find_constant_slope_region(slopes, tolerance):
    longest_region = []
    current_region = []

    for i in range(len(slopes)):
        if not current_region:
            current_region.append(i)
        else:
            max_slope = max(slopes[current_region[0]:i + 1])
            min_slope = min(slopes[current_region[0]:i + 1])
            if max_slope - min_slope <= tolerance * max_slope:
                current_region.append(i)
            else:
                if len(current_region) > len(longest_region):
                    longest_region = current_region
                current_region = [i]

    if len(current_region) > len(longest_region):
        longest_region = current_region

    return longest_region

# Process each group
for idx, group_i in enumerate(groups_of_i, start=1):
    slopes_by_group = []

    for index in group_i:
        x = dataframes[index].iloc[:, 4].values  # Strain
        y = dataframes[index].iloc[:, 3].values  # Stress

        # Filter data based on threshold
        indices = y > threshold
        filtered_y = y[indices] - np.min(y[indices])
        filtered_x = x[indices] - np.min(x[indices])

        # Locate maximum stress value and perform rolling regression
        max_index = np.argmax(filtered_y)
        slopes = []

        for i in range(0, max_index - window_size + 1):
            x_window = filtered_x[i:i + window_size]
            y_window = filtered_y[i:i + window_size]
            slope, _, _, _, _ = linregress(x_window, y_window)
            slopes.append(slope)

        slopes = np.array(slopes)
        constant_slope_region = find_constant_slope_region(slopes, tolerance)

        if constant_slope_region:
            start_index = constant_slope_region[0]
            end_index = constant_slope_region[-1] + window_size
            average_slope = np.mean(slopes[constant_slope_region])
        else:
            average_slope = None

        slopes_by_group.append(average_slope)

        # Optionally display graphs
        if slope_graphs_on == 'yes' and average_slope is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(slopes)), slopes, label='Slopes')
            if constant_slope_region:
                plt.axvspan(start_index, end_index - window_size, color='yellow', alpha=0.3, label='Constant Slope Region')
                plt.axhline(average_slope, color='orange', label=f'Average Slope: {average_slope:.4f}')
            plt.xlabel('Window Start Index')
            plt.ylabel('Slope')
            plt.legend()
            plt.show()

    # Compute mean and standard deviation for the group
    mean_slope = np.mean(slopes_by_group)
    std_slope = np.std(slopes_by_group)
    data.append(slopes_by_group + [mean_slope, std_slope])

# Create DataFrame and save to Excel
columns = [f'Slope {i}' for i in range(1, 7)] + ['Mean', 'Std Dev']
slope_data = pd.DataFrame(data, columns=columns)

# Calcola i valori medi e le deviazioni standard aggiornati per ciascun run
mean_values = slope_data['Mean'].values
std_devs = slope_data['Std Dev'].values

output_file = 'Cmod.xlsx' if test_type == 'compression' else 'Bmod.xlsx'
slope_data.to_excel(output_file, index=False)

# Identify and exclude outliers, recompute statistics
for i, group in slope_data.iterrows():
    mean_value = np.mean(group[:-2])
    distances = np.abs(group[:-2] - mean_value)
    outlier_index = distances.idxmax()
    slope_data.loc[i, outlier_index] = np.nan
    slope_data.loc[i, 'Mean_'] = np.nanmean(group[:-2])
    slope_data.loc[i, 'Std Dev_'] = np.nanstd(group[:-2])

# Calcola i valori medi e le deviazioni standard aggiornati per ciascun run
mean_values_updated = slope_data['Mean_'].values
std_devs_updated = slope_data['Std Dev_'].values

# Save updated DataFrame and optional plot
output_file_updated = 'Cmod_.xlsx' if test_type == 'compression' else 'Bmod_.xlsx'
slope_data.to_excel(output_file_updated, index=False)




# Plot

runs_labels = list(range(1, 13))  # Labels for each run (1 to 12)

# Set figure size
plt.figure(figsize=(7.5, 6.25))

# Define bar properties
blue_bar_width = 0.35 * 0.75  # Width of the blue bars (25% thinner)
orange_bar_width = 0.35 * 1.25  # Width of the orange bars (25% thicker)
gap = 0.1  # Additional space between bars in the same group
x_pos = np.arange(len(runs_labels))  # Positions of bars on the x-axis

# Plot bars for data before outlier removal
plt.bar(
    x_pos,
    mean_values,
    blue_bar_width,
    yerr=std_devs,
    capsize=5,
    label='Before Outlier Removal',
    color='blue'
)

# Plot bars for data after outlier removal
plt.bar(
    x_pos + blue_bar_width + gap,
    mean_values_updated,
    orange_bar_width,
    yerr=std_devs_updated,
    capsize=5,
    label='After Outlier Removal',
    color='orange'
)

# Set x-axis ticks and labels
positions = x_pos + (blue_bar_width + gap + orange_bar_width) / 2  # Center between grouped bars
plt.xticks(positions, runs_labels, fontsize=12)

# Adjust x-axis limits to reduce empty space
plt.xlim(-0.3, len(runs_labels) - 0.2)

# Add vertical red dotted lines to separate groups
plt.axvline(x=3.75, color='red', linestyle='--', linewidth=1.5)  # Between groups 4 and 5
plt.axvline(x=7.75, color='red', linestyle='--', linewidth=1.5)  # Between groups 8 and 9

# Add group labels above respective areas
if test_type == 'compression':
    plt.text(
        1.75, plt.ylim()[1] * 1.000, 'PEEK 5 wt%', fontsize=10, ha='center',
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
    )
    plt.text(
        5.75, plt.ylim()[1] * 1.000, 'PEEK 10 wt%', fontsize=10, ha='center',
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
    )
    plt.text(
        9.75, plt.ylim()[1] * 1.000, 'PEEK 15 wt%', fontsize=10, ha='center',
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
    )
elif test_type == 'bending':
    plt.text(
        1.75, plt.ylim()[1] * 1.000, 'PEEK 5 wt%', fontsize=10, ha='center',
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
    )
    plt.text(
        5.75, plt.ylim()[1] * 1.000, 'PEEK 10 wt%', fontsize=10, ha='center',
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
    )
    plt.text(
        9.75, plt.ylim()[1] * 1.000, 'PEEK 15 wt%', fontsize=10, ha='center',
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
    )
else:
    raise ValueError("Unknown test type. Please specify 'compression' or 'bending'.")

# Add axis labels and legend
plt.xlabel('DoE run no.', fontsize=14)
plt.yticks(fontsize=12)
plt.legend()

# Set y-axis label and limits based on test type
if test_type == 'compression':
    plt.ylabel('Compression Modulus [MPa]', fontsize=14)
    plt.ylim(0, 800)  # Adjust y-axis limits
    plt.savefig('DoE_Cmod_comb.pdf', bbox_inches='tight')  # Save plot as PDF
elif test_type == 'bending':
    plt.ylabel('Bending Modulus [MPa]', fontsize=14)
    plt.ylim(0, 2500)  # Adjust y-axis limits
    plt.savefig('DoE_Bmod_comb.pdf', bbox_inches='tight')  # Save plot as PDF
else:
    raise ValueError("Unknown test type. Please specify 'compression' or 'bending'.")

# Display the plot
plt.show()
