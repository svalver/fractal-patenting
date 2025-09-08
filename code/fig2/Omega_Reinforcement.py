import numpy as np
from collections import defaultdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


def plot_grouped_likelihoods(bin_likelihoods, bin_size, step_size, group_size=10):
    global dataset, Length, style, data_all
    """
    Plot the mean and standard deviation of the likelihoods by grouping them based on proximity
    in sliding windows.

    Args:
    - bin_likelihoods: List of dictionaries containing likelihoods for each sliding window.
    - bin_size: The size of each sliding window.
    - step_size: The step size between sliding windows.
    - group_size: Number of consecutive sliding windows to group together.
    """

    num_bins = len(bin_likelihoods)
    colors = plt.cm.jet(np.linspace(0, 1, (1+num_bins // group_size)))

    # Group the sliding windows based on the group size
    grouped_likelihoods = []
    for i in range(0, num_bins, group_size):
        print(dataset, num_bins, i, group_size)
        group = bin_likelihoods[i:i + group_size]

        # Initialize a dictionary to store grouped likelihoods
        aggregated_group = defaultdict(list)

        for likelihood in group:
            likelihood.pop(0, None)  # Remove frequency class 0 if present
            for freq_class, value in likelihood.items():
                aggregated_group[freq_class].append(value)
                if i == 14*group_size:
                    d_ = pd.DataFrame([[dataset, freq_class, value]], columns=['Dataset', 'Firm Size', 'Productivity'])
                    data_all = pd.concat((data_all, d_))

        # For each frequency class, calculate mean and standard deviation
        group_stats = {}
        for freq_class, values in aggregated_group.items():
            '''if freq_class > 0:
                values = [vi / freq_class for vi in values]'''
            mean_val = np.mean(values)
            std_dev_val = np.std(values)
            if mean_val != 0:  # Only keep non-zero mean likelihoods
                group_stats[freq_class] = (mean_val, std_dev_val)

        grouped_likelihoods.append(group_stats)

    # Plotting
    plt.figure(figsize=(7, 4))
    for idx, group_stats in enumerate(grouped_likelihoods):
        sorted_list = sorted(group_stats.items())
        sorted_dict = {}
        for key, (mean, std_dev) in sorted_list:
            sorted_dict[key] = (mean, std_dev)

        x = list(sorted_dict.keys())
        y = [v[0] for v in sorted_dict.values()]  # Mean values
        y_err = [v[1] for v in sorted_dict.values()]  # Standard deviations

        #plt.errorbar(x, y, yerr=y_err, label=f'Group {idx + 1}', fmt='-o', color=colors[idx])
        plt.errorbar(x, y, yerr=y_err, fmt='-o', markersize=5, color=colors[idx])


    # Configure plot aesthetics
    plt.xscale('log')  # Uncomment if you want log scale on x-axis
    plt.yscale('log')  # Uncomment if you want log scale on y-axis
    plt.xlabel('Firm Size ($N_i$)')
    plt.ylabel('Average productivity ($\Delta N_i$/ \Delta N)')
    #plt.legend()
    plt.title('Patenting activity by firm size in '+ dataset + ' (N='+str(Length)+')')
    plt.grid(True)
    #plt.show()
    plt.savefig('Productivity_' + dataset + '_' + style + '.pdf')
    plt.close()


def calculate_sliding_likelihoods(coordinate_sequence, bin_size=200, step_size=25):
    """
    Calculate the likelihood of observing a location in a given sliding window (bin)
    based on how often it has appeared before that window.

    Args:
    - coordinate_sequence: List of coordinates (locations).
    - bin_size: Size of each sliding window (default is 50).
    - step_size: The number of units to move the window forward each time (default is 10).

    Returns:
    - sliding_likelihoods: A list of dictionaries, where each dictionary represents
      the likelihood for each sliding window with respect to different frequency classes.
    """

    # Step 1: Organize coordinates into sliding windows
    sliding_likelihoods = []  # To store likelihoods for each sliding window
    num_steps = (len(coordinate_sequence) - bin_size) // step_size + 1

    for step in range(num_steps):
        start_index = step * step_size
        end_index = start_index + bin_size
        current_window = coordinate_sequence[start_index:end_index]

        if len(current_window) < bin_size:
            break  # Stop if the window is incomplete (this handles edge cases)

        # Initialize variables to track frequency and likelihood
        location_frequencies = defaultdict(int)  # To count how many times each location appeared before this window

        # Calculate prior frequencies up to the start of the current window
        prior_sequence = coordinate_sequence[:start_index]
        for loc in prior_sequence:
            location_frequencies[loc] += 1

        # Calculate prior class counts (same structure as in the original code)
        frequency_prior_counts = defaultdict(int)
        frequency_prior_counts[0] = 0
        frequency_prior_counts[1] = 0
        frequency_prior_counts[2] = 0
        frequency_prior_counts[4] = 0
        frequency_prior_counts[8] = 0
        frequency_prior_counts[16] = 0
        frequency_prior_counts[32] = 0
        frequency_prior_counts[64] = 0
        frequency_prior_counts[128] = 0
        frequency_prior_counts[256] = 0
        frequency_prior_counts[512] = 0

        for loc_freq in location_frequencies.values():
            if loc_freq == 1:
                frequency_prior_counts[1] += 1
            elif loc_freq == 2:
                frequency_prior_counts[2] += 1
            elif 3 <= loc_freq <= 4:
                frequency_prior_counts[4] += 1
            elif 5 <= loc_freq <= 8:
                frequency_prior_counts[8] += 1
            elif 9 <= loc_freq <= 16:
                frequency_prior_counts[16] += 1
            elif 17 <= loc_freq <= 32:
                frequency_prior_counts[32] += 1
            elif 33 <= loc_freq <= 64:
                frequency_prior_counts[64] += 1
            elif 65 <= loc_freq <= 128:
                frequency_prior_counts[128] += 1
            elif 129 <= loc_freq <= 256:
                frequency_prior_counts[256] += 1
            else:
                frequency_prior_counts[512] += 1

        # Initialize the frequency class counts for the current window
        frequency_class_counts = defaultdict(int)
        frequency_class_counts[0] = 0
        frequency_class_counts[1] = 0
        frequency_class_counts[2] = 0
        frequency_class_counts[4] = 0
        frequency_class_counts[8] = 0
        frequency_class_counts[16] = 0
        frequency_class_counts[32] = 0
        frequency_class_counts[64] = 0
        frequency_class_counts[128] = 0
        frequency_class_counts[256] = 0
        frequency_class_counts[512] = 0

        # Count how many times each location appeared before this window
        for loc in current_window:
            prev_count = location_frequencies[loc]

            # Classify the location based on its previous count using powers of 2
            if prev_count == 1:
                frequency_class_counts[1] += 1
            elif prev_count == 2:
                frequency_class_counts[2] += 1
            elif 3 <= prev_count <= 4:
                frequency_class_counts[4] += 1
            elif 5 <= prev_count <= 8:
                frequency_class_counts[8] += 1
            elif 9 <= prev_count <= 16:
                frequency_class_counts[16] += 1
            elif 17 <= prev_count <= 32:
                frequency_class_counts[32] += 1
            elif 33 <= prev_count <= 64:
                frequency_class_counts[64] += 1
            elif 65 <= prev_count <= 128:
                frequency_class_counts[128] += 1
            elif 129 <= prev_count <= 256:
                frequency_class_counts[256] += 1
            elif prev_count > 256:
                frequency_class_counts[512] += 1
            else:
                frequency_class_counts[0] += 1

        # Step 4: Calculate likelihoods based on the frequency classes
        likelihoods = {}
        for frequency_class, count in frequency_class_counts.items():
            likelihoods[frequency_class] = count / ((frequency_prior_counts[frequency_class] + 1) * bin_size)

        # Store the likelihoods for this sliding window
        sliding_likelihoods.append(likelihoods)

    return sliding_likelihoods


def calculate_sliding_likelihoods_linear(coordinate_sequence, bin_size=50, step_size=5, max_frequency_class=300):
    """
    Calculate the likelihood of observing a location in a given sliding window (bin)
    based on how often it has appeared before that window, using linear frequency classes.

    Args:
    - coordinate_sequence: List of coordinates (locations).
    - bin_size: Size of each sliding window (default is 50).
    - step_size: The number of units to move the window forward each time (default is 5).
    - max_frequency_class: Maximum frequency class for linear categories (default is 20).

    Returns:
    - sliding_likelihoods: A list of dictionaries, where each dictionary represents
      the likelihood for each sliding window with respect to different linear frequency classes.
    """

    # Step 1: Organize coordinates into sliding windows
    sliding_likelihoods = []  # To store likelihoods for each sliding window
    num_steps = (len(coordinate_sequence) - bin_size) // step_size + 1

    for step in range(num_steps):
        start_index = step * step_size
        end_index = start_index + bin_size
        current_window = coordinate_sequence[start_index:end_index]

        if len(current_window) < bin_size:
            break  # Stop if the window is incomplete (this handles edge cases)

        # Initialize variables to track frequency and likelihood
        location_frequencies = defaultdict(int)  # To count how many times each location appeared before this window

        # Calculate prior frequencies up to the start of the current window
        prior_sequence = coordinate_sequence[:start_index]
        for loc in prior_sequence:
            location_frequencies[loc] += 1

        # Initialize the frequency prior counts (linear from 0 up to max_frequency_class)
        frequency_prior_counts = defaultdict(int)
        for i in range(max_frequency_class + 1):
            frequency_prior_counts[i] = 0

        # Count how many times each frequency occurred before this window
        for loc_freq in location_frequencies.values():
            if loc_freq <= max_frequency_class:
                frequency_prior_counts[loc_freq] += 1
            else:
                frequency_prior_counts[max_frequency_class] += 1  # All larger values are lumped into the last class

        # Initialize the frequency class counts for the current window (linear classes)
        frequency_class_counts = defaultdict(int)
        for i in range(max_frequency_class + 1):
            frequency_class_counts[i] = 0

        # Count how many times each location appeared before this window
        for loc in current_window:
            prev_count = location_frequencies[loc]

            # Classify the location based on its previous count in linear categories
            if prev_count <= max_frequency_class:
                frequency_class_counts[prev_count] += 1
            else:
                frequency_class_counts[max_frequency_class] += 1  # All larger values are lumped into the last class

        # Step 4: Calculate likelihoods based on the frequency classes
        likelihoods = {}
        for frequency_class, count in frequency_class_counts.items():
            likelihoods[frequency_class] = count / ((frequency_prior_counts[frequency_class] + 1) * bin_size)

        # Store the likelihoods for this sliding window
        sliding_likelihoods.append(likelihoods)

    return sliding_likelihoods


def calculate_bin_likelihoods(coordinate_sequence, bin_size=50):
    """
    Calculate the likelihood of observing a location in a given bin based on
    how often it has appeared before that bin.

    Args:
    - coordinate_sequence: List of coordinates (locations).
    - bin_size: Size of each bin (default is 20).

    Returns:
    - bin_likelihoods: A list of dictionaries, where each dictionary represents
      the likelihood for each bin with respect to different frequency classes.
    """

    # Step 1: Organize coordinates into bins
    num_bins = len(coordinate_sequence) // bin_size
    bins = []
    for i in range(num_bins-1):
        slice = coordinate_sequence[(i + 1) * bin_size:(i + 2) * bin_size]
        bins.append(slice)


    # Iterate through each bin
    bin_likelihoods = []  # To store likelihoods for each bin
    for i, current_bin in enumerate(bins):
        # Initialize variables to track frequency and likelihood
        location_frequencies = defaultdict(int)  # To count how many times each location appears before a bin

        # Calculate prior frequencies
        prior_sequence = coordinate_sequence[0:(i + 1)*bin_size]
        for loc in prior_sequence:
            # Update the frequency count for this location
            if loc in location_frequencies:
                location_frequencies[loc] += 1
            else:
                location_frequencies[loc] = 1

        # Calculate prior class counts
        frequency_prior_counts = {}
        frequency_prior_counts[0] = 0
        frequency_prior_counts[1] = 0
        frequency_prior_counts[2] = 0
        frequency_prior_counts[4] = 0
        frequency_prior_counts[8] = 0
        frequency_prior_counts[16] = 0
        frequency_prior_counts[32] = 0
        frequency_prior_counts[64] = 0
        frequency_prior_counts[128] = 0
        frequency_prior_counts[256] = 0
        for loc in location_frequencies.values():
            if loc == 1:
                frequency_prior_counts[1] += 1
            elif loc == 2:
                frequency_prior_counts[2] += 1
            elif 3 <= loc <= 4:
                frequency_prior_counts[4] += 1
            elif 5 <= loc <= 8:
                frequency_prior_counts[8] += 1
            elif 9 <= loc <= 16:
                frequency_prior_counts[16] += 1
            elif 17 <= loc <= 32:
                frequency_prior_counts[32] += 1
            elif 33 <= loc <= 64:
                frequency_prior_counts[64] += 1
            elif 65 <= loc <= 128:
                frequency_prior_counts[128] += 1
            else:
                frequency_prior_counts[256] += 1  # For more than 64 occurrences


        frequency_class_counts = defaultdict(int)  # Track how many locations fall into each frequency class

        # Count how many times each location appeared before this bin
        frequency_class_counts[0] = 0
        frequency_class_counts[1] = 0
        frequency_class_counts[2] = 0
        frequency_class_counts[4] = 0
        frequency_class_counts[8] = 0
        frequency_class_counts[16] = 0
        frequency_class_counts[32] = 0
        frequency_class_counts[64] = 0
        frequency_class_counts[128] = 0
        frequency_class_counts[256] = 0
        for loc in current_bin:
            if loc in location_frequencies:
                prev_count = location_frequencies[loc]

                # Classify the location based on its previous count using powers of 2
                if prev_count == 1:
                    frequency_class_counts[1] += 1
                elif prev_count == 2:
                    frequency_class_counts[2] += 1
                elif 3 <= prev_count <= 4:
                    frequency_class_counts[4] += 1
                elif 5 <= prev_count <= 8:
                    frequency_class_counts[8] += 1
                elif 9 <= prev_count <= 16:
                    frequency_class_counts[16] += 1
                elif 17 <= prev_count <= 32:
                    frequency_class_counts[32] += 1
                elif 33 <= prev_count <= 64:
                    frequency_class_counts[64] += 1
                elif 65 <= prev_count <= 128:
                    frequency_class_counts[128] += 1
                else:
                    frequency_class_counts[256] += 1  # For more than 64 occurrences

            else:
                frequency_class_counts[0] += 1

        # Step 4: Calculate likelihoods based on the frequency classes
        total_locations_in_bin = len(current_bin)
        likelihoods = {}

        # Calculate likelihood for each frequency class
        for frequency_class, count in frequency_class_counts.items():
            likelihoods[frequency_class] = count / ((frequency_prior_counts[frequency_class] + 1) * bin_size)
            print(frequency_class, count, frequency_prior_counts[frequency_class])

        # Store the likelihoods for this bin
        bin_likelihoods.append(likelihoods)

    return bin_likelihoods


DATAFOLDER = "../../data/"

# Example usage:
data_all = pd.DataFrame(columns=['Dataset','N','Omega'])
for dataset in ['chips', 'smartphone', 'plugin2', 'fracking', 'ml', 'videogames', 'apps2', 'biotech']:
    #Aggregate Data
    df = pd.read_csv(DATAFOLDER + 'full_' + dataset + '.csv', encoding="latin-1")
    #print(dataset, len(df))

    #Drop useless columns
    df = df.drop(['patent_id', 'filing_date', 'cpc','type', 'title', 'assignee', 'person', 'first',
             'second', 'inventors', 'loc', 'cites', 'birth', 'death'], axis=1)
    df.reset_index(inplace=True)

    #Normalize naming
    for i in range(len(df)):
        if df.loc[i, 'country'] == 'USA':
            df.loc[i, 'country'] = 'US'

    #Drop non US data
    df_country = df[df['country'] == 'US']
    df_country.reset_index(drop=True, inplace=True)
    df_country.loc[:,'city'] = ''

    #Reformat location and date
    for i in range(len(df_country)):
        df_country.loc[i, 'city'] = str(df_country.loc[i, 'longitude']) + ', ' + str(df_country.loc[i, 'latitude'])
        df_country.loc[i, 'date'] = datetime.strptime(df_country.loc[i, 'date'], '%Y-%m-%d')

    #print(df_country.head())

    #Sort Values
    df_c = df_country.sort_values(by=['date'])
    df_c.reset_index(inplace=True)
    #dates = list(df_c['date'])
    #print(dates)

    #Obtain sequence of empirical patents
    sequence = list(df_c.loc[:, 'city'])
    Length = len(sequence)
    print(dataset, len(sequence))

    bin_size=50
    step_size = 20
    if 1:
        style = 'linear'
        max_f = 512
        num_bins = (len(sequence) - bin_size) // step_size + 1
        bin_likelihoods = calculate_sliding_likelihoods_linear(sequence, bin_size=bin_size,
                                                               step_size=step_size, max_frequency_class=max_f)
        classes = list(np.arange(0,max_f,1))
    else:
        style = 'logarithmic'
        num_bins = (len(sequence) - bin_size) // step_size + 1
        bin_likelihoods = calculate_sliding_likelihoods(sequence, bin_size=bin_size, step_size=step_size)
        classes = [0,1,2,4,8,16,32,64,128,256,512]

    # Output size-class contribution over time
    class_ = 0
    x_ = 1
    for i, likelihood in enumerate(bin_likelihoods):
        if likelihood[class_] > 0:
            df_ = pd.DataFrame([[dataset, x_, likelihood[class_], len(set(sequence[:x_]))]], columns=['Dataset', 'N', 'Omega','D'])
            data_all = pd.concat((data_all, df_))
        x_ += step_size




data_all.to_csv('Omega_D_'+style[:3]+'.csv')