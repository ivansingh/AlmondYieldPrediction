from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os
from tensorflow.keras.optimizers import Adam

#Take image and convert to 1d np array
def load_and_preprocess(path):
    # Load TIFF image
    image = Image.open('image.tif')
    print(image.mode)
    # Convert to NumPy array
    image_np = np.array(image)

    flat = image_np.flatten()

    return flat  
    # # Convert to TensorFlow tensor
    # image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)

    # tf.reshape(image_tensor, [-1, 1])

    # return image_tensor
    # Now you can use image_tensor with TensorFlow


def normalize_and_histogram(pixel_values):
    """
    Normalize a 1D array of pixel values to [0, 1] and compute a histogram with 10 buckets,
    with each bucket showing the proportion of total values falling into that range.

    Args:
    pixel_values (np.array): A 1D numpy array of pixel values.

    Returns:
    np.array: A 1D array of length 10 containing the histogram proportions.
    """
    # Min-Max Normalization
    min_val = np.min(pixel_values)
    max_val = np.max(pixel_values)
    
    # Handle case where all pixel values are the same
    if max_val == min_val:
        return np.zeros(10)  # Return an array of zeros as histogram

    normalized = (pixel_values - min_val) / (max_val - min_val)

    # Compute the histogram
    # np.histogram returns two arrays: the counts and the bin edges
    histogram, _ = np.histogram(normalized, bins=10, range=(0, 1))

    # Normalize histogram to show proportions
    total_values = len(pixel_values)
    histogram_proportions = histogram / total_values

    return histogram_proportions


# def image_to_histogram(image, bins=10):
    
#     # Calculate histogram for this image
#     histogram = tf.histogram_fixed_width(image, [0, 1], nbins=bins)
#     # Normalize the histogram to sum to 1 (probability distribution)
#     histogram = histogram / tf.reduce_sum(histogram)

#     return histogram

#min max normalize tensor in place
def min_max_normalize(tensor):
    """
    Min-max normalizes a one-dimensional TensorFlow tensor.

    Parameters:
        tensor (tf.Tensor): A one-dimensional tensor.

    Returns:
        tf.Tensor: The normalized tensor with values scaled between 0 and 1.
    """
    # Calculate the minimum and maximum of the tensor
    min_val = tf.reduce_min(tensor)
    max_val = tf.reduce_max(tensor)
    
    # Avoid division by zero in case all elements are the same
    if min_val == max_val:
        return tf.zeros_like(tensor)

    # Normalize the tensor
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def specialized_histogram(image, bins=10):
    """
    Creates a histogram where the first 5 bins cover the lower 10% and the last 5 bins cover the upper 10% of pixel values.

    Parameters:
        image (tf.Tensor): A one-dimensional flattened image tensor.
        bins (int): The total number of bins (default 10).

    Returns:
        tf.Tensor: The normalized histogram as a probability distribution.
    """
    # Calculate the minimum and maximum of the tensor
    min_val = tf.reduce_min(image)
    max_val = tf.reduce_max(image)

    # Calculate the thresholds for 10% and 90% of the range
    lower_10_percent = min_val + 0.1 * (max_val - min_val)
    upper_10_percent = min_val + 0.9 * (max_val - min_val)

    # Set the range for the bins
    first_bins_range = tf.linspace(min_val, lower_10_percent, num=6)
    last_bins_range = tf.linspace(upper_10_percent, max_val, num=6)

    # Concatenate the bin ranges, removing the duplicate middle value
    full_bins_range = tf.concat([first_bins_range[:-1], last_bins_range], axis=0)

    # Calculate histogram using the custom bin edges
    histogram_values = tf.histogram_fixed_width(image, [min_val, max_val], nbins=bins, value_range=[full_bins_range[0], full_bins_range[-1]])

    # Normalize the histogram to sum to 1 (probability distribution)
    normalized_histogram = histogram_values / tf.reduce_sum(histogram_values)

    return normalized_histogram

def compile_LSTM():
    model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(45, 20), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
    optimizer = Adam(learning_rate=0.2)  # You might increase it to 0.01 or more, or use a learning rate scheduler
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()
    return model

def get_counts():
    year_counts = {}
    band_counts = {}
    i = 1
    for filename in os.listdir('SSRNASA'):
        parts = filename.split('_')
        date = parts[4][3:]
        year = date[:4]
        band = parts[3]
        if i == 1:
            print(filename)
            print(date)
            print(year)
            print(band)
            i+=1
        if year in year_counts:
            year_counts[year] += 1
        else:
        # Otherwise, add the year to the dictionary with a count of 1
            year_counts[year] = 1
        if band in band_counts:
            band_counts[band] += 1
        else:
            band_counts[band] = 1
    return year_counts,band_counts

def full_bands():
    year_counts1 = {}
    year_counts2 = {}
    for filename in os.listdir('SSRNASA'):
        parts = filename.split('_')
        date = parts[4][3:]
        year = date[:4]
        band = parts[3]

        if band == "b01":
            if year in year_counts1:
                year_counts1[year] += 1
            else:
                # Otherwise, add the year to the dictionary with a count of 1
                year_counts1[year] = 1

        if band == "b02":
            if year in year_counts2:
                year_counts2[year] += 1
            else:
                # Otherwise, add the year to the dictionary with a count of 1
                year_counts2[year] = 1
    return year_counts1, year_counts2

def organize_files(directory, bands=['b01', 'b02'], files_per_band=45, exclude_years=['2024', '2009']):
    """
    Organize files by year and band, and limit the number of files per band.
    Exclude files that end with '(1)' which are identified as duplicates.
    Additionally, exclude files from specific years.

    Args:
    directory (str): Directory containing the files.
    bands (list): List of bands to include.
    files_per_band (int): Maximum number of files per band per year.
    exclude_years (list): Years to exclude from processing.

    Returns:
    dict: A dictionary mapping bands to dictionaries of years and file lists.
    """
    file_map = {band: {} for band in bands}
    for filename in sorted(os.listdir(directory)):
        if any(band in filename for band in bands) and not filename.endswith('(1).tif'):
            parts = filename.split('_')
            date = parts[4][3:]
            year = date[:4]
            band = parts[3]
            
            # Skip excluded years
            if year in exclude_years:
                continue
            
            if band in bands:
                if year not in file_map[band]:
                    file_map[band][year] = []
                if len(file_map[band][year]) < files_per_band:
                    file_map[band][year].append(os.path.join(directory, filename))
    return file_map

def process_year_data(year_data):
    """
    Process data for a single year: load images, compute histograms, concatenate.
    """
    histograms = []
    for b01_path, b02_path in zip(year_data['b01'], year_data['b02']):
        # Load and process b01 and b02 images, compute histograms
        hist_b01 = normalize_and_histogram(load_and_preprocess(b01_path))
        hist_b02 = normalize_and_histogram(load_and_preprocess(b02_path))
        # Concatenate histograms into a single array
        combined_histogram = np.concatenate([hist_b01, hist_b02])
        histograms.append(combined_histogram)
    return np.array(histograms)

def create_dataset(directory):
    """
    Create the dataset from the files in the given directory.
    """
    # Organize files by year and band
    file_map = organize_files(directory)
    years_sorted = sorted(file_map['b01'].keys())
    dataset = []

    for year in years_sorted:
        if year in file_map['b01'] and year in file_map['b02']:
            # Ensure both bands have sufficient data
            if len(file_map['b01'][year]) == 45 and len(file_map['b02'][year]) == 45:
                year_data = {'b01': file_map['b01'][year], 'b02': file_map['b02'][year]}
                year_histograms = process_year_data(year_data)
                dataset.append(year_histograms)

    return np.array(dataset)

def MAE(network, testing_X, testing_y):

    outputs = network.predict(testing_X)

    errors = tf.abs(outputs - testing_y)

    mae = tf.reduce_mean(errors)
    return mae.numpy()

def main():
    # start = time.time()
    # img = load_and_preprocess('image.tif')
    # histogram = normalize_and_histogram(img)
    # end = time.time()
    # print(end-start)
    # print(histogram)


    # data = np.ones((12, 45, 70))
    # labels = np.ones((12,))
    # data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    # labels_tensor = tf.convert_to_tensor(labels)
    # model = compile_LSTM()
    # history = model.fit(data_tensor, labels_tensor, epochs=100, validation_split=0.2)
    # years, bands = get_counts()
    # print(years)
    # print(bands)
    # year_counts1, year_counts2 = full_bands()

    # print(year_counts1)
    # print(year_counts2)



    data = create_dataset('SSRNASA')
    print(data.shape)
    training_X = data[:-3]
    testing_X = data[-3:]
    data_tensor = tf.convert_to_tensor(training_X, dtype=tf.float32)
    tdata_tensor = tf.convert_to_tensor(testing_X, dtype=tf.float32)

    rlabels = [1790,
1910,
2240,
2490,
2170,
2090,
2200,
2270,
2210,
2000,
2010,
2280,
2310,
2540,
2130]
    labels = rlabels[::-1]


    npLabels = np.array(labels)
    tLabels = npLabels[-3:]
    labels = npLabels[:-3]
    labels_tensor = tf.convert_to_tensor(labels)
    model = compile_LSTM()
    history = model.fit(training_X, labels_tensor, epochs=1200, validation_split=0.20, batch_size=64)
    model.save('model950epochs.h5')
    print(MAE(model, tdata_tensor, tLabels))
    













if __name__ == "__main__":
    main()