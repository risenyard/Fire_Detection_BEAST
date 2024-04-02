import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import Rbeast as rb
from tqdm import tqdm
import datetime as dt
from datetime import datetime
import os
import sys
# import pycrs
import pyproj
pyproj.datadir.get_data_dir()
pyproj.datadir.set_data_dir(os.path.dirname(sys.argv[0]))
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score
import pickle

# Class for reading MODIS-NDVI time cube and detecting the possible fire
class fire_detector:

    # Initialise the class with environment variables
    def __init__(self, file_path, file_band_names, crs, unmask_value, nan_ratio_threshold, BEAST_belief_threshold, result_directory):
        self.image, self.transform, self.crs_wkt = self.read_geotiff(file_path) # read the geotiff file
        self.crs = crs # set the crs
        self.image = np.where (self.image == unmask_value, np.nan, self.image) # replace the unmask value with nan
        self.band_names = np.load(file_band_names)
        self.time_label = [s.split('_NDVI')[0].replace('_', '-') for s in self.band_names] # get the time label from band names
        self.num_bands, self.height, self.width = self.image.shape # get the number of bands, height, and width
        self.time_series = self.get_time_series(nan_ratio_threshold) # Set the biggest ratio for nan values in a pixel's time series 
        self.BEAST_belief_threshold = BEAST_belief_threshold # Set the belief threshold for the BEAST algorithm
        self.result_directory = result_directory # Set the directory for the result
    
    # Read the geotiff
    @staticmethod
    def read_geotiff(file_path):
        with rasterio.open(file_path) as src:
            image = src.read()
            transform = src.transform
            crs = src.crs
            band_names = src.descriptions
        return image, transform, crs

    # Get the time series of each pixel of the image
    def get_time_series(self, nan_ratio_threshold):
        # Initialize the result list
        result = []
        time_series = []
        # Loop the image to store height, width, and pixel values into time_series list
        for i in range(self.height):
            for j in range(self.width):
                pixel_values = self.image[:, i, j]
                # filter out the pixel with ratio of nan value more than the threshould
                nan_ratio = np.isnan(pixel_values).sum() / len(pixel_values)
                if nan_ratio < nan_ratio_threshold:
                    result.append([i, j, pixel_values])
        # Convert the result list into a pandas dataframe
        time_series = pd.DataFrame(result, columns=["height", "width", "value"])
        time_series = time_series.reset_index(drop=True)
        return time_series

    # Visualize the time series of a pixel
    def visualize_ts(self, index):
        # get the time series of the pixel
        try:
            sequence = self.time_series['value'][index]
        except KeyError:
            print("Error: 'value' key not found in time_series dictionary.")
            return
        except IndexError:
            print("Error: Index out of range.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(sequence, color='red', linestyle='-', label='Interpolated')
        plt.title('NDVI Time Series Plot')
        plt.xlabel('Time')
        plt.ylabel('NDVI')
        plt.xticks(np.arange(0, len(sequence), 100), self.time_label[::100])
        plt.legend()
        # Save the image
        image_path = self.result_directory + f"image_{index}.png"
        plt.savefig(image_path)
        print(f"Time series plot for pixel {index} saved to {image_path}")

    # convert the float to datetime
    @staticmethod
    def float_to_datetime(float_time):
        year = int(float_time)
        year_fraction = float_time - year
        days_in_year = 365.25   # Account for leap years
        days = int(year_fraction * days_in_year)
        start_of_year = datetime(year, 1, 1)
        dt_time = start_of_year + dt.timedelta(days=days)
        dt_time = dt_time.date()
        return dt_time

    # Detect the fire with BEAST at each pixel
    def detect_fire(self):
        # Create an empty list of the same length as time_series
        time_series_length = len(self.time_series['value'])
        Breaks = [None]*time_series_length
        # Get the first year and temportal resoltion for BEAST
        first_year = datetime.strptime(self.time_label[0], '%Y-%m-%d').year
        count = sum(1 for date in self.time_label if datetime.strptime(date, '%Y-%m-%d').year == first_year)
        temporal_resolution = 1 / count
        # Loop each pixel to detect the fire
        for i in tqdm(range(time_series_length), desc="Detecting fire"):  # Use tqdm here
        # get the time series of the pixel
            dt = self.time_series['value'][i]
            # BEAST algorithm
            o = rb.beast(dt, start = first_year, deltat = temporal_resolution, period = '1.0 year', tcp_minmax = [0,1])
            if o.trend.ncpPr[1] > self.BEAST_belief_threshold:
                Breaks[i] = self.float_to_datetime(o.trend.cpCI[1])
        # # Loop each pixel to detect the fire
        # for i in range(time_series_length):
        #     # get the time series of the pixel
        #     dt = self.time_series['value'][i]
        #     # BEAST algorithm
        #     o = rb.beast(dt, start = first_year, deltat = temporal_resolution, period = '1.0 year', tcp_minmax = [0,1])
        #     if o.trend.ncpPr[1] > self.BEAST_belief_threshold:
        #         Breaks[i] = self.float_to_datetime(o.trend.cpCI[1])
        # Write fire time, whether fire exists or not, id into time_series
        time_series_after_detection = self.time_series.copy()
        time_series_after_detection['fire_time'] = Breaks
        time_series_after_detection['fire_exists'] = time_series_after_detection['fire_time'].notnull().astype(int)
        time_series_after_detection['id'] = range(0, len(time_series_after_detection))
        result_table_path = self.result_directory + 'fire.csv'
        time_series_after_detection.to_csv(result_table_path,index=False)
        print("Fire detection result saved to", result_table_path)
        return time_series_after_detection
    
    # save the temporal variable to file
    @staticmethod
    def save_temporal_variable(temporal_variable, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(temporal_variable, f)
        print(f"Temporal variable saved to {file_path}")

    # load the temporal variable from file
    @staticmethod
    def load_temporal_variable(file_path):
        with open(file_path, 'rb') as f:
            temporal_variable = pickle.load(f)
        print(f"Temporal variable loaded from {file_path}")
        return temporal_variable
    
    # result visualization (taking the output of def detct_fire)
    def fire_visualization(self, time_series_after_detection):
        data_with_xy = time_series_after_detection
        image_class = np.full((self.height, self.width),np.nan)
        for i in range(len(data_with_xy)):
            if data_with_xy['fire_exists'][i]==1:
                image_class[data_with_xy['height'][i],data_with_xy['width'][i]] = 1
            else:
                image_class[data_with_xy['height'][i],data_with_xy['width'][i]] = 0

        plt.figure(figsize=(6, 7))
        # Define a colormap with red and green
        cmap = mcolors.ListedColormap(['green', 'red'])
        black_patch = mpatches.Patch(color='red', label='with fire')
        white_patch = mpatches.Patch(color='green', label='without fire')
        # Transform the coordinates
        lon_transformed, lat_transformed = self.coordinate_setting(self.height, self.width, self.transform, self.crs)
        # Display the image in geographic coordinates
        plt.pcolormesh(lon_transformed, lat_transformed, image_class, cmap=cmap)
        # Create a MaxNLocator instance
        locator = ticker.MaxNLocator(nbins=5)
        locator2 = ticker.MaxNLocator(nbins=4)
        # Set x ticks
        plt.gca().xaxis.set_major_locator(locator)
        # Set y ticks
        plt.gca().yaxis.set_major_locator(locator2)
        # Add the legend and title
        plt.legend(handles=[black_patch, white_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Fire Detection in RMNP')
        # Save the image
        result_path = self.result_directory + 'Fire_Detection_Result.png'
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        print(f"Fire detection visualisation saved to {result_path}")
        return image_class

    # result visualization (taking the output og def detct_fire)
    def fire_visualization_by_time(self, time_series_after_detection):
        # Convert 'fire_time' to datetime
        time_series_after_detection['fire_time'] = pd.to_datetime(time_series_after_detection['fire_time'])
        # Filter rows where 'fire_exists' is 1
        tsf = time_series_after_detection[time_series_after_detection['fire_exists'] == 1]

        # Create a new DataFrame with 'fire_time' as the index
        ts_new = tsf.set_index('fire_time')

        # Create a new figure with specified width and height
        plt.figure(figsize=(10,5))
        # Resample by half-year and count the number of each period
        counts_by_half_year = ts_new.resample('6MS').size()
        
        # Generate half-year labels
        half_year_labels = [f"{year}{'A' if month <= 6 else 'B'}" for year, month in zip(counts_by_half_year.index.year, counts_by_half_year.index.month)]
        counts_by_half_year.index = half_year_labels
        counts_by_half_year.plot(kind='bar')

        plt.title('Bar Chart of Fire Time Counts by Half Year')
        plt.xlabel('Half Year (A: first half, B: second half)')
        plt.ylabel('Counts')
        # Prevent x-axis labels from rotating
        plt.xticks(rotation=0)
        # Save the image
        result_path = self.result_directory + 'Fire_Time_Count_by_Half_Year.png'
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        print(f"Fire detection visualisation (by time) saved to {result_path}")
    
    # build coordinate for the visualisation image
    @staticmethod
    def coordinate_setting(height, width, transform, input_EPSG):
        # Create the longitude and latitude arrays
        lon = np.zeros((height, width))
        lat = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                lon[i, j], lat[i, j] = transform * (j, i)
        # Create a transformer from the current CRS to EPSG
        transformer = pyproj.Transformer.from_crs(input_EPSG, 'EPSG:4326', always_xy=True)
        # Transform the coordinates
        lon_transformed, lat_transformed = transformer.transform(lon, lat)
        return lon_transformed, lat_transformed
    
    # export the detection result as raster (taking the output of def fire_visualisation)
    def result_exported_as_raster(self, image_class):
        # Define the output file path
        result_raster_path = self.result_directory + 'Fire_Detection_Result.tif'
        # Expand dimensions
        image_class_3d = np.expand_dims(image_class, axis=0)
        # Open the output file in 'write' mode
        with rasterio.open(result_raster_path, 'w', driver='GTiff', height=self.height, width=self.width, 
                        count=1, dtype=image_class.dtype, crs=self.crs_wkt, transform=self.transform) as dst:
            # Write the data to the file
            dst.write(image_class_3d)
        print("Result Raster image saved to", result_raster_path)
        return result_raster_path

    # result validation (taking the output of def result_exported_as_raster)
    def result_validation(self, result_raster_path, validation_shp_path):    
        true_fire, result_raster, gdf= self.mask_raster(result_raster_path, validation_shp_path)
        true_fire = true_fire[0].flatten()
        result_raster = np.squeeze(result_raster, axis=0)
        index1 = true_fire == 1 
        index0 = true_fire == 0
        indices = np.logical_or(index1, index0)

        # (Part 1) Validation of the result
        # Find the indices where image_class_flat is NaN
        image_flat = result_raster.flatten()
        nan_indices = np.isnan(image_flat)
        # Differentiate the ground truth fire(1) and non-fire(0) pixels
        true_flat = result_raster.flatten()
        true_flat[indices] = 1
        true_flat[~indices] = 0

        # Remove the NaN values from true_flat and image_flat
        true = true_flat[~nan_indices]
        pred = image_flat[~nan_indices]

        # Calculate the confusion matrix and related indices
        cm = confusion_matrix(true, pred)
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)
        kappa = cohen_kappa_score(true, pred)

        # Print results in a consistent format
        print(f"Confusion Matrix:\n{cm}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Kappa Score: {kappa:.2f}")

        # Open a file to write the results
        validation_result_path = self.result_directory + "/Validation_Result/confusion_matrix.txt"
        with open(validation_result_path, 'w') as f:
            # Write results in a consistent format
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f"Accuracy: {accuracy:.2f}\n")
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall: {recall:.2f}\n")
            f.write(f"F1 Score: {f1:.2f}\n")
            f.write(f"Kappa Score: {kappa:.2f}\n")

        # (Part 2) Visualisaion of validation
        gdf = gdf.to_crs('EPSG:4326')
        # Display the image in geographic coordinates
        # Define a colormap with red and green
        cmap = mcolors.ListedColormap(['green', 'red'])
        lon_transformed, lat_transformed = self.coordinate_setting(self.height, self.width, self.transform, self.crs)
        black_patch = mpatches.Patch(color='red', label='with fire')
        white_patch = mpatches.Patch(color='green', label='without fire')
        # Display the image in geographic coordinates
        fig, ax = plt.subplots(figsize=(6, 7))
        plt.pcolormesh(lon_transformed, lat_transformed, result_raster, cmap=cmap)
        gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5)
        # Create a MaxNLocator instance
        locator = ticker.MaxNLocator(nbins=5)
        locator2 = ticker.MaxNLocator(nbins=4)
        # Set x ticks
        plt.gca().xaxis.set_major_locator(locator)
        # Set y ticks
        plt.gca().yaxis.set_major_locator(locator2)
        bound_patch = mpatches.Patch(facecolor='none', edgecolor='black', label='true fire boundary')
        # Add the legend and title
        plt.legend(handles=[black_patch, white_patch, bound_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Fire Detection in RMNP')
        # Save the image
        result_path = self.result_directory + '/Validation_Result/validation_result.png'
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        print(f"Fire detection visualisation saved to {result_path}")

    # mask the raster with shapefile
    @staticmethod
    def mask_raster(result_raster_path, validation_shp_path):
         # Open the raster file
        with rasterio.open(result_raster_path) as src:
            # Convert the GeoDataFrame's geometry to GeoJSON format
            gdf = gpd.read_file(validation_shp_path)
            geoms = gdf.geometry.values.tolist()
            raster_image = src.read()
            #Use rasterio's mask function to create the masked data
            out_image, _ = mask(src, geoms, crop=False, filled=False)
        return out_image, raster_image, gdf
