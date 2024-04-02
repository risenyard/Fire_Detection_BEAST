# Manual: Forest Fire Detection from GEE MODIS Data

## Quick Introduction

>This project can detect the forest fire from a given study area and time by detecting the changepoint of the time series from the MODIS data. For usage of the project quickly, please check [3. Execution: Run the code](#3-execution-run-the-code).

This manual gives detailed instructions about how to set up the Python environment, understand the code structure and run the code. By default, the manual applies to both Windows and macOS and has been tested in Windows 10 and macOS Ventura platform. Anything about the two platforms needing attention are explained in the body.

The links for all the software and Python package documentation introduced can be found in the end of the manual.

## 1 Starter: Set up Python Environment

### 1.1 Set up the environment in Anaconda and install packages

Anaconda is a popular Python distribution that allows users to create virtual Python environments with their own set of dependencies, packages, and configurations. Setting up Python environment in Anaconda is easy, time-saving and clean. Anaconda is available for all mainstream systems including Windows, macOS, and Linux, so it is convenient to share Python environments across different operating systems.

![Untitled](README%20Images/Untitled.png)

Let’s open Anaconda and create a new environment as Python 3.8, which supports for all libraries we need and is stable for running the code.

![Untitled](README%20Images/Untitled%201.png)

We can then search for the packages in the search bar and install them (this could also be done by "conda install" the package in the terminal). For our project we need to install the libraries as follows:

(image below shows the installation of `geopandas`)

- `io`: provides a set of tools for input and output.
- `os`: provides API to interact with the operating system.
- `sys`: provides tools to interact with the Python interpreter and the Python environment.
- `contextlib`: provides tools to manage the context (in this project, the library is used to skip printing the output on the screen to save resources).
- `json`: handles json format data in the Python project (here, it reads all the input data from json file).
- `tqdm`: visualizes the progress of code execution (fire detection) with progress bar.
- `ee`: python API of Google Earth Engine to download, process and visualize the Earth Observation data from Google Earth.
- `geemap`: creates the interactive map and performs spatial analysis in Python.
- `pandas`: analyses and manipulates for tabular and time-series data.
- `geopandas`: extends `Pandas`, with additional geospatial functionality.
- `numpy`: performs scientific computing that supports large, multidimensional arrays and matrices, and mathematical functions.
- `matplotlib`: visualizes the data by creating a variety of plots, graphs, and charts.
- `rasterio`: handles raster data including GeoTIFF, Jpeg and Png.
- `pyproj`: converts the geographic coordinates and performs projections.
- `pycrs`: analyses, creates and manipulates geographic coordinate system.
- `Rbeast`: conducts time series decomposition and Bayesian change point detection. It is used for fire detection in the Python project.
- `datetime`: handles date and time values with standardized format and functions.
- `sklearn`: performs simple machine learning tasks with easy and clear APIs.

![Untitled](README%20Images/Untitled%202.png)

### 1.2 Jupyter Notebook (ipynb.) / Python script (py.) in Visual Studio Code

After we set up the environment, the next step is to understand the format of scripts. All the coding were done in VS Code. VS Code is a free and open-source code editor developed by Microsoft with customizable interface, and vast library of extensions. 

VS Code supports multiple programming languages and provides features like debugging, Git integration, and intelligent code completion. By simply installing extension of Python, we can edit Python script (py.) and Jupyter Notebook (ipynb.) in VS Code.

![Untitled](README%20Images/Untitled%203.png)

A Python script is a plain text file with a “.py” extension that contains Python code. It can be run on any system with Python installed by invoking the Python interpreter and specifying the path to the script. Python scripts can automate tasks, manipulate data, and build applications. Jupyter Notebook is a web-based interactive platform with the “.ipynb” extension. It contains live Python code, visualizations, and narrative text. It provides a user-friendly interface for data exploration, and prototyping. We can run the code by blocks and get real-time results. The codes of the projects were firstly done with Jupyter Notebook (ipynb.) for better control and later converted into Python script (py.) for easier execution.

To deal with Jupyter notebook, we switch the environment to what we have created and open VS Code in Anaconda.  After opening, we could find the VS code was running with the Python environment we created. (See top right corner).

![Untitled](README%20Images/Untitled%204.png)

![Untitled](README%20Images/Untitled%205.png)

We can run the codes of Jupyter notebook by tapping “Run” button beside each block, and we will get the result of this block beneath it. We can also tap “Run All” button in the above panel and get the results of all blocks.

![Untitled](README%20Images/Untitled%206.png)

For Python script, we simply tap “Run Python File” and the code will be run from the start to the end. The output information can be checked in the Terminal. Or we can execute the script directly in the terminal by first `conda activate 'Python 38'` (activating the python environment) and then `python path_of_the_script.py`.

![Untitled](README%20Images/Untitled%207.png)



### 1.3 Do version control using Git and push the project to GitHub

Version control is essential as it allows developers to track changes, revert to previous versions, and maintain code integrity. Git is an open-source version control system widely used to manage changes in code, and GitHub is a web-based hosting service that provides a platform for Git repositories. VS Code supports the user to build and develop the repository for version control after installing the Git and pushes the repository to GitHub. The manual will also introduce how to the version control via Git for the project.

First, open source control and add the repository from the local directory.

![Untitled](README%20Images/Untitled%208.png)

Next, after every change of our code in the directory, we get notification from the source control panel. We can check the changes we made in the panel. If it is fine, we can stage the changes, add commitment message and commit them.

![Untitled](README%20Images/Untitled%209.png)

![Untitled](README%20Images/Untitled%2010.png)

If it is the first time of commitment, we can publish branch, log in our GitHub account and publish the repository to GitHub. After that, we can also synchronize the changes to the repository in GitHub every time we commit.

![Untitled](README%20Images/Untitled%2011.png)

By doing this, we will have local and remote (in GitHub) repositories that contain our project and track the changes of the codes.

## 2 Explanation: Code structure and functions

- `GEE_Processor.py`: data download and pre-process module. It connects Google Earth Engine, downloads MODIS file, performs clipping and cloud masking, and exports the image.
- `Fire_Detector.py`: fire detection module. It re-structures the standard MODIS data into a time-series list, detects the pixel of fire based on the generated time-series list with BEAST algorithm, visualizes the results and validate the results with true fire boundary.
- `main.py`: **execution** file of the project to refer different modules to conduct the workflow.
- `main.ipynb`: **execution** file in the form of Jupyter Notebook to provide interactive working experiences. This has the same functionality as `main.py`.
- `input_parameter.json`: json file to set all parameters ahead flexibly to avoid any hard-coded vars.
- `Boundary_File`: provides boundary files for research area and true fire for validation.
- `MODIS_File`: directory to save the MODIS images and their band names downloaded from `GEE_Processor.py` and needed for `Fire_Detector.py`.
- `Fire_Detections_Result`: directory to save the image and raster of the fire detection results and validation results as well.


### 2.1 GEE_Processor.py

This is the first module utilized by the project. It connects Google Earth Engine, downloads MODIS file, performs clipping and cloud masking, and exports the image. It contains a class to handle all workflow with following functions.

- `_init_`: authenticates and initializes the connection to Google Earth Engine, gets land cover data and the MODIS data, mask the MODIS data by study area and land cover type, filter the MODIS data by time and adds NDVI band. 
- `maskStudyArea`: mask the MODIS File with study area and forest land cover data.
- `addVariables`: add NDVI band to the image.
- `bitwiseExtract`:extract the binary information from the quality band of MODIS file.
- `maskClouds`: mask clouds in the MODIS data.
- `NDVI_Exract_Stack`: extract NDVI from all bands at each timestep and make all extracted NDVI as bands in stacked image at the time series.
- `export_image`: download the image (with its bandnames) to the local directory.
- `geemap_export`: export the final dataset to a geemap for visualization at html page.

The suggested way to use this module is to initialize the class first and refer `NDVI_Exract_Stack` and `export_image` to create and export dedicated input for fire detection.

### 2.2 Fire_Detector.py

This is the second module utilized by the project. It re-structures the standard MODIS data into a time-series list, detects the pixel of fire based on the generated time-series list with BEAST algorithm, visualizes the results and validate the results with true fire boundary. It contains a class to handle all workflow with following functions.

- `_init_`: the class initializes by reading the MODIS data and bandnames from Google Earth Engine first, creates the time-seires lists from them, and sets up the threshold for fire detection algorithms.
- `read_geotiff`: read the geotiff data from file.
- `get_time_series`: get the time series of each pixel of the image by resorting the input MODIS NDVI stack and make them into a list.
- `visualize_ts`: visualize the time series of a pixel.
- `float_to_datetime`: convert the float to datetime format.
- `detect_fire`: detect the fire with BEAST at each pixel.
- `save_temporal_variable`: save the temporal variable to file.
- `load_temporal_variable`: load the temporal variable from file.
- `fire_visualization`: visualizes the results of fire detection by showing the predicted fire areas on the image. (taking the output of `detct_fire`)
- `fire_visualization_by_time`: visualizes the results of fire detection by distribution of fire time. (taking the output of `detct_fire`)
- `coordinate_setting`: build coordinate for the visualization image.
- `result_exported_as_raster`: export the detection result as raster (taking the output of `fire_visualisation`)
- `result_validation`: validate the detection result with input true fire boundaries and generate confusion matrix and validation map,(taking the output of `result_exported_as_raster`)
- `mask_raster`: mask the raster with shapefile

The suggested way to use this module is to initialize the class first and refer `detect_fire` to perform fire detection. Output of `detect_fire` can be stored at local file by `save_temporal_variable` to make rerunning the program easier. Output of `detect_fire` can be input for `fire_visualization` to get the result image, which is input for `result_exported_as_raster`. The output raster is partly input for validation function `result_validation`.

### 2.3 input.json

The file sets up all parameters ahead avoid any hard-coded vars. All modifications about the input parameters only need to be done here.

- `gee_processor`: this chapter includes all input for getting data in `Gee_Processor.py`:
  - `study_area_shp_path`: path to shapefile of study area
  - `start_date`: start date of the MODIS data
  - `end_date`: end date of the MODIS data
  - `land_cover_data`: name of the land cover data to be downloaded in Google Earth Engine.
  - `remote_sensing_data`: name of the MODIS data to be downloaded in Google.

- `export_image`: this chapter declares the parameter for downloading images from GEE in `Gee_Processor.py` and reading downloaded images in `Fire_Detector.py`.
  - `image_path`: path to save the image
  - `image_band_name`: path to save the band names of the image
  - `scale`: scale of the image
  - `crs`: coordinate reference system of the image (number of EPSG)
  - `unmask_value`: values to be masked as nan
  - `geemap_html`: path to save the html page of the exported geemap
- `fire_detector`: this chapter includes parameters for fire detection, result output and validation.
  - `nan_ratio_threshold`: when the ratio of nan values at one time series is above this threshold, the, get rid of this series to make detection more meaningful.
  - `BEAST_belief_threshold`: the confidence threshold for BEAST algorithm, above which the changepoint is considered valid.
  - `result_directory`: directory to contain output files.
  - `validation_shp_path`: path to the true fire boundary shapefiles.

## 3 Execution: Run the code

>**Test Environment**
>
>*Python environment: Python 3.8*
>
>*Libraries: declared in [1.1 Set up the environment in Anaconda and install packages](#11-set-up-the-environment-in-anaconda-and-install-packages)*
>
>*Input Data: all parameters declared in [2.3 input.json](#23-inputjson)*
>
>*Output Data: fire detection and its validation visualization and values under directory `Fire_Detection_Result`*

Navigate to the directory where Fire_Detector.py is located.
Run the script using the following command:
Functionality
The script works by processing the input from a camera feed or a video file. It uses image processing techniques to identify potential fire in the frames.

Troubleshooting
If you encounter any issues while running the script, ensure that:

You have the correct version of Python installed.
All the required libraries are installed and up-to-date.
The script has the necessary permissions to access the camera feed or video file.
Support
For further assistance, please refer to the documentation or contact the developer.

Please note that this is a general manual. For a more detailed manual, I would need more specific information about the functions and methods used in the Fire_Detector.py script.
#### 3.1 Run with main.py



```python

```



#### 3.2 Run with main.ipynb

The figure set by `figure()` is the base for plotting with `Matplotlib`. On top of the figure, we need to set an axis object accommodating the map. As we create global map, we can use `set_global()` to set the extent of the axis to be global. The linewidth of axis is set to 1 and colour to grey.

```python
################# 2 canvas settings ###########################

# establish a canvas
fig = plt.figure(figsize=(10, 5))
# set up and ax object
ax = fig.add_subplot(1, 1, 1, projection=crs)

# Set the extent and crs of the map
ax.set_global()
# Set parameters of the ax
for spine in ax.spines.values():
    spine.set_linewidth(1)
    spine.set_color('grey')
```



#### 3.3 Classify the spatial data and plot the result

Before mapping, we can classify the data to be 5 classes. We basically use Quantiles method to do it, as every classes contain equal number of countries. `mapclassify` provides `Quantiles` function to conduct it. After that, we can check the breaks of the classification. And then we manually adjust the break values to them more neat. Then we redo the classification with custom breaks. `UserDefined` function of `mapclassify` can be applied. We then  get an classifier.  `classifier.yb` can extract  an array containing the class labels for each country in the `data['GDP per capita']` array based on the breaks
defined earlier.

```python
################ 3 classification and plotting ################

# classify the data to be 5 classes using breaks inspired by Quantiles
class_number = 5
breaks = [2000, 5000, 10000, 30000, 200000] # the breaks were inspired by results of Quantiles
#classifier = mc.Quantiles(data['GDP per capita'], k=class_number) # this script was run first to get the raw breaks
classifier = mc.UserDefined(data['GDP per capita'], breaks)
classifications = np.array(classifier.yb)
```

Before plotting the classification, we set the colormap to be ‘RdYnGn’. Lower value is more red and higher one is more green. Then we have a loop to iterate over each class in the classification and plots the corresponding data on the map. For each class `i`, it creates a subset of the Dataframe containing only the rows where `classifications == i` . It then plots the geometries  from this subset.

```python
# plot the classification
cmap = plt.get_cmap('RdYlGn', class_number) #set the colormap
for i in range(class_number):
    subset = data[classifications == i]
    ax.add_geometries(subset['geometry'], crs, facecolor=cmap(i), edgecolor='black', linewidth=0.2)
```

Additionally, we can add the ocean and lakes features from `Cartopy.feature` , which will get the data from Natural Earth.

```python
# add the ocean and lakes
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES)
```



#### 3.4 Design map element

All the map elements can be designed based on axis. For the title, font name is Times New Roman and font size is 16. It is by default on the center of the map.

```python
############### 4 map layout setting ############################

#set title
ax.set_title("GDP per capita in 2019 for all countries")
ax.title.set_fontname('times new roman')
ax.title.set_fontsize(16)
```

For legend, we can build 5 rectangles icons with the different colors in the colormap. as we make custom breaks, it is also suitable to make custom labels beside the rectangles. And the position of legend is custom to be in the place near to left bottom corner of the map.

```python
# set the legend
# create a list of color labels for the legend (NB: this is only fot the case of 5 classes)
color_labels = ['< 2000', '2000-5000', '5000-10000', '10000-30000', '> 30000']
# create a list of colored rectangles to use for the legend
rectangles = [Rectangle((0, 0), 1, 1, fc=cmap(i)) for i in range(class_number)]
# create the legend with the colored rectangles
legend = ax.legend(rectangles, color_labels, title='GDP per capita ($)', bbox_to_anchor=(0.25,0.51), 
fontsize=8, prop='times new roman', frameon=True, facecolor='white', edgecolor='grey', fancybox=False)
legend.get_title().set_fontname('times new roman')
legend.get_title().set_fontsize(10)
```

The gridlines are set to be dash lines with width of 0.25 and colour of blue. We have title on the top so we disable the labels in the top.

```python
# set the gridlines
gl=ax.gridlines(draw_labels=True,linestyle=":",linewidth=0.25,color='b')
gl.top_labels=False                                                   
gl.xlabel_style={'size':8,'fontproperties':'times new roman'}                          
gl.ylabel_style={'size':8,'fontproperties':'times new roman'}
```

Finally, we create a text box containing all the metadata. And the position is also custom to be in the place near the bottom centre. 

```python
# set the metadata
# Create a text box
text_str = 'Made by: Senyang Li | Environment: Python 3.8.16\nSource: Natural Earth | CRS: Eckert IV'
text_box = ax.text(x=0.543, y=0.006, s=text_str,transform=ax.transAxes,
bbox=dict(facecolor='none', edgecolor='none', pad=5))
text_box.set_fontproperties('times new roman')
text_box.set_fontsize(5)
```



#### 3.5 Export the map

We save the figure to be a png. with resolution of 300m and trim the unnecessary fringe. The figure is also shown after running the script by using `plt.show()`.

```python
################# 5 map export and show ###########################

# export the map
plt.savefig('Task-2-Result (by Python).png', dpi=300, bbox_inches='tight')
# show the map
plt.show()
```

![Task-2-Result (by Python).png](README%20Images/Task-2-Result_(by_Python).png)



## Useful Links

### Software Download

Anaconda: [https://www.anaconda.com/download/](https://www.anaconda.com/download/)

VS Code: [https://code.visualstudio.com/](https://code.visualstudio.com/)

Git: [https://git-scm.com/](https://git-scm.com/)

### Python Library Documentation

`io`: [https://docs.python.org/3/library/io.html](https://docs.python.org/3/library/io.html)

`os`: [https://docs.python.org/3/library/os.html](https://docs.python.org/3/library/os.html)

`sys`: [https://docs.python.org/3/library/sys.html](https://docs.python.org/3/library/sys.html)

`contextlib`: [https://docs.python.org/3/library/contextlib.html](https://docs.python.org/3/library/contextlib.html)

`json`: [https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html)

`tqdm`: [https://tqdm.github.io/](https://tqdm.github.io/)

`ee`: [https://developers.google.com/earth-engine/guides/python_install](https://developers.google.com/earth-engine/guides/python_install)

`geemap`: [https://geemap.org/](https://geemap.org/)

pandas: [https://pandas.pydata.org/docs/user_guide/index.html](https://pandas.pydata.org/docs/user_guide/index.html)

`geopandas`: [https://geopandas.org/en/stable/docs.html](https://geopandas.org/en/stable/docs.html)

`numpy`: [https://numpy.org/doc/](https://numpy.org/doc/)

`matplotlib`: [https://matplotlib.org/stable/index.html](https://matplotlib.org/stable/index.html)

`rasterio`: [https://rasterio.readthedocs.io/en/stable/](https://rasterio.readthedocs.io/en/stable/)

`pyproj`: [https://pyproj4.github.io/pyproj/stable/](https://pyproj4.github.io/pyproj/stable/)

`pycrs`: [https://github.com/karimbahgat/PyCRS] (https://github.com/karimbahgat/PyCRS)

`Rbeast`: [https://github.com/zhaokg/Rbeast](https://github.com/zhaokg/Rbeast)

`datetime`: [https://docs.python.org/3/library/datetime.html](https://docs.python.org/3/library/datetime.html)

`sklearn`: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)