import ee
import geemap
import sys
import numpy as np

class gee_processor: 
    
    # study_area_shp_path: Path to the shapefile of the study area
    # start_date: Start date of the study period; end_date: End date of the study period
    # land_cover_data: Land cover data compatible with the remote sensing data, study area and study period
    # remote_sensing_data: MODIS data are required for this class
    def __init__(self, study_area_shp_path, start_date, end_date, land_cover_data, remote_sensing_data):        
        # Gee Authentication and Initialization
        # Please run this just once to get access to the resources in GEE
        ee.Authenticate()
        try:
            #Initialize the Earth Engine object, using the authentication credentials
            ee.Initialize()
            print('Google Earth Engine has initialized successfully!')
        except ee.EEException as e:
            print('Google Earth Engine has failed to initialize!')
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        # Initialize the class with the study area, start and end dates
        self.study_area = geemap.shp_to_ee(study_area_shp_path)  # Convert shapefile to Earth Engine object
        self.start_date = start_date
        self.end_date = end_date
        self.land_cover_data = land_cover_data
        self.remote_sensing_data = remote_sensing_data
        self.landCover = ee.Image(self.land_cover_data).select('landcover')  # Load land cover data
        # Load MODIS data, filter by date and apply forest mask
        self.forest_data = ee.ImageCollection(self.remote_sensing_data).filter(ee.Filter.date(self.start_date, self.end_date)).map(self.maskStudyArea)
        self.forest_data_with_NDVI = self.forest_data.map(self.addVariables)  # Add variables to the filtered data

    # Mask the remote sensing data with the land cover of forest
    def maskStudyArea(self, image):
        # Create a mask for the study area
        mask = self.landCover.clip(self.study_area).updateMask(self.landCover.eq(1).Or(self.landCover.eq(2)).Or(self.landCover.eq(3)).Or(self.landCover.eq(4)).Or(self.landCover.eq(5)))
        return image.updateMask(mask)  # Apply the mask to the image

    # Add NDVI band to the remote sensing data
    def addVariables(self, image):
        # Add NDVI band to the image
        return image.addBands(image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).rename('NDVI'))

    # (Part of Cloud masking) Extract binary information from the quality band
    def bitwiseExtract(self, value, fromBit, toBit):
        # Extract bits from a number
        if toBit is None:
            toBit = fromBit
        maskSize = ee.Number(1).add(toBit).subtract(fromBit)
        mask = ee.Number(1).leftShift(maskSize).subtract(1)
        return value.rightShift(fromBit).bitwiseAnd(mask)

    # Mask clouds in the remote sensing data (parameters applicable for MODIS)
    def maskClouds(self, image):
        # Create a mask for clouds
        quality = image.select('QA')
        cloudState = self.bitwiseExtract(quality, 0, 1)
        cloudShadowState = self.bitwiseExtract(quality, 2)
        cirrusState = self.bitwiseExtract(quality, 8, 9)
        mask = cloudState.eq(0).And(cloudShadowState.eq(0)).And(cirrusState.eq(0))  # Clear, No cloud shadow, No cirrus
        return image.updateMask(mask)  # Apply the mask to the image

    # Extrace NDVI from each time series and make them bands in one image
    def NDVI_Exract_Stack(self):
        # Process the MODIS data
        NDVI = self.forest_data_with_NDVI.select("NDVI")
        NDVI_stack = ee.Image(NDVI.toBands())
        return NDVI_stack
    
    # Download the image (with its bandnames) to the local directory
    def export_image(self, image, filename, filename_band_name, scale, crs, unmask_value):
        # Define export parameters
        export_params = {
            'ee_object': image,
            'filename': filename,
            'region': self.study_area.geometry(),
            'scale': scale,
            'crs': crs,
            'unmask_value': unmask_value,
            'file_per_band': False
        }

        # Export image
        geemap.ee_export_image(**export_params)
        # Print out a message to indicate that the export task has started
        print(f'Exporting {filename} to GeoTIFF...')

        # Get the list of band names.
        band_names = image.bandNames().getInfo()
        # Convert the list to a NumPy array.
        band_names_array = np.array(band_names)
        # Save the NumPy array to a file.
        np.save(filename_band_name, band_names_array)
        print(f'Band names saved to {filename_band_name}')
        
        return band_names

    # Export the final dataset to a geemap
    def geemap_export(self, image, filename):
        # Visualize the study area and the NDVI stack
        Map = geemap.Map()
        Map.addLayer(self.study_area, {}, 'Study Area')
        Map.addLayer(image, {}, 'Fianl dataset')
        Map.centerObject(self.study_area, 10)
        Map.to_html(filename)
        print(f'Geemap saved to {filename}.')
