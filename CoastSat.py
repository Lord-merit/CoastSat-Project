########### Libraries ###########
#################################
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.ion()

from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
#################################

###### 1. Initial settings ######
#################################

# region of interest (longitude, latitude in WGS84)
polygon = [[[32.016931, 36.571692],
            [31.934707, 36.574691],
            [31.930087, 36.532495],
            [31.993986, 36.534978],
            [32.016931, 36.571692]]] 
# converting polygon to smallest rectangle
polygon = SDS_tools.smallest_rectangle(polygon)
# Date range
dates = ['2013-01-01', '2022-12-01']
# Satellite
sat_list = ['L8']
# Landsat collection
collection = 'C02'
# directory name
sitename = 'Kleopatra'
# directory to store data
filepath = os.path.join(os.getcwd(), 'data')
# putting all the inputs into a dictionary
inputs = {'polygon': polygon, 'dates': dates, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath,
         'landsat_collection': collection}

# checking how many images are available for your inputs before downloading images
SDS_download.check_images_available(inputs);


###### 2. Retrieve images #######
#################################

# retrieve satellite images from GEE
metadata = SDS_download.retrieve_images(inputs)


###### 3. Batch shoreline detection #######
###########################################

# settings for the shoreline extraction
settings = { 
    # General Parameters:
    'cloud_thresh': 0.5,        # Threshold on maximum cloud cover
    'dist_clouds': 300,         # Distance around clouds where shoreline can't be mapped
    'output_epsg': 4326,        # EPSG code of spatial reference system desired for the output (if error occurse try 3857)
    # Quality Control:
    'check_detection': True,    # Show each Shoreline detection for verification
    'adjust_detection': False,  # Adjusting the position of each Shoreline
    'save_figure': True,        # Saving a .jpg file showing the mapped Shoreline for each image
    # Advanced Controls:
    'min_beach_area': 1000,     # Minimum space for an object to be labeled as a beach (meters^2)
    'min_length_sl': 400,       # The minimum length of the Shoreline perimeter to apply (meters)
    'cloud_mask_issue': False,  # Are the sand pixels masked?
    'sand_color': 'default',    # Sand Color
    'pan_off': False,           # Horizontal scrolling for Landsat 8 images
    # Adding the inputs defined previously
    'inputs': inputs,
}
#Saved images as jpg
SDS_preprocess.save_jpg(metadata, settings)


# create a reference shoreline (helps to identify outliers and false detections)
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100

# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_shoreline.extract_shorelines(metadata, settings)

output = SDS_tools.remove_duplicates(output) # remove duplicates (images taken by the same satellite on the same date)
output = SDS_tools.remove_inaccurate_georef(output, 10) # remove incorrect georeferencing (threshold set to 10m)

# save output into a GEOJSON layer
from pyproj import CRS
geomtype = 'points' 
gdf = SDS_tools.output_to_gdf(output, geomtype)
gdf.crs = CRS(settings['output_epsg']) # setting the layer projection
# Saving GEOJSON layer to file 
gdf.to_file(os.path.join(inputs['filepath'], inputs['sitename'], '%s_output_%s.geojson'%(sitename,geomtype)),
                                driver='GeoJSON', encoding='utf-8')

# plot the mapped shorelines
fig = plt.figure(figsize=[15,8])
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
plt.legend();


£#### 4. Shoreline analysis #####
#################################

#load the transects from a .geojson file
geojson_file = os.path.join(os.getcwd(),'data', sitename, sitename + '_transects.geojson')
transects = SDS_tools.transects_from_geojson(geojson_file)

# plot the transects
fig = plt.figure(figsize=[15,8], tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
for i,key in enumerate(list(transects.keys())):
    plt.plot(transects[key][0,0],transects[key][0,1], 'bo', ms=5)
    plt.plot(transects[key][:,0],transects[key][:,1],'k-',lw=1)
    plt.text(transects[key][0,0]-100, transects[key][0,1]+100, key,
                va='center', ha='right', bbox=dict(boxstyle="square", ec='k',fc='w'))


# Distance along the shore from which shoreline points are to be considered to calculate the mid-intercept
settings_transects = {'along_dist':25}
cross_distance = SDS_transects.compute_intersection(output, transects, settings_transects)

settings_transects = { # parameters for calculating intersections
                      'along_dist':          25,        # Distance along the shore to be used to calculate the intersection
                      'min_points':          2,         # minimum number of Shorelinepoints to calculate intersection
                      'max_std':             15,        # maximum standard deviation for points around the slice
                      'max_range':           30,        # max range for points around the slice
                      'min_chainage':        -100,      # largest negative value across slice (slice origin towards land)
                      'multiple_inter':      'auto',    # mode to remove outliers
                      'prc_multiple':         0.1,      # percentage of time multiple intersections were present to use the maximum value
                     }

cross_distance = SDS_transects.compute_intersection_QC(output, transects, settings_transects)

# Plot the time-series of cross-shore shoreline change
fig = plt.figure(figsize=[15,8], tight_layout=True)
gs = gridspec.GridSpec(len(cross_distance),1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i,key in enumerate(cross_distance.keys()):
    if np.all(np.isnan(cross_distance[key])):
        continue
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.set_ylim([-50,50])
    ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w')
    ax.set_ylabel('distance [m]', fontsize=12)
    ax.text(0.5,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
            va='top', transform=ax.transAxes, fontsize=14)

# Save as .csv file
out_dict = dict([])
out_dict['dates'] = output['dates']
for key in transects.keys():
    out_dict[key] = cross_distance[key]
df = pd.DataFrame(out_dict)
fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],
                  'Kleopatra_time_series.csv')
df.to_csv(fn, sep=',')
print('Time-series of the shoreline change along the transects saved as:\n%s'%fn)