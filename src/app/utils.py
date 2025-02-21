import glob 
import os 

import numpy as np 

def delete_assets():
   files = glob.glob("assets/*")
   for file in files: 
      os.remove(file)

def create_figure_template(title):
    return {  # Remove the 'layout' nesting
        'title': {'text': title, 'font': {'size': 18}},
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': "Segoe UI, Arial"},
        'margin': {'t': 40, 'b': 40},
        'hoverlabel': {'font_size': 12}
    }


def polar_to_cartesian(azimuth, elevation, r):
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    x = r * np.cos(elevation) * np.sin(azimuth)
    y = r * np.sin(elevation)
    z = -r * np.cos(elevation) * np.cos(azimuth)  # Negative Z for forward
    return x[0], y[0], z[0]
