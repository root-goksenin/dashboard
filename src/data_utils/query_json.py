import numpy as np 
import os

def find_area_from_size(size: np.ndarray):
    return size[0] * size[1] * size[2]

def find_floor_area_from_size(size: np.ndarray):
    return size[0] * size[2]

def find_floor_area(data):
    areas = []
    for region in data['sampled_regions']:
        for v in region['region'].values():
            if isinstance(v, dict) and 'category_name' in v.keys(): 
                size = find_floor_area_from_size(v['bbox']['size'])
                assert size > 0, "area can not be negative"
                areas.append(size)
    return np.array(areas)

def find_area(data):
    areas = []
    for region in data['sampled_regions']:
        for v in region['region'].values():
            if isinstance(v, dict) and 'category_name' in v.keys(): 
                size = find_area_from_size(v['bbox']['size'])
                assert size > 0, "area can not be negative"
                areas.append(size)
    return np.array(areas)

def find_category_names(data):
    category_names = []
    for region in data['sampled_regions']:
        for v in region['region'].values():
            if isinstance(v, dict) and 'category_name' in v.keys(): 
                category_names.append(v['category_name'])
    return category_names

def find_agent_rotation(data):
    azimuth = []
    elevation = []
    for region in data['sampled_regions']:
        azimuth.append(region['region']['agent_data']['azimuth'][0])
        elevation.append(region['region']['agent_data']['elevation'][0])
    return np.array(azimuth), np.array(elevation)

def find_agent_height(data):
    height = []
    for region in data['sampled_regions']:
        height.append(region['region']['sensor_states']['audio_sensor']['sensor_position'][1])
    return np.array(height)

def find_sound_source_position(data):
    radius = [] 
    elevation = [] 
    azimuth = []
    for region in data['sampled_regions']:
        azimuth.append(region['region']['scene']['source']['azimuth'])
        elevation.append(region['region']['scene']['source']['elevation'])
        radius.append(region['region']['scene']['source']['radius'])
    return np.array(radius), np.array(azimuth), np.array(elevation)

def find_noise_position(data):
    radius = [] 
    elevation = [] 
    azimuth = []
    nr_of_noise = []
    for region in data['sampled_regions']:
        nr_of_noise.append(len(region['region']['scene']['noise']))
        for noise in region['region']['scene']['noise']:
            azimuth.append(noise['azimuth'])
            elevation.append(noise['elevation'])
            radius.append(noise['radius'])
    return np.array(radius), np.array(elevation), np.array(azimuth), np.array(nr_of_noise)


def get_agent_ear_position(sample):
    return sample["region"]["sensor_states"]["audio_sensor"]["agent_sensor_position"]

def get_sphere_radius(sample):
    scene = sample["region"]["scene"]
    return scene["source"]["radius"] 

def get_agent_rotation(sample):
    azimuth =  sample["region"]["sensor_states"]["audio_sensor"]["sensor_azimuth"]
    elevation =  sample["region"]["sensor_states"]["audio_sensor"]["sensor_elevation"]
    return azimuth, elevation 

def get_source_location(sample):
    scene = sample["region"]["scene"]
    return scene["source"]["position"]    

def get_source_azimuth(sample):
    scene = sample["region"]["scene"]
    return scene["source"]["azimuth"]   
 
def get_source_elevation(sample):
    scene = sample["region"]["scene"]
    return scene["source"]["elevation"]  

def get_sampled_region_with_rir(rir, data):
    for region in data['sampled_regions']:
        scene = region['region']['scene']
        if os.path.basename(scene['source']['rir']['rir_path']) == rir:
            return region, scene['source']['position'], scene['source']['radius']
        for noise in scene['noise']:
            if os.path.basename(noise['rir']['rir_path']) == rir:
                return region, noise['position'], noise['radius']
            
def get_source_efficency(data):
    eff = []
    for region in data['sampled_regions']:
        eff.append(region['region']['scene']['source']['rir']['ray_efficiency'])
    return np.array(eff)

def get_noise_efficency(data):
    eff = []
    for region in data['sampled_regions']:
        for noise in region['region']['scene']['noise']:
            eff.append(noise['rir']['ray_efficiency'])
    return np.array(eff)
