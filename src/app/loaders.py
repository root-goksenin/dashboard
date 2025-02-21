import numpy as np 
import tarfile
import json 
import os 
from io import BytesIO


def load_json_with_houseid(houseid:str):
    with open(f"data/{houseid}.json", "r") as f: 
        data = json.load(f)
        return data
    
def load_rir_from_npy(rir_path):
    source_rir = np.load(rir_path)
    return source_rir

def load_rir_from_tar(rir_uuid, tar_):
  with tarfile.open(tar_, 'r:*') as f:
    path = os.path.commonprefix(f.getnames())
    extracted = f.extractfile(member=f"{path}/{rir_uuid}").read()
    array_file = BytesIO()  
    array_file.write(extracted)
    array_file.seek(0)
    na = np.load(array_file)
    return na
  
def load_json_from_tar(houseid:str):
  with tarfile.open(f"/home/gyuksel3/habitat_folder/rir_generator/train/{houseid}", 'r:*') as f:
    path = os.path.commonprefix(f.getnames())
    print(path)
    extracted = f.extractfile(member=f"{path}/rir_meta_data.json").read()
    return json.loads(extracted)


  

  