import json
import glob 
import os

existing = 0
files = glob.glob("*.json")
for file in files:
    with open(file, "r") as json_file: 
        data = json.load(json_file)
        for regions in data['sampled_regions']: 
            source_rir = os.path.basename(regions['region']['scene']['source']['rir']['rir_path'])
            if os.path.exists(f"/projects/0/prjs1338/RIRs/{source_rir}"):
                existing += 1
            else:
                print(f"SOURCE RIR DOES NOT EXIST : {source_rir}")
            for noise in regions['region']['scene']['noise']:
                noise_rir = os.path.basename(noise['rir']['rir_path'])
                if os.path.exists(f"/projects/0/prjs1338/RIRs/{noise_rir}"):
                    existing += 1
                else:
                    print(f"NOISE RIR DOES NOT EXIST : {noise_rir}")

print("TOTAL RIRS CHECKED", existing)