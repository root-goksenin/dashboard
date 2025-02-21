import os 
import glob 
import shutil 
import json


if __name__ == "__main__":
    ids = glob.glob("../backend_plots/*.png")
    house_id_to_seed = {}
    for id in ids:
        id = id.replace(".png", "")
        house_id, seed = os.path.basename(id).split("_")
        if house_id not in house_id_to_seed:
            house_id_to_seed[house_id] = [] 
        house_id_to_seed[house_id].append(seed)
    
    
    house_id_to_seed = {house_id : list(sorted(map(lambda x: int(x), arr))) for house_id, arr in house_id_to_seed.items()}
    rirs = []
    for house_id in house_id_to_seed:
        with open(f"../data/{house_id}.json", "r") as f:
            data = json.load(f)
        for data_region in data['sampled_regions']:
            for seed in house_id_to_seed[house_id][:10]:
                if str(seed) == str(data_region["region"]["seed"]):
                    rirs.append(os.path.basename(data_region["region"]["scene"]["source"]["rir"]["rir_path"]))
                    for noise in data_region["region"]["scene"]["noise"]:
                        rirs.append(os.path.basename(noise["rir"]["rir_path"]))
    print(len(rirs))
    for rir in rirs:
        shutil.copy(f"/projects/0/prjs1338/RIRs/{rir}", ".")
    
    for house_id in house_id_to_seed: 
        for seed in house_id_to_seed[house_id][:10]:
             shutil.copy(f"../backend_plots/{house_id}_{seed}.png", "../plots")

