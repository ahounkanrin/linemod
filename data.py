import os
from tqdm import tqdm
import pandas as pd

classes = ["01", "02", "03", "04", "05", "06", "07", "08", "09",
            "10", "11", "12", "13", "14", "15"]
data_dir = "/home/user/Datasets/hinterstoisser/train/"

for i in classes:
    df = pd.DataFrame(columns=["rgb", "depth"])
    df["rgb"] = sorted(os.listdir(data_dir+i+"/rgb/"))
    df["depth"] = sorted(os.listdir(data_dir+i+"/depth/"))
    df.to_csv(i+".csv", index=False)
