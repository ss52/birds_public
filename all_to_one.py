# %%
from pathlib import Path
import pandas as pd
# import glob
# import gc
import logging

# %%
base_dir = "data\\raw\\"
log_file = "log.txt"

# %%
logging.basicConfig(filename=log_file, encoding='utf-8')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%
path_to_files = f"{base_dir}gus separate files"

df = pd.DataFrame()
# not_first_file = False

for file in Path(path_to_files).glob("*.parquet"):
    # here we concat all files to one df
    # if not_first_file:
    #    df = pd.read_parquet(f"{base_dir}all_gus_2024.parquet")
    # else:
    #    df = pd.DataFrame()
    #    not_first_file = True
    df = pd.concat([df, pd.read_parquet(file)])
    logger.info(file)
    # print(file)
    # df.to_parquet(f"{base_dir}all_gus_2024.parquet")
    # del df
    # gc.collect()

# %%
# df.head()
df['Bird_id'] = df['Bird_id'].astype(str)

# %%
df.to_parquet(f"{base_dir}geese_all_11_2024.parquet")
# %%
