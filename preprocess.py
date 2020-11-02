from os import listdir, remove
from os.path import isfile, join
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError

masks_dir = Path("./train_data/masks")
covers_dir = Path('./train_data/covers')
files = [f for f in listdir(masks_dir) if isfile(join(masks_dir, f))]
exclusions = []

for f in files:
    try:
        mask = Image.open(join(masks_dir, f))
        mask_data = np.asarray(mask)
        colors, counts = np.unique(mask_data.reshape(-1, 3), return_counts=True, axis=0)
        if counts.size != 4:
            exclusions.append(f)
            print(f, colors, counts)
    except UnidentifiedImageError:
        pass
print('{} images removed:\n'.format(len(exclusions), exclusions))

for e in exclusions:
    remove(join(masks_dir, e))
    remove(join(covers_dir, e))
