from pathlib import Path
from PIL import Image
import dvc.api

params = dvc.api.params_show()


NEW_IM_WIDTH = params["prepare"]["IM_WIDTH"]
NEW_IM_HEIGHT = params["prepare"]["IM_HEIGHT"]

RAW_DATA_PATH = Path('data/raw')
TRAIN_RAW_DATA_PATH = RAW_DATA_PATH / 'train'

INTERIM_DATA_PATH = Path('data/interim')
TRAIN_INTERIM_DATA_PATH = INTERIM_DATA_PATH / 'train'

IM_WIDTH = 224
IM_HEIGHT = 224

def resize_image(im: Image):
    return im.resize(IM_WIDTH, IM_HEIGHT)

def process_image(im_path: Path, output_dir: Path):
    im = Image.open(im_path).convert('RGB')
    im = resize_image(im)
    im.save(str(output_dir / im_path.name))

def main():
    for im_path in TRAIN_RAW_DATA_PATH.iterdir():
        process_image(im_path, TRAIN_INTERIM_DATA_PATH)
        
if __name__ == '__main__':
    main()