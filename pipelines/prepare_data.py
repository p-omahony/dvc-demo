from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import dvc.api

params = dvc.api.params_show()

NEW_IM_WIDTH = params["prepare"]["IM_WIDTH"]
NEW_IM_HEIGHT = params["prepare"]["IM_HEIGHT"]

RAW_DATA_PATH = Path('data/raw')
TRAIN_RAW_DATA_PATH = RAW_DATA_PATH / 'train'

INTERIM_DATA_PATH = Path('data/interim')
TRAIN_INTERIM_DATA_PATH = INTERIM_DATA_PATH / 'train'
TEST_INTERIM_DATA_PATH = INTERIM_DATA_PATH / 'test'

def resize_image(im: Image):
    return im.resize((NEW_IM_WIDTH, NEW_IM_HEIGHT))

def process_image(im_path: Path, output_dir: Path):
    im = Image.open(im_path).convert('RGB')
    im = resize_image(im)
    im.save(str(output_dir / im_path.name))

def main():
    images = list(TRAIN_RAW_DATA_PATH.iterdir())
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
    
    if not TRAIN_INTERIM_DATA_PATH.exists(): TRAIN_INTERIM_DATA_PATH.mkdir()
    for im_path in train_images:
        process_image(im_path, TRAIN_INTERIM_DATA_PATH)
        
    if not TEST_INTERIM_DATA_PATH.exists(): TEST_INTERIM_DATA_PATH.mkdir()
    for im_path in test_images:
        process_image(im_path, TEST_INTERIM_DATA_PATH)
        
if __name__ == '__main__':
    main()