from cv2 import imread, IMREAD_GRAYSCALE
from table_ocr.extract_tables import find_tables
from glob import glob

for img_path in glob("*.png"):
    img = imread(img_path, IMREAD_GRAYSCALE)
    print(img_path, find_tables(img))


