import random
import string
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import locale
import nltk

from nltk.corpus import words as nltk_words

# Function to generate a random date in the format "dd.mm"
def generate_random_date():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    return random_date.strftime("%d.%m")

# Function to generate a random positive or negative amount
def generate_random_amount():
    return round(random.uniform(-5000, 5000), 2)

# Function to generate a random entry text with spaces
def generate_random_text(min_words=1, max_words=4):
    use_real_words = random.random() < .05
    if use_real_words:
        word_count = random.randint(1, 4)  # Random number of words in the text
        word_list = nltk_words.words()
        words = [random.choice(word_list).capitalize() for _ in range(word_count)]
        return " ".join(words)
    else:
        word_count = random.randint(1, 4)  # Random number of words in the text
        alphabet = "æøå" + string.ascii_lowercase + "æøå"
        n = len(alphabet)
        words = [''.join([alphabet[random.randint(0, n - 1)] for _ in range(random.randint(1, 12))]) for _ in range(word_count)]
        words = [words[0].capitalize()] + words[1:]
        return " ".join(words)

# Function to generate the fake data
def generate_fake_data(num_entries, flip_format=False):
    data = []
    balance = 0
    for _ in range(num_entries):
        date = generate_random_date()
        text = generate_random_text()
        text = (text[:28] + '...') if len(text) > 31 else text
        interest_date = date if random.random() < 0.2 else ""
        amount = generate_random_amount() if random.random() < 0.8 else 0.0
        balance += amount
        amount_fmtd = f"{amount:>12,.2f}"
        balance_fmtd = f"{balance:>12,.2f}"
        if flip_format:
            amount_fmtd = amount_fmtd.replace(".", ',').replace(',', '.', 1)
            balance_fmtd = balance_fmtd.replace(".", ',').replace(',', '.', 1)
        entry = f"{date} | {text.ljust(31)} | {interest_date.ljust(14)} {amount_fmtd} {balance_fmtd}"
        data.append(entry)
    return data

def get_font():
    random.seed(datetime.now().timestamp())
    r = random.randint(0, 2)
    match r:
        case 0:
            return ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf', 14)
        case 1:
            return ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 13)
        case _:
            return ImageFont.truetype('/usr/share/fonts/truetype/firacodenf/FiraCodeNerdFontMono-Regular.ttf', 12)

data_font = get_font()
header_font = get_font()

def create_training_set(i):
    global monospace
    header = f"Account+                                                 Account no. {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(100, 999)}"
    separator = "+-----+---------------------------------+-----------------------------------------"
    column_headers = "Date  | Text                            | Intrst. Date     Amount        Balance"
    new_balance = "        New balance                                                       0.00"

    num_entries = random.randint(16, 48)  # You can adjust the number of entries as per your requirement
    flip = random.random() < 0.5
    data = generate_fake_data(num_entries, flip)
    lines = []
    lines.append(header)
    lines.append(separator)
    lines.append(column_headers)
    lines.append(separator)
    for entry in data:
        lines.append(entry)
    lines.append(separator)
    lines.append(new_balance)
    lines.append("-" * 82)

    with open(f"fake_data/output_{i}.gt.txt", "w") as file:
        file.writelines([f'{line}\n' for line in lines])

    arr = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    h, w, _ = arr.shape
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    fake_header = generate_random_text()
    line_start = (int(random.uniform(124, 124*2)), int(random.uniform(124, 124*2)))
    draw.text((60, 60), fake_header, (0, 0, 0), font=header_font)
    line_height = 14
    for j, line in enumerate(lines):
        draw.text((line_start[0], line_start[1] + line_height * j), line, (0, 0, 0), font=data_font)
    line_end = line_start[1] + line_height * (j + 1)
    offset = int(random.uniform(60, 124*4))
    for k in range(random.randint(6, 20)):
        draw.text((offset, line_end + line_height + (line_height + 2) * k), generate_random_text(8, 12), (0, 0, 0), font=header_font)
    arr = np.array(img)
    graininess_strength = random.random() * 0.04
    max_distance = np.sqrt((w // 2)**2 + (h // 2)**2)
    center_x, center_y = random.randint(0, w - 1), random.randint(0, h - 1)
    for y in range(h):
        for x in range(w):
            dist = np.hypot((x - center_x), (y - center_y))
            noise_strength = graininess_strength * (1 - dist / max_distance)
            color = [255 * noise_strength for _ in range(3)]
            arr[y, x] = np.clip(arr[y, x] + color, 0, 255)
    
    rotation_angle = random.uniform(-5.0, 5.0)
    M = cv2.getRotationMatrix2D((h / 2, w / 2), rotation_angle, 1.0)
    cv2.warpAffine(arr, M, (h, w), dst=arr)
    cv2.imwrite(f"fake_data/output_{i}.png", arr)

from multiprocessing import Pool
from glob import glob

# Main function to print the fake data
def main():
    nltk.download('words')
    random.seed(datetime.now().timestamp())
    offset = max([int(g.split('/')[-1].split("_")[1].split(".")[0]) for g in glob('fake_data/*.png')])
    with Pool(16) as pool:
        pool.map(create_training_set, range(offset + 1, offset + 16, 1))
    
if __name__ == "__main__":
    main()
