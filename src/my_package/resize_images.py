import os
from PIL import Image
from tqdm import tqdm

from my_package.constants import DATA_DIR

def process_images(input_root, output_root, new_size):
    # Ensure the output base directory exists
    os.makedirs(output_root, exist_ok=True)

    # Iterate through each folder (person) in lfw2
    persons = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    print(f"Found {len(persons)} folders. Starting resizing...")

    for person in tqdm(persons, desc="Processing Persons"):
        input_person_path = os.path.join(input_root, person)
        output_person_path = os.path.join(output_root, person)

        # Create the corresponding folder in the foreground directory
        os.makedirs(output_person_path, exist_ok=True)

        # Process each image in the person's folder
        for img_name in os.listdir(input_person_path):
            input_img_path = os.path.join(input_person_path, img_name)
            
            # Change extension to .png to support transparency
            base_name = os.path.splitext(img_name)[0]
            output_img_path = os.path.join(output_person_path, f"{base_name}.jpg")

            # Skip if already processed
            if os.path.exists(output_img_path):
                continue

            with Image.open(input_img_path) as img:
                img = img.convert('L')   
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                img.save(output_img_path)

def main():
    data_main_path = DATA_DIR
    # Fixed the os.join typo to os.path.join
    imgs_main_path = os.path.join(data_main_path, 'lfw2')
    imgs_resized_path = os.path.join(data_main_path, 'resized')
    new_size = (105, 105)

    process_images(imgs_main_path, imgs_resized_path, new_size)
    print("Resizing removal complete.")

if __name__ == '__main__':
    main()