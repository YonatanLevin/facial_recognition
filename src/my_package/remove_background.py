import os
from rembg import remove
from PIL import Image
from tqdm import tqdm

def process_images(input_root, output_root):
    # Ensure the output base directory exists
    os.makedirs(output_root, exist_ok=True)

    # Iterate through each folder (person) in lfw2
    persons = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    print(f"Found {len(persons)} folders. Starting background removal...")

    for person in tqdm(persons, desc="Processing Persons"):
        input_person_path = os.path.join(input_root, person)
        output_person_path = os.path.join(output_root, person)

        # Create the corresponding folder in the foreground directory
        os.makedirs(output_person_path, exist_ok=True)

        # Process each image in the person's folder
        for img_name in os.listdir(input_person_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_img_path = os.path.join(input_person_path, img_name)
                
                # Change extension to .png to support transparency
                base_name = os.path.splitext(img_name)[0]
                output_img_path = os.path.join(output_person_path, f"{base_name}.png")

                # Skip if already processed
                if os.path.exists(output_img_path):
                    continue

                try:
                    # Open the image and remove background
                    with open(input_img_path, 'rb') as i:
                        input_data = i.read()
                        output_data = remove(input_data)
                    
                    with open(output_img_path, 'wb') as o:
                        o.write(output_data)
                except Exception as e:
                    print(f"Error processing {input_img_path}: {e}")

def main():
    data_main_path = 'data'
    # Fixed the os.join typo to os.path.join
    imgs_main_path = os.path.join(data_main_path, 'lfw2')
    imgs_foreground_path = os.path.join(data_main_path, 'foreground')

    process_images(imgs_main_path, imgs_foreground_path)
    print("Background removal complete.")

if __name__ == '__main__':
    main()