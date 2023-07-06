from sklearn.model_selection import train_test_split
import os
import shutil


def create_dir(path):
    if (os.path.exists(path)):
        shutil.rmtree(path)
    os.makedirs(path)
    return


def copy_files(in_path, out_path, files):
    for file in files:
        in_file_path = os.path.join(in_path, file)
        out_file_path = os.path.join(out_path, file)
        shutil.copy(in_file_path, out_file_path)
    return


def split_data(input_images, input_masks, train_images, train_masks, test_images):
    image_files = os.listdir(input_images)

    train_files, test_files = train_test_split(
        image_files, test_size=0.2, random_state=1204)

    create_dir(train_images)
    create_dir(train_masks)
    create_dir(test_images)

    copy_files(input_images, train_images, train_files)
    copy_files(input_masks, train_masks, train_files)
    copy_files(input_images, test_images, test_files)
