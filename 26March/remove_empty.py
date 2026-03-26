import os

def remove_empty_bad_quality_folders(root_folder):
    """
    Remove empty directories ending with '_bad_Q' within the specified root folder.

    Args:
        root_folder (str): The path to the root directory to search within.
    """
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if dirname.endswith('_bad_Q') and not os.listdir(full_path):
                os.rmdir(full_path)
                print(f"Removed empty directory: {full_path}")

if __name__ == "__main__":
    root_folder = "K:/Bart export/Stationary_phase_screen/PLATE33A"  # Replace with your root folder path
    remove_empty_bad_quality_folders(root_folder)
