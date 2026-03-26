import os
import shutil

def move_back_files(root_folder):
    """
    Moves files from directories ending with '_bad_Q' back to their original folders.
    The original folder is assumed to be the same as the '_bad_Q' folder with that suffix removed.
    """
    # List all directories in root_folder ending with "_bad_Q"
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path) and folder_name.endswith("_bad_Q"):
            # Determine the original folder name by removing the "_bad_Q" suffix
            original_folder_name = folder_name[:-6]  # Remove last 6 characters ("_bad_Q")
            original_folder_path = os.path.join(root_folder, original_folder_name)

            if not os.path.exists(original_folder_path):
                print(f"Original folder '{original_folder_path}' does not exist. Skipping '{folder_path}'.")
                continue

            # Move all files from the bad folder back to the original folder
            for file in os.listdir(folder_path):
                source_path = os.path.join(folder_path, file)
                target_path = os.path.join(original_folder_path, file)
                try:
                    shutil.move(source_path, target_path)
                    print(f"Moved '{file}' from '{folder_path}' back to '{original_folder_path}'.")
                except Exception as e:
                    print(f"Error moving '{file}': {e}")

            # Optionally, remove the empty bad folder
            try:
                os.rmdir(folder_path)
                print(f"Removed empty folder '{folder_path}'.")
            except Exception as e:
                print(f"Could not remove folder '{folder_path}': {e}")

if __name__ == "__main__":
    base_path = 'F:/Export Bart/Stationary_phase_screen'
    plate_ids = [ '51B', '53A', '53B', '55A', '55B', '57A', '57B', 
    '59A', '59B', '61A', '61B', '63A', '63B', '65A', '65B', '67A', '67B', '69A', '69B', '71A', '71B', 
    '73A', '73B', '75A',
    '75B', '77A', '77B', '79A', '79B', '81A', '81B', 
    '83A', '83B', '85A', '85B', '87A', '87B', '89A', '89B', '91-3-5']  # Add more plate IDs as needed

    ev_dirs = [f"{base_path}/PLATE{plate_id}" for plate_id in plate_ids]
    for ev_dir in ev_dirs:
        print(f"Processing directory: {ev_dir}")
        move_back_files(ev_dir)
        print(f"Finished processing: {ev_dir}")
    