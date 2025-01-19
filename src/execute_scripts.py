import os
import subprocess

def get_files_in_folder(folder_path):
    # List all files in the specified folder
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return file_list

# Example usage
folder_path = 'scripts/'  # Change to the folder path you're interested in
files = get_files_in_folder(folder_path)

for i in range(0,10):
    if i < len(files):
        command = ['echo', files[i]]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    
    
    
