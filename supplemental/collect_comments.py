import os

# Directory where data is stored
base_directory = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results\\ecc-all-LC'

# Output file where combined comments will be stored
output_file = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results\\ecc-all-LC\\combined_comments.txt'

# Open the output file in append mode
with open(output_file, 'w') as combined_file:
    # Iterate over directories in base_directory
    for root, dirs, files in os.walk(base_directory):
        # Iterate over subdirectories (koi_id directories)
        for dir_name in dirs:
            koi_id_dir = os.path.join(root, dir_name)
            comments_file = os.path.join(koi_id_dir, f'{dir_name}_comments.txt')
            
            # Check if the comments file exists
            if os.path.exists(comments_file):
                with open(comments_file, 'r') as f:
                    # Read comments from koi_id_comments.txt and write to combined_file
                    comments = f.read()
                    combined_file.write(f"=== {dir_name} ===\n")
                    combined_file.write(comments)
                    combined_file.write("\n\n")
                    print(f"Processed {comments_file}")

print(f"All comments combined and stored in {output_file}")