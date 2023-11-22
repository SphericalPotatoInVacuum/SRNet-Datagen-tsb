#!/bin/bash

# fonts_dir is the directory where the fonts are located
fonts_dir="$1"
# destination_dir is the directory where the fonts will be moved to
destination_dir="$2"

# Create the destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Read font_path from fonts_dir/fontlist_sub.txt and move the fints_dir/font_path to destination_dir
while IFS= read -r font_path; do
    mv "$fonts_dir/$font_path" "$destination_dir"
done < "$fonts_dir/fontlist_sub.txt"

echo "All fonts have been moved to ${destination_dir}."
