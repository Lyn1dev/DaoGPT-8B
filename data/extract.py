import os
from epub2txt import epub2txt

# Explicitly ordered list based on your directory structure
epub_files = [
    "reverend-insanity-c1-c500.epub",
    "reverend-insanity-c501-1000.epub",
    "reverend-insanity-c1001-1500.epub",
    "reverend-insanity-c1501-c2000.epub",
    "reverend-insanity-c2001-end.epub"
]

output_file = "reverend_insanity_raw.txt"

print(f"Starting extraction for {len(epub_files)} files...")

with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in epub_files:
        if os.path.exists(filename):
            print(f"Extracting: {filename}")
            
            # Extracts plain text from the EPUB
            text = epub2txt(filename) 
            
            outfile.write(text)
            outfile.write("\n\n") # Add spacing between books
        else:
            print(f"Error: Could not find {filename}. Check the file name.")

print(f"Success! All text compiled into {output_file}")
