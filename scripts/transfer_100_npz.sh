#!/bin/bash
# Transfer 100 npz files from source to destination via rsync
# Usage: bash scripts/transfer_100_npz.sh

SOURCE_DIR="/shared/rc/llm-gen-agent/mhu/videos/train_crop_videos_720_mask"
DEST="ztao@ztao-scu.main.ad.rit.edu:/local1/mhu/sign_language_llm/how2sign/masks"

# Create file list with only filenames (not full paths)
FILE_LIST="/tmp/npz_files_to_transfer.txt"
ls -1 "$SOURCE_DIR"/*.npz | head -100 | sed 's|.*/||' > "$FILE_LIST"

echo "Prepared file list with $(wc -l < "$FILE_LIST") files"
echo "First 5 files:"
head -5 "$FILE_LIST"
echo ""

# Transfer files using rsync
echo "Starting transfer..."
rsync -avz --progress --files-from="$FILE_LIST" "$SOURCE_DIR/" "$DEST"

# Clean up
rm -f "$FILE_LIST"
echo "Transfer completed!"

