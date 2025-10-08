#!/bin/bash
# Script to cleanup old checkpoints, keeping only the latest N

OUTPUT_DIR="$1"
KEEP_LAST="${2:-2}"  # Default keep 2

if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <output_directory> [num_to_keep]"
    echo "Example: $0 /path/to/output 2"
    exit 1
fi

echo "Cleaning up checkpoints in: $OUTPUT_DIR"
echo "Keeping last: $KEEP_LAST checkpoints"

# List all checkpoint directories sorted by step number
CHECKPOINTS=($(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n))

TOTAL=${#CHECKPOINTS[@]}

if [ $TOTAL -le $KEEP_LAST ]; then
    echo "Only $TOTAL checkpoints found. Nothing to delete."
    exit 0
fi

# Calculate how many to delete
TO_DELETE=$((TOTAL - KEEP_LAST))

echo "Found $TOTAL checkpoints, will delete $TO_DELETE oldest ones"

# Delete the oldest checkpoints
for (( i=0; i<$TO_DELETE; i++ )); do
    CHECKPOINT="${CHECKPOINTS[$i]}"
    echo "Deleting: $CHECKPOINT"
    rm -rf "$CHECKPOINT"
done

echo "Cleanup complete. Remaining checkpoints:"
ls -lhtr $OUTPUT_DIR/checkpoint-* 2>/dev/null



