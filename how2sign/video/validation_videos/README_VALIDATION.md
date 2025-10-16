# How2Sign Validation Data Processing

This directory contains scripts for processing How2Sign validation videos and generating JSON files for training.

## Files Created

1. **segment_videos_val.py** - Segments validation videos into clips
2. **generate_segmented_json_val.py** - Generates JSON file for training with question diversity
3. **how2sign_realigned_val.csv** - Source CSV with validation data (1,743 segments)

## Processing Steps

### Step 1: Segment Validation Videos

Segments videos from the raw videos folder into individual clips based on timestamps in the CSV.

```bash
cd D:\sign_language_llm\how2sign\video\validation_videos
python segment_videos_val.py
```

**Default Parameters:**
- Input CSV: `how2sign_realigned_val.csv`
- Input videos: `D:\val_raw_videos\raw_videos`
- Output clips: `D:\how2sign_val_segment_clips`

**Optional Parameters:**
```bash
# Test with 10 clips first
python segment_videos_val.py --max-clips 10

# Custom paths
python segment_videos_val.py --input-csv "path/to/csv" --input-dir "path/to/videos" --output-dir "path/to/output"

# Verbose logging
python segment_videos_val.py --verbose
```

**Output:**
- Segmented video clips in `D:\how2sign_val_segment_clips`
- Log file: `video_segmentation_val.log`
- Metadata: `D:\how2sign_val_segment_clips\segmentation_metadata_val.json`

### Step 2: Generate JSON for Training

Creates a JSON file with diverse questions for each video clip.

```bash
cd D:\sign_language_llm\how2sign\video\validation_videos
python generate_segmented_json_val.py
```

**Default Parameters:**
- Input CSV: `how2sign_realigned_val.csv`
- Video directory: `D:\how2sign_val_segment_clips`
- Output JSON: `segmented_val_videos.json`
- Random seed: 42

**Optional Parameters:**
```bash
# Custom seed for different question distribution
python generate_segmented_json_val.py --seed 123

# Custom paths
python generate_segmented_json_val.py --video-dir "path/to/clips" --output-json "output.json"

# Verbose logging
python generate_segmented_json_val.py --verbose
```

**Output:**
- JSON file: `segmented_val_videos.json`
- Log file: `generate_segmented_json_val.log`

## JSON Format

The generated JSON follows this format with 5 diverse question variations:

```json
[
  {
    "id": "video_id",
    "video": "video_filename.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\n[One of 5 question variations]"
      },
      {
        "from": "gpt",
        "value": "Translation text"
      }
    ]
  }
]
```

### Question Variations (randomly selected):
1. "Translate the American Sign Language in this video to English."
2. "Provide the English translation of this ASL video."
3. "What is being signed in American Sign Language in this video?"
4. "Translate this ASL video to English."
5. "Convert this American Sign Language video to English text."

## Expected Results

- **Validation segments**: ~1,743 clips
- **Success rate**: Should be close to 100% if all videos are present
- **Question distribution**: Approximately 20% for each of the 5 questions (due to random seed 42)

## Troubleshooting

### Video files not found
- Verify videos are in `D:\val_raw_videos\raw_videos`
- Check CSV file has correct VIDEO_NAME values
- Ensure videos have `.mp4` extension

### Python not found
- Make sure Python 3.10+ is installed
- Install required packages: `pip install opencv-python pandas tqdm`

### No JSON entries created
- First run the segmentation script to create video clips
- Verify clips exist in output directory
- Check CSV has SENTENCE_NAME and SENTENCE columns

## Related Files

For training and test data, see:
- Training: `how2sign/video/train_videos/`
- Testing: `how2sign/video/test_raw_videos/`

