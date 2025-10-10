#!/usr/bin/env python3
"""
Video Segmentation Script for How2Sign Training Dataset
=======================================================

This script segments training videos into clips based on start and end times from a CSV file.
It processes the how2sign_realigned_train.csv file and creates individual video clips
for each sentence/segment.

Usage:
    python segment_videos_train.py [--input-csv CSV_FILE] [--input-dir VIDEO_DIR] [--output-dir OUTPUT_DIR] [--max-clips MAX_CLIPS]

Author: Generated for Sign Language LLM project
Date: 2025
"""

import os
import sys
import pandas as pd
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_segmentation_train.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VideoSegmenter:
    """Class to handle video segmentation based on CSV data"""
    
    def __init__(self, input_csv: str, input_dir: str, output_dir: str):
        """
        Initialize the video segmenter
        
        Args:
            input_csv: Path to the CSV file with segmentation data
            input_dir: Directory containing the input videos
            output_dir: Directory to save the segmented clips
        """
        self.input_csv = input_csv
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CSV data
        self.df = self._load_csv()
        
        # Statistics
        self.stats = {
            'total_segments': 0,
            'successful_segments': 0,
            'failed_segments': 0,
            'missing_videos': 0,
            'processing_errors': 0
        }
    
    def _load_csv(self) -> pd.DataFrame:
        """Load and validate the CSV file"""
        try:
            df = pd.read_csv(self.input_csv, sep='\t')
            logger.info(f"Loaded CSV with {len(df)} rows")
            
            # Validate required columns
            required_columns = ['VIDEO_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert time columns to float
            df['START_REALIGNED'] = pd.to_numeric(df['START_REALIGNED'], errors='coerce')
            df['END_REALIGNED'] = pd.to_numeric(df['END_REALIGNED'], errors='coerce')
            
            # Remove rows with invalid time data
            initial_count = len(df)
            df = df.dropna(subset=['START_REALIGNED', 'END_REALIGNED'])
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} rows with invalid time data")
            
            # Validate time ranges
            invalid_times = df[df['START_REALIGNED'] >= df['END_REALIGNED']]
            if len(invalid_times) > 0:
                logger.warning(f"Found {len(invalid_times)} rows with invalid time ranges (start >= end)")
                df = df[df['START_REALIGNED'] < df['END_REALIGNED']]
            
            logger.info(f"Valid data: {len(df)} segments")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def _get_video_path(self, video_name: str) -> Optional[Path]:
        """Get the full path to a video file"""
        # Try with .mp4 extension if not already present
        if not video_name.endswith('.mp4'):
            video_name_with_ext = video_name + '.mp4'
        else:
            video_name_with_ext = video_name
            
        video_path = self.input_dir / video_name_with_ext
        if video_path.exists():
            return video_path
        else:
            logger.warning(f"Video file not found: {video_path}")
            return None
    
    def _create_output_filename(self, row: pd.Series) -> str:
        """Create output filename for a video segment"""
        sentence_name = row['SENTENCE_NAME']
        
        # Clean the sentence name for filename (remove special characters)
        # Remove .mp4 extension if present and clean the name
        clean_name = sentence_name.replace('.mp4', '')
        clean_name = clean_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c in '._-')
        
        # Create filename
        filename = f"{clean_name}.mp4"
        
        return filename
    
    def _segment_video(self, video_path: Path, start_time: float, end_time: float, output_path: Path) -> bool:
        """
        Segment a video using OpenCV
        
        Args:
            video_path: Path to input video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path for output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Validate frame numbers
            if start_frame >= total_frames or end_frame > total_frames:
                logger.error(f"Invalid frame range: {start_frame}-{end_frame} (total: {total_frames})")
                cap.release()
                return False
            
            # Set up video writer with more compatible codec
            # Try XVID codec first (most compatible), then fallback to others
            codecs_to_try = ['XVID', 'MJPG', 'mp4v', 'H264']
            out = None
            
            for codec in codecs_to_try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if out.isOpened():
                    logger.debug(f"Using codec: {codec}")
                    break
                else:
                    out.release()
            
            if not out.isOpened():
                logger.error(f"Cannot create output video: {output_path}")
                cap.release()
                return False
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read and write frames
            current_frame = start_frame
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                current_frame += 1
            
            # Clean up
            cap.release()
            out.release()
            
            # Verify output file was created and has content
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.debug(f"Successfully created segment: {output_path}")
                return True
            else:
                logger.error(f"Output file is empty or doesn't exist: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error segmenting video {video_path}: {e}")
            return False
    
    def process_segments(self, max_clips: Optional[int] = None, video_filter: Optional[str] = None) -> dict:
        """
        Process all video segments
        
        Args:
            max_clips: Maximum number of clips to process (for testing)
            video_filter: Filter to process only specific videos (e.g., "-fZc293MpJk")
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting video segmentation process...")
        
        # Filter data if needed
        df_to_process = self.df.copy()
        
        if video_filter:
            df_to_process = df_to_process[df_to_process['VIDEO_NAME'].str.contains(video_filter, na=False)]
            logger.info(f"Filtered to {len(df_to_process)} segments for video: {video_filter}")
        
        if max_clips:
            df_to_process = df_to_process.head(max_clips)
            logger.info(f"Limited to {max_clips} clips for testing")
        
        self.stats['total_segments'] = len(df_to_process)
        
        # Group by video to process efficiently
        video_groups = df_to_process.groupby('VIDEO_NAME')
        
        for video_name, group in tqdm(video_groups, desc="Processing videos"):
            video_path = self._get_video_path(video_name)
            
            if video_path is None:
                self.stats['missing_videos'] += len(group)
                continue
            
            logger.info(f"Processing video: {video_name} ({len(group)} segments)")
            
            # Process each segment for this video
            for idx, row in group.iterrows():
                try:
                    # Create output filename
                    output_filename = self._create_output_filename(row)
                    output_path = self.output_dir / output_filename
                    
                    # Skip if output already exists
                    if output_path.exists():
                        logger.debug(f"Segment already exists: {output_filename}")
                        self.stats['successful_segments'] += 1
                        continue
                    
                    # Segment the video
                    success = self._segment_video(
                        video_path,
                        row['START_REALIGNED'],
                        row['END_REALIGNED'],
                        output_path
                    )
                    
                    if success:
                        self.stats['successful_segments'] += 1
                        logger.debug(f"✓ Created: {output_filename}")
                    else:
                        self.stats['failed_segments'] += 1
                        logger.error(f"✗ Failed: {output_filename}")
                        
                except Exception as e:
                    self.stats['processing_errors'] += 1
                    logger.error(f"Error processing segment {idx}: {e}")
        
        return self.stats
    
    def save_metadata(self, output_file: str = "segmentation_metadata_train.json"):
        """Save metadata about the segmentation process"""
        metadata = {
            'input_csv': str(self.input_csv),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'statistics': self.stats,
            'total_videos_processed': len(self.df['VIDEO_NAME'].unique()),
            'total_segments': len(self.df)
        }
        
        metadata_path = self.output_dir / output_file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Segment training videos based on CSV timing data")
    parser.add_argument("--input-csv", 
                       default=r"how2sign_realigned_train.csv",
                       help="Path to input CSV file (default: how2sign_realigned_train.csv)")
    parser.add_argument("--input-dir", 
                       default=r"C:\Projects\Sign-language\train_raw_videos\raw_videos",
                       help="Directory containing input videos")
    parser.add_argument("--output-dir", 
                       default=r"D:\how2sign_train_segment_clips",
                       help="Directory for output clips")
    parser.add_argument("--max-clips", 
                       type=int, 
                       help="Maximum number of clips to process (for testing)")
    parser.add_argument("--video-filter", 
                       help="Filter to process only specific videos (e.g., '-fZc293MpJk')")
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.input_csv):
        logger.error(f"CSV file not found: {args.input_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    try:
        # Initialize segmenter
        segmenter = VideoSegmenter(args.input_csv, args.input_dir, args.output_dir)
        
        # Process segments
        stats = segmenter.process_segments(args.max_clips, args.video_filter)
        
        # Save metadata
        segmenter.save_metadata()
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("SEGMENTATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total segments: {stats['total_segments']}")
        logger.info(f"Successful: {stats['successful_segments']}")
        logger.info(f"Failed: {stats['failed_segments']}")
        logger.info(f"Missing videos: {stats['missing_videos']}")
        logger.info(f"Processing errors: {stats['processing_errors']}")
        if stats['total_segments'] > 0:
            logger.info(f"Success rate: {stats['successful_segments']/stats['total_segments']*100:.1f}%")
        logger.info(f"Output directory: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

