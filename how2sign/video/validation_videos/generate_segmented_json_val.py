#!/usr/bin/env python3
"""
Generate JSON for Segmented Validation Video Clips with Question Diversity
===========================================================================

This script generates a JSON file similar to segmented_train_videos.json but for validation segmented video clips.
It matches the segmented videos with their corresponding sentences from the CSV file.

Features:
- Uses 5 different question variations for diversity (randomly selected for each entry)
- Supports custom random seed for reproducibility

Usage:
    python generate_segmented_json_val.py [--input-csv CSV_FILE] [--video-dir VIDEO_DIR] [--output-json OUTPUT_FILE] [--seed SEED]

Author: Generated for Sign Language LLM project
Date: 2025
"""

import os
import sys
import pandas as pd
import json
import argparse
from pathlib import Path
import logging
import random
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generate_segmented_json_val.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SegmentedJSONGenerator:
    """Class to generate JSON file for segmented video clips"""
    
    def __init__(self, input_csv: str, video_dir: str, output_json: str):
        """
        Initialize the JSON generator
        
        Args:
            input_csv: Path to the CSV file with segmentation data
            video_dir: Directory containing the segmented video clips
            output_json: Path to the output JSON file
        """
        self.input_csv = input_csv
        self.video_dir = Path(video_dir)
        self.output_json = output_json
        
        # Load CSV data
        self.df = self._load_csv()
        
        # Statistics
        self.stats = {
            'total_videos_found': 0,
            'total_videos_matched': 0,
            'total_videos_missing': 0,
            'total_entries_created': 0,
            'question_distribution': {}
        }
    
    def _load_csv(self) -> pd.DataFrame:
        """Load and validate the CSV file"""
        try:
            df = pd.read_csv(self.input_csv, sep='\t')
            logger.info(f"Loaded CSV with {len(df)} rows")
            
            # Validate required columns
            required_columns = ['SENTENCE_NAME', 'SENTENCE']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Valid data: {len(df)} segments")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def _get_video_files(self) -> List[Path]:
        """Get list of video files in the directory"""
        if not self.video_dir.exists():
            logger.error(f"Video directory not found: {self.video_dir}")
            return []
        
        video_files = list(self.video_dir.glob("*.mp4"))
        logger.info(f"Found {len(video_files)} video files in {self.video_dir}")
        return video_files
    
    def _create_json_entry(self, video_file: Path, sentence: str) -> tuple[Dict[str, Any], str]:
        """
        Create a JSON entry for a video file
        
        Args:
            video_file: Path to the video file
            sentence: The sentence/transcript for this video
            
        Returns:
            Tuple of (JSON entry dictionary, selected question)
        """
        # Use the video filename (without extension) as the ID
        video_id = video_file.stem
        
        # Define 5 different question variations for diversity
        questions = [
            "<video>\nTranslate the American Sign Language in this video to English.",
            "<video>\nProvide the English translation of this ASL video.",
            "<video>\nWhat is being signed in American Sign Language in this video?",
            "<video>\nTranslate this ASL video to English.",
            "<video>\nConvert this American Sign Language video to English text."
        ]
        
        # Randomly select one question
        selected_question = random.choice(questions)
        
        entry = {
            "id": video_id,
            "video": video_file.name,
            "conversations": [
                {
                    "from": "human",
                    "value": selected_question
                },
                {
                    "from": "gpt",
                    "value": sentence
                }
            ]
        }
        
        return entry, selected_question
    
    def generate_json(self) -> List[Dict[str, Any]]:
        """
        Generate the JSON data for all segmented videos
        
        Returns:
            List of dictionaries representing the JSON entries
        """
        logger.info("Starting JSON generation process...")
        
        # Get all video files
        video_files = self._get_video_files()
        self.stats['total_videos_found'] = len(video_files)
        
        if not video_files:
            logger.error("No video files found!")
            return []
        
        # Create a mapping from video filename to sentence
        video_to_sentence = {}
        for _, row in self.df.iterrows():
            sentence_name = row['SENTENCE_NAME']
            sentence = row['SENTENCE']
            
            # Remove .mp4 extension if present for matching
            clean_name = sentence_name.replace('.mp4', '')
            video_to_sentence[clean_name] = sentence
        
        logger.info(f"Created mapping for {len(video_to_sentence)} sentences")
        
        # Generate JSON entries
        json_entries = []
        matched_videos = 0
        missing_videos = 0
        
        for video_file in video_files:
            video_name = video_file.stem
            
            if video_name in video_to_sentence:
                sentence = video_to_sentence[video_name]
                entry, question = self._create_json_entry(video_file, sentence)
                json_entries.append(entry)
                matched_videos += 1
                
                # Track question distribution
                if question not in self.stats['question_distribution']:
                    self.stats['question_distribution'][question] = 0
                self.stats['question_distribution'][question] += 1
                
                logger.debug(f"✓ Matched: {video_file.name} -> {sentence[:50]}...")
            else:
                missing_videos += 1
                logger.warning(f"✗ No sentence found for: {video_file.name}")
        
        # Update statistics
        self.stats['total_videos_matched'] = matched_videos
        self.stats['total_videos_missing'] = missing_videos
        self.stats['total_entries_created'] = len(json_entries)
        
        logger.info(f"Generated {len(json_entries)} JSON entries")
        return json_entries
    
    def save_json(self, json_data: List[Dict[str, Any]]) -> None:
        """Save the JSON data to file"""
        try:
            with open(self.output_json, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON file saved to: {self.output_json}")
            
        except Exception as e:
            logger.error(f"Error saving JSON file: {e}")
            raise
    
    def print_summary(self) -> None:
        """Print processing summary"""
        logger.info("\n" + "="*50)
        logger.info("JSON GENERATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total videos found: {self.stats['total_videos_found']}")
        logger.info(f"Videos matched: {self.stats['total_videos_matched']}")
        logger.info(f"Videos missing: {self.stats['total_videos_missing']}")
        logger.info(f"JSON entries created: {self.stats['total_entries_created']}")
        if self.stats['total_videos_found'] > 0:
            logger.info(f"Success rate: {self.stats['total_videos_matched']/self.stats['total_videos_found']*100:.1f}%")
        logger.info(f"Output file: {self.output_json}")
        
        # Print question distribution
        if self.stats['question_distribution']:
            logger.info("\n" + "="*50)
            logger.info("QUESTION DISTRIBUTION")
            logger.info("="*50)
            for i, (question, count) in enumerate(sorted(self.stats['question_distribution'].items()), 1):
                # Extract just the question part (after <video>\n)
                question_text = question.split('\n', 1)[1] if '\n' in question else question
                percentage = (count / self.stats['total_entries_created'] * 100) if self.stats['total_entries_created'] > 0 else 0
                logger.info(f"Q{i}: {count:>6} ({percentage:>5.1f}%) - {question_text}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate JSON for segmented validation video clips")
    parser.add_argument("--input-csv", 
                       default="how2sign_realigned_val.csv",
                       help="Path to input CSV file (default: how2sign_realigned_val.csv)")
    parser.add_argument("--video-dir", 
                       default=r"D:\how2sign_val_segment_clips",
                       help="Directory containing segmented videos (default: D:\\how2sign_val_segment_clips)")
    parser.add_argument("--output-json", 
                       default="segmented_val_videos.json",
                       help="Output JSON file (default: segmented_val_videos.json)")
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--seed", 
                       type=int,
                       default=42,
                       help="Random seed for question selection (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.input_csv):
        logger.error(f"CSV file not found: {args.input_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.video_dir):
        logger.error(f"Video directory not found: {args.video_dir}")
        sys.exit(1)
    
    try:
        # Initialize generator
        generator = SegmentedJSONGenerator(args.input_csv, args.video_dir, args.output_json)
        
        # Generate JSON data
        json_data = generator.generate_json()
        
        if not json_data:
            logger.error("No JSON data generated!")
            sys.exit(1)
        
        # Save JSON file
        generator.save_json(json_data)
        
        # Print summary
        generator.print_summary()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

