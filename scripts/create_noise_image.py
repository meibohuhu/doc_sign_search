#!/usr/bin/env python3
"""
Create a total noise image for testing attention visualization.
Generates an image with random noise (uniform or Gaussian).

Usage:
    python scripts/create_noise_image.py --output noise_image.png --width 224 --height 224
    python scripts/create_noise_image.py --output noise_image.png --width 224 --height 224 --noise_type gaussian
"""

import numpy as np
import cv2
import argparse
from PIL import Image


def create_uniform_noise_image(width, height, channels=3):
    """Create image with uniform random noise [0, 255]"""
    noise = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
    return noise


def create_gaussian_noise_image(width, height, channels=3, mean=128, std=50):
    """Create image with Gaussian random noise"""
    noise = np.random.normal(mean, std, size=(height, width, channels))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return noise


def create_black_image(width, height, channels=3):
    """Create completely black image (all zeros)"""
    return np.zeros((height, width, channels), dtype=np.uint8)


def create_white_image(width, height, channels=3):
    """Create completely white image (all 255)"""
    return np.ones((height, width, channels), dtype=np.uint8) * 255


def create_checkerboard_image(width, height, channels=3, square_size=10):
    """Create checkerboard pattern"""
    img = np.zeros((height, width, channels), dtype=np.uint8)
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = 255
    return img


def main():
    parser = argparse.ArgumentParser(description='Create noise image for testing')
    parser.add_argument('--output', type=str, default='noise_image.png', 
                       help='Output image path')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--noise_type', type=str, default='uniform',
                        choices=['uniform', 'gaussian', 'black', 'white', 'checkerboard'],
                        help='Type of noise/image to generate')
    parser.add_argument('--channels', type=int, default=3, choices=[1, 3],
                        help='Number of channels (1=grayscale, 3=RGB)')
    parser.add_argument('--gaussian_mean', type=float, default=128,
                        help='Mean for Gaussian noise (0-255)')
    parser.add_argument('--gaussian_std', type=float, default=50,
                        help='Standard deviation for Gaussian noise')
    parser.add_argument('--checkerboard_size', type=int, default=10,
                        help='Square size for checkerboard pattern')
    
    args = parser.parse_args()
    
    print(f"🎨 Creating {args.noise_type} noise image...")
    print(f"   Size: {args.width}x{args.height}")
    print(f"   Channels: {args.channels}")
    
    # Create image based on noise type
    if args.noise_type == 'uniform':
        img = create_uniform_noise_image(args.width, args.height, args.channels)
        print(f"   Uniform random noise [0, 255]")
    elif args.noise_type == 'gaussian':
        img = create_gaussian_noise_image(args.width, args.height, args.channels, 
                                         args.gaussian_mean, args.gaussian_std)
        print(f"   Gaussian noise (mean={args.gaussian_mean}, std={args.gaussian_std})")
    elif args.noise_type == 'black':
        img = create_black_image(args.width, args.height, args.channels)
        print(f"   Black image (all zeros)")
    elif args.noise_type == 'white':
        img = create_white_image(args.width, args.height, args.channels)
        print(f"   White image (all 255)")
    elif args.noise_type == 'checkerboard':
        img = create_checkerboard_image(args.width, args.height, args.channels, 
                                       args.checkerboard_size)
        print(f"   Checkerboard pattern (square_size={args.checkerboard_size})")
    
    # Convert grayscale if needed
    if args.channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img[:, :, 0:1]
        img = img.squeeze()
    
    # Save image
    if args.output.endswith('.png'):
        Image.fromarray(img).save(args.output)
    else:
        cv2.imwrite(args.output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if args.channels == 3 else img)
    
    # Print statistics
    if args.channels == 3:
        print(f"   Image stats:")
        print(f"      Min: {img.min()}, Max: {img.max()}")
        print(f"      Mean: {img.mean():.2f}, Std: {img.std():.2f}")
        print(f"      Mean per channel: R={img[:,:,0].mean():.2f}, G={img[:,:,1].mean():.2f}, B={img[:,:,2].mean():.2f}")
    else:
        print(f"   Image stats:")
        print(f"      Min: {img.min()}, Max: {img.max()}")
        print(f"      Mean: {img.mean():.2f}, Std: {img.std():.2f}")
    
    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    main()


