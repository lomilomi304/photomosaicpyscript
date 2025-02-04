from PIL import Image
import numpy as np
import os
from pathlib import Path
import colorsys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random
from scipy.spatial import cKDTree

@dataclass
class TileImage:
    """Class to store tile image and its color properties."""
    image: Image.Image
    hsv: Tuple[float, float, float]  # Average HSV values
    rgb: Tuple[float, float, float]  # Average RGB values
    uses: int = 0  # Track number of times tile is used

def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert RGB values to HSV."""
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s * 100, v * 100

def get_average_colors(img: Image.Image) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Calculate both the average RGB and HSV values of an image."""
    img_array = np.array(img)
    r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
    r, g, b = r/255.0, g/255.0, b/255.0
    
    # Calculate average RGB
    avg_r = float(np.mean(r))
    avg_g = float(np.mean(g))
    avg_b = float(np.mean(b))
    
    # Convert to HSV
    hsv = rgb_to_hsv(avg_r, avg_g, avg_b)
    
    return (avg_r, avg_g, avg_b), hsv

def process_tile_image(file_path: Path, tile_size: Tuple[int, int], shape: str) -> Optional[TileImage]:
    """Process a single tile image with the specified shape."""
    try:
        with Image.open(file_path).convert('RGB') as img:
            # Resize image
            img = img.resize(tile_size, Image.Resampling.LANCZOS)
            
            # Create mask for different shapes
            mask = Image.new('L', tile_size, 0)
            if shape == 'circle':
                cx, cy = tile_size[0] // 2, tile_size[1] // 2
                radius = min(cx, cy)
                for x in range(tile_size[0]):
                    for y in range(tile_size[1]):
                        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                        mask.putpixel((x, y), 255 if dist <= radius else 0)
            elif shape == 'diamond':
                cx, cy = tile_size[0] // 2, tile_size[1] // 2
                for x in range(tile_size[0]):
                    for y in range(tile_size[1]):
                        if abs(x - cx) + abs(y - cy) <= min(cx, cy):
                            mask.putpixel((x, y), 255)
            else:  # rectangle (default)
                mask = Image.new('L', tile_size, 255)
            
            # Apply mask
            img.putalpha(mask)
            
            # Calculate color properties
            rgb, hsv = get_average_colors(img)
            return TileImage(img, hsv, rgb)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_photo_mosaic(
    main_image_path: str,
    tile_folder_path: str,
    tile_size: Tuple[int, int] = (50, 50),
    output_path: str = 'mosaic.png',
    max_tile_reuse: int = 3,
    shape: str = 'rectangle',
    num_threads: int = 4
) -> Image.Image:
    """Create a photo mosaic with enhanced features using KD-Tree for efficient color matching."""
    # Load and resize main image
    main_image = Image.open(main_image_path)
    
    # Calculate dimensions
    width, height = main_image.size
    tile_width, tile_height = tile_size
    num_tiles_x = width // tile_width
    num_tiles_y = height // tile_height
    
    # Resize main image to fit tile grid (pixelation step)
    main_image = main_image.resize((num_tiles_x, num_tiles_y), Image.Resampling.LANCZOS)
    main_image = main_image.convert('RGB')
    
    # Create output image with alpha channel
    output_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Process tile images in parallel
    print("Processing tile images...")
    tile_paths = list(Path(tile_folder_path).glob('*'))
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_path = {
            executor.submit(process_tile_image, path, tile_size, shape): path
            for path in tile_paths
            if path.suffix.lower() in ['.png', '.jpg', '.jpeg']
        }
        
        tiles = []
        for future in tqdm(future_to_path, desc="Loading tiles"):
            tile = future.result()
            if tile is not None:
                tiles.append(tile)
    
    if not tiles:
        raise ValueError("No valid images found in tile folder")
    
    print(f"Processed {len(tiles)} tile images")
    
    # Create KD-Tree for RGB color matching
    rgb_colors = np.array([tile.rgb for tile in tiles])
    kdtree = cKDTree(rgb_colors)
    
    # Convert main image to numpy array for faster processing
    main_array = np.array(main_image)
    
    # Process each cell in the main image
    print("Creating mosaic...")
    for y in tqdm(range(num_tiles_y), desc="Rows completed"):
        for x in range(num_tiles_x):
            # Get the pixel color from the resized (pixelated) main image
            pixel_color = main_array[y, x] / 255.0
            
            # Find k nearest neighbors using KD-Tree
            k = min(10, len(tiles))  # Get top 10 matches or less if fewer tiles
            distances, indices = kdtree.query(pixel_color, k=k)
            
            # Select the least used tile among the top matches
            best_idx = min(
                indices,
                key=lambda i: (tiles[i].uses, distances[list(indices).index(i)])
            )
            
            chosen_tile = tiles[best_idx]
            chosen_tile.uses += 1
            
            # If tile is overused, consider resetting usage counts
            if chosen_tile.uses >= max_tile_reuse:
                min_uses = min(tile.uses for tile in tiles)
                if min_uses >= max_tile_reuse:
                    for tile in tiles:
                        tile.uses = 0
            
            # Calculate position in output image
            box = (x * tile_width, y * tile_height,
                  (x + 1) * tile_width, (y + 1) * tile_height)
            
            # Place matching tile
            output_image.paste(chosen_tile.image, box, chosen_tile.image)
    
    # Save the result
    output_image.save(output_path, "PNG")
    return output_image

def main():
    """Main function with example usage and parameter options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a photo mosaic')
    parser.add_argument('main_image', help='Path to the main image')
    parser.add_argument('tile_folder', help='Path to the folder containing tile images')
    parser.add_argument('--output', default='mosaic.png', help='Output file path')
    parser.add_argument('--tile-width', type=int, default=50, help='Tile width in pixels')
    parser.add_argument('--tile-height', type=int, default=50, help='Tile height in pixels')
    parser.add_argument('--max-reuse', type=int, default=3, help='Maximum times a tile can be reused')
    parser.add_argument('--shape', choices=['rectangle', 'circle', 'diamond'], default='rectangle',
                        help='Shape of the tiles')
    parser.add_argument('--threads', type=int, default=4, help='Number of processing threads')
    
    args = parser.parse_args()
    
    try:
        mosaic = create_photo_mosaic(
            args.main_image,
            args.tile_folder,
            tile_size=(args.tile_width, args.tile_height),
            output_path=args.output,
            max_tile_reuse=args.max_reuse,
            shape=args.shape,
            num_threads=args.threads
        )
        print(f"Mosaic created successfully and saved to {args.output}")
    except Exception as e:
        print(f"Error creating mosaic: {e}")

if __name__ == "__main__":
    main()