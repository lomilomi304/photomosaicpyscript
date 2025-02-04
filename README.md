This script transforms a source image into a mosaic by:
1. Breaking down the main image into a grid
2. Analyzing the color patterns of each grid cell
3. Matching each cell with the most suitable tile from an image collection
4. Assembling these tiles to create a mosaic version of the original image

required packages:
[python3 -m venv 'name'] -> source 'name'/bin/activate
(pip install Pillow numpy tqdm scipy)

### Command Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|----------|
| main_image | Source image path | Required | Any PNG/JPG file |
| tile_folder | Folder with tile images | Required | Directory path |
| --output | Output image path | mosaic.png | Any PNG filename |
| --tile-width | Width of each tile | 50 | Positive integer |
| --tile-height | Height of each tile | 50 | Positive integer |
| --max-reuse | Max times a tile can be used | 3 | Positive integer |
| --shape | Shape of the tiles | rectangle | rectangle, circle, diamond |
| --threads | Number of processing threads | 4 | Positive integer |

eg: python mosaic.py main_image.png photos --tile-width 30  --tile-height 30 --max-reuse 2 --shape rectangle --threads 2 --output custom_mosaic.png


## Image Dataset

Used [Natural Images dataset from Kaggle](https://www.kaggle.com/datasets/prasunroy/natural-images?resource=download),for photos


### Color Matching Process
The script uses sophisticated color matching techniques:
1. **K-d Tree Implementation**: Efficient spatial searching for color matching
2. **Dual Color Space**: 
   - RGB for initial matching
   - HSV for fine-tuning and better color perception
3. **Nearest Neighbor Search**: Finds the k closest color matches for each tile

### Image Processing

- Uses PIL (Python Imaging Library) for image manipulation
- Supports alpha channel for transparent tiles
- Implements Lanczos resampling for high-quality resizing


