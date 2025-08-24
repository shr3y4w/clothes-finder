# Dataset Directory

This directory should contain your clothing images for the similarity search functionality.

## Structure

```
dataset/
├── cloth/           # Place your clothing images here
│   ├── 00000_00.jpg
│   ├── 00001_00.jpg
│   └── ...
└── styles.csv       # Optional: metadata about clothing items
```

## Supported Image Formats

- JPG/JPEG
- PNG
- GIF

## Adding Images

1. Place your clothing images in the `cloth/` subdirectory
2. Use descriptive filenames for better organization
3. Ensure images are clear and well-lit for better feature extraction
4. Recommended image size: 256x256 pixels or larger

## Notes

- The system will automatically extract features from all images in the `cloth/` directory
- Feature extraction happens on first run and is cached in `features.pkl`
- Large datasets may take significant time to process initially
- Images with transparent backgrounds work best for virtual try-on

## Example Dataset

You can use any clothing dataset such as:
- Fashion-MNIST
- DeepFashion
- Custom clothing collections

Make sure you have the rights to use any images you include in your dataset.
