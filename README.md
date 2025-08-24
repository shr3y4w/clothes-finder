# Clothes Finder and Virtual Try-On App

A Flask-based web application that uses  computer vision to find similar clothing items and provides real-time virtual try-on capabilities.

## Features

### Core Functionality
- Upload clothing images and find similar items using Color, SIFT, and LBP feature descriptors
- Real-Time Virtual Try-On
- Image preprocessing with morphological operations
- In-memory caching system for instant repeated queries



### Performance Optimizations
- Vectorized Similarity Search: Fast numpy-based distance calculations
- Limited to top 50 most relevant results for better performance
- 200-result cache with automatic cleanup

## Technology Stack

### Backend
- **Flask** 
- **OpenCV, scikit-image** - Computer vision and image processing
- **MediaPipe** - Pose detection for virtual try-on
- **NumPy/SciPy** - Distance calculations

### Frontend
- **HTML5/CSS3**
- **JavaScript**
- **Font Awesome** 

### Computer Vision
- **SIFT Features** - Scale-invariant feature transform for texture analysis
- **Color Histograms** - HSV color space analysis
- **Local Binary Patterns (LBP)** - Texture pattern recognition
- **Background Removal** - Morphological operations


## To use

### Prerequisites
- **Python 3.8+**
- **Webcam** (for virtual try-on)
- **Browser** with camera permissions

### Installation

1. **Clone and navigate**
   ```bash
   git clone <clothes-finder>
   cd clothes-finder
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare dataset**
   ```bash
   mkdir -p dataset/cloth
   ```
   - Create dataset directory
   - Add your clothing images (JPG, PNG)
   - Image features are extracted on first run

5. **Run the application**
   ```bash
   python app.py
   ```


## Configuration

### Feature Weights 
```python
color_weight = 0.8 
sift_weight = 0.5    
lbp_weight = 0.6     
```

### Cache Settings
```python
CACHE_SIZE_LIMIT = 200 
```

### Performance Tuning
- **Reduce dataset size** for faster initial load
- **Adjust feature weights** for different clothing types