# Digital Scale Analyzer

A Python application for analyzing digital scale readings from images. This tool can process images of digital scale displays to extract values, with support for both image file input and direct camera capture.

## Features

- Image file import and camera capture support
- Real-time image preview and adjustment
- Automatic value detection
- Image calibration tools
- Batch processing capabilities
- Data persistence

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/digital-scale-analyzer.git
cd digital-scale-analyzer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python scale_analyzer.py
```

### Basic Usage

1. Click "Add Image" to import scale images
2. Select an image from the list
3. Click "Analyze Selected" to process the image
4. View the detected value

### Advanced Features

- Use the preview window to adjust image settings
- Calibrate the scale using known reference points
- Capture images directly from a connected camera
- Batch process multiple images

## License

This project is licensed under the MIT License - see the LICENSE file for details. 