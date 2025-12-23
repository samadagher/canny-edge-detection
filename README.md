# Canny Edge Detection & Frequency Filtering

Real-time webcam application for edge detection and frequency domain filtering.

## Features
- Canny edge detection
- Sobel edge detection (X and Y)
- Ideal, Gaussian, and Butterworth filters (LPF & HPF)
- Adjustable parameters (D0, Butterworth order)

## Controls
- `o` - Original image
- `i/I` - Ideal LPF/HPF
- `g/G` - Gaussian LPF/HPF
- `b/B` - Butterworth LPF/HPF
- `c` - Canny edges
- `x/y` - Sobel X/Y
- `+/-` - Adjust D0
- `1-9` - Set Butterworth order
- `q` - Quit

## Requirements
```bash
pip install opencv-python numpy
```

## Usage
```bash
python main.py
```

## Project Structure
- `main.py` - Main webcam application
- `canny.py` - Canny edge detection implementation + Frequency domain filters (ILPF, GLPF, BLPF, etc.)
