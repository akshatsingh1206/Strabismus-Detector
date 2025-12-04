# Strabismus Detection Using Deep Learning

A deep-learning-based system that detects strabismus (eye misalignment) using U-Net segmentation and geometric feature analysis. The project includes a Streamlit web app that allows users to upload eye images, view segmentation masks, compute alignment metrics, and download a diagnostic report.

## Overview
This project uses two U-Net models to segment the pupil and sclera from eye images. After segmentation, the system extracts geometric features including pupil center, sclera center, eye boundaries, offset differences, and gaze deviation. Based on these measurements, the system determines whether strabismus is present and provides alignment severity information.

## Core Features
- U-Net based pupil and sclera segmentation
- Automatic geometric feature extraction
- Misalignment detection using offset differences
- Visual overlays showing centers, edges, and masks
- Downloadable report with metrics
- Streamlit interface for easy use
- Supports image upload and real-time processing

## Project Structure
Strabismus-Detector/
- app.py                → Streamlit application
- pupilggg.h5           → Trained U-Net model for pupil segmentation
- iris_ggg.h5           → Trained U-Net model for sclera segmentation
- requirements.txt      → Dependencies
- .gitattributes        → Git LFS tracking rules
- .gitignore
- README.md

## How It Works
1. User uploads left and right eye images.
2. Images are preprocessed (grayscale, resize).
3. U-Net models generate pupil and sclera masks.
4. Contours and centroids are extracted.
5. Alignment difference is calculated:
   diff = | left_offset - right_offset |
6. If the difference exceeds ~8% of eye length, strabismus is flagged.
7. A visual overlay and a text-based report are generated.

## Installation & Setup

### Step 1: Clone the repository
git clone https://github.com/yourusername/Strabismus-Detector.git
cd Strabismus-Detector

### Step 2: Install dependencies
pip install -r requirements.txt

### Step 3: Run the application
streamlit run app.py

## Model Files
Git LFS is used to store large .h5 model files.  
If cloning this repo, run:
git lfs install
git pull

The U-Net models will automatically download.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- Git LFS

## Applications
- Telemedicine screening
- School eye-check programs
- Clinical decision support
- Early misalignment detection
- Low-cost ophthalmic tools

## Limitations
- Sensitive to low-quality images
- Works only on 2D static images
- Accuracy depends on dataset quality
- No depth or video tracking

## Future Enhancements
- Real-time video-based analysis
- Mobile app deployment
- 3D gaze estimation
- Larger training dataset
- Explainable AI (Grad-CAM)

## Contributor
- Akshat Singh

## License
All Rights Reserved.
