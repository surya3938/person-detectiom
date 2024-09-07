# Person Detection, Tracking, and Emotion Detection

## Overview
This project detects and tracks individuals (specifically children and therapists) in video footage, assigns unique IDs, tracks re-entries, and handles occlusions. Additionally, it uses facial recognition to detect and display the emotions of the individuals.

The project is designed to identify children with Autism Spectrum Disorder (ASD) and therapists to analyze their behaviors, emotions, and engagement levels.

### Features
- *Person Detection*: Detects individuals in the video using the YOLOv8 model.
- *Tracking*: Tracks the detected individuals across frames using the Deep SORT algorithm, which handles re-entries and post-occlusion tracking.
- *Emotion Detection*: Uses the FER (Facial Emotion Recognition) model to detect the emotions of the individuals based on facial expressions.
- *Optimized Processing*: Emotion detection runs every 5 frames to reduce computational overhead.

## Prerequisites

Before running the code, ensure you have the following:
- Python 3.8+ installed on your system.
- Necessary libraries and dependencies installed (instructions below).

## Setup and Installation

1. *Clone the Repository*

   Clone or download the project to your local system:

   ```bash
   git clone https://github.com/your-repo/person-detection-tracking.git
   cd person-detection-tracking
