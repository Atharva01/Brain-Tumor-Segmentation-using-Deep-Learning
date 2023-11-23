# Brain Tumor Segmentation Project

## Overview
This project focuses on the segmentation of brain tumors using deep learning techniques, specifically the U-Net architecture. The goal is to develop a model capable of accurately identifying and delineating tumors in medical images.

## Project Structure
The project is organized into several modules, each serving a specific purpose:

- **data_processing.py:** Contains functions for loading, preprocessing, and augmenting data.
- **model.py:** Defines the U-Net neural network architecture.
- **train.py:** Includes the training script for the model.
- **evaluate.py:** Includes the evaluation script for the trained model.
- **pipeline.py:** Orchestrates the entire workflow, from data processing to training and evaluation.

## Getting Started
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Data Setup**
   Place your training data in the data/BRATS2015_Training directory.
   Place your evaluation data in the data/Evaluation directory.
4. **Run the pipeline**
   ```bash
   python src/pipeline.py
   ```
5. **This project assumes a U-Net architecture for brain tumor segmentation.**
