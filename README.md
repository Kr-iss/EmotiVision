# EmotiVision 
### Real-Time Face Recognition & Facial Expression Recognition System

EmotiVision is a Python-based computer vision application that combines **face recognition** and **facial expression recognition** in a single desktop interface. It uses a webcam to detect faces, identify registered individuals, and estimate facial expressions in real time.

The project demonstrates how computer vision and machine learning techniques can be used to build an interactive facial analysis system.

---

## Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Algorithms Used](#-algorithms-used)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Privacy and Data Protection](#-privacy-and-data-protection)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [References](#-references)
- [Author](#-author)

---

## Overview

EmotiVision integrates two computer vision tasks:

1. **Face Recognition:** Identifies registered individuals by comparing detected faces against a locally trained recognition model.
2. **Facial Expression Recognition:** Estimates facial expressions such as happiness, sadness, anger, fear, surprise, disgust, and neutrality using a pretrained model.

The application provides a graphical user interface (GUI) through which users can register faces, train the recognition model, and perform live webcam recognition.

EmotiVision is designed as an educational project to demonstrate the practical application of computer vision, machine learning, image processing, and graphical user interface development.

## Problem Statement

Identifying registered individuals and estimating facial expressions are two different computer vision tasks that can be challenging to perform manually and consistently.

The objective of EmotiVision is to develop an integrated desktop application that uses a webcam to detect faces, recognise previously registered individuals, and estimate their visible facial expressions in real time.

The system combines traditional computer vision techniques with a pretrained facial-expression recognition model to demonstrate an accessible approach to real-time facial analysis.

## Objectives

- Develop a real-time face detection system using a webcam.
- Register individuals by capturing multiple face images.
- Train a face recognition model using registered images.
- Identify known individuals and label unrecognised faces as "Unknown".
- Estimate facial expressions using a pretrained deep learning model.
- Develop an interactive desktop application using Tkinter.
- Store face data and trained models locally.
- Understand the strengths and limitations of facial recognition and expression classification.

## Features

- **Live Webcam Recognition:** Processes webcam frames in real time.
- **Face Detection:** Detects faces in camera frames.
- **Face Registration:** Captures multiple face samples for each registered person.
- **LBPH Model Training:** Trains and saves a local face recognition model.
- **Identity Recognition:** Matches detected faces against registered identities.
- **Unknown Face Detection:** Displays "Unknown" when a face does not meet the configured recognition threshold.
- **Expression Estimation:** Estimates facial expressions using the FER library.
- **Graphical User Interface:** Provides an interactive interface built with Tkinter.
- **Model Reloading:** Loads a previously saved recognition model.
- **Local Data Storage:** Keeps registered face images and recognition data on the local computer.
- **Algorithm Assessment:** Provides a qualitative comparison of selected face recognition algorithms.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image processing, webcam access, face detection, and recognition |
| NumPy | Numerical operations and image array handling |
| Pillow (PIL) | Image conversion and display in the GUI |
| Tkinter | Desktop graphical user interface |
| FER | Facial expression recognition |
| LBPH | Face recognition algorithm |
| Haar Cascade | Face detection |
| Git and GitHub | Version control and project hosting |

## Algorithms Used

### 1. Haar Cascade Classifier

Haar Cascade is used for face detection. It identifies potential face regions in an image using learned visual features and a cascade of classifiers.

In EmotiVision, the detector locates faces in webcam frames before the recognition and expression estimation stages.

### 2. Local Binary Patterns Histograms (LBPH)

LBPH is the implemented face recognition algorithm in EmotiVision.

It describes local texture patterns in grayscale face images and builds histograms that can be used to distinguish registered individuals.

The application trains an LBPH recogniser using the registered face samples and saves the resulting model locally.

During live recognition, the system compares detected face regions against the trained model. The configured threshold determines whether a prediction is accepted or displayed as "Unknown".

**Important:** The LBPH prediction distance is not a probability and should not be interpreted as a percentage of identity confidence.

### 3. FER — Facial Expression Recognition

EmotiVision uses the FER Python library to estimate facial expressions.

The application uses a pretrained model through the FER library to classify visible facial-expression patterns.

The supported expression categories are:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The model returns expression labels and associated scores.

These outputs are estimates based on visible facial patterns. They do not reliably establish a person's actual emotions, intentions, or psychological state.

### 4. Other Face Recognition Algorithms Considered

The application includes a qualitative assessment of other face recognition approaches:

| Algorithm | Main concept | Role in this project |
|---|---|---|
| Eigenfaces | Principal Component Analysis (PCA) | Theoretical comparison |
| Fisherfaces | Linear Discriminant Analysis (LDA) | Theoretical comparison |
| LBPH | Local facial texture patterns | Implemented algorithm |
| Deep Face Embeddings | Deep learning representations of faces | Theoretical comparison |

Eigenfaces, Fisherfaces, and deep face embeddings are included for comparison and are not implemented as alternative recognition models in the current application.

## How It Works

The application follows this general workflow:

1. **Webcam Initialisation:** OpenCV accesses the webcam and captures video frames.
2. **Frame Processing:** Frames are prepared for face detection, including conversion to grayscale where appropriate.
3. **Face Detection:** Haar Cascade identifies potential face regions.
4. **Face Registration:** When a person is registered, the application captures multiple face samples and saves them locally.
5. **Model Training:** The LBPH recogniser learns from the saved face samples and stores the trained model.
6. **Identity Recognition:** During live operation, detected faces are compared with the trained model.
7. **Unknown Classification:** Faces that do not meet the configured acceptance threshold are labelled "Unknown".
8. **Expression Estimation:** Detected face regions are processed by the FER model to estimate facial expressions.
9. **GUI Display:** The application displays the camera feed, detected faces, identity labels, and expression estimates.

Face recognition and facial expression recognition are separate processes. Recognising a person's identity does not determine their emotional state.

## Project Structure

The repository is intended to contain the source code and documentation, while private face data and trained files remain on the local computer.

```text
EmotiVision/
│
├── emotivision_fixed.py
├── haarcascade_frontalface_default.xml
├── README.md
├── requirements.txt
├── EmotiVision_Project_Report.docx
│
└── .gitignore
```

The following files and folders may exist locally but should not be uploaded:

```text
face_data/                 # Registered face images
emotivision_model.yml      # Trained LBPH model
labels.json                # Local identity labels
contacts.csv               # Personal contact information
.venv/                     # Local Python environment
```

The exact local filenames depend on the configuration used when running the application.

## Installation

### Prerequisites

- Python installed on your computer
- Visual Studio Code or another Python editor
- A working webcam
- Git, if you want to clone the repository
- A compatible OpenCV installation with the `cv2.face` module available

### 1. Clone the Repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/EmotiVision.git
cd EmotiVision
```

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

If a `requirements.txt` file is available, run:

```bash
python -m pip install -r requirements.txt
```

The project uses Python libraries including OpenCV, NumPy, Pillow, and FER.

**Note:** OpenCV must include the face recognition module used by `cv2.face.LBPHFaceRecognizer_create()`. Depending on your environment, this may require a compatible `opencv-contrib-python` installation rather than the standard OpenCV package. Avoid installing conflicting OpenCV package variants in the same environment.

The FER package and its dependencies must also be compatible with your Python version.

## How to Run

1. Open the project folder in Visual Studio Code.
2. Activate the project's virtual environment.
3. Confirm that the required dependencies are installed.
4. Confirm that the Haar Cascade XML file is available where the application expects it.
5. Run the application:

```bash
python emotivision_fixed.py
```

6. Allow webcam access if your operating system requests permission.
7. Use the application interface to register faces, train the model, and start live recognition.

### Registering a Person

- Select the face registration option.
- Enter the requested information.
- Allow the application to capture the face samples.
- Train or retrain the LBPH model after registration, as required.

### Running Recognition

- Start the camera.
- The application detects faces in the video feed.
- Registered faces may be identified by name.
- Unrecognised faces may be labelled "Unknown".
- Facial expression estimates appear when expression recognition is available.

**Important:** A fresh clone of this repository will not contain your registered face images or trained LBPH model. Each user must register their own test subjects and train their own model locally before using identity recognition.

## Privacy and Data Protection

EmotiVision processes facial images, which can be sensitive personal information. Protecting these images is an important part of the project.

The repository is configured to exclude local data files through `.gitignore`.

The following files and folders should remain private:

- `face_data/`
- `emotivision_model.yml`
- `labels.json`
- `contacts.csv`
- `.venv/`

The repository may contain the application source code and documentation without containing the photographs used during local testing.

### Privacy Practices

- Obtain consent before registering another person's face.
- Store captured images securely.
- Do not upload face photographs or personal contact details to a public repository.
- Avoid including identifiable faces in screenshots or demonstrations.
- Do not assume that excluding photographs automatically makes every trained model anonymous.
- Delete local face data when it is no longer required.
- Review repository contents before every upload.

A `.gitignore` file helps prevent accidental uploads, but it does not remove files that have already been committed. Previously uploaded sensitive data may require removal from the repository's history.

## Limitations

- Recognition performance depends on lighting, face position, image quality, and the registered samples.
- LBPH may perform less reliably when faces change significantly in pose, lighting, or appearance.
- A recognition threshold is a configurable decision boundary, not a guaranteed measure of accuracy.
- Facial expression classification may produce incorrect predictions.
- Expression labels do not reliably reveal a person's actual internal emotional state.
- Performance depends on the computer, webcam, and installed software dependencies.
- The current project is an educational demonstration, not a validated security or psychological assessment system.

## Future Improvements

Possible future enhancements include:

- Evaluate recognition accuracy on a separate test dataset.
- Measure false matches and missed identifications.
- Compare inference time under different conditions.
- Add better registration guidance for lighting and face positioning.
- Improve error handling and dependency management.
- Add secure data deletion controls.
- Add configurable recognition thresholds.
- Improve the user interface and accessibility.
- Explore modern face embedding models as a separate experimental extension.
- Add privacy-preserving options for local processing and data retention.

Any algorithm comparison should be supported by actual experiments before reporting numerical performance claims.

## References

1. Viola, P., & Jones, M. (2001). *Rapid Object Detection Using a Boosted Cascade of Simple Features.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

2. Turk, M., & Pentland, A. (1991). *Eigenfaces for Recognition.* Journal of Cognitive Neuroscience, 3(1), 71–86.

3. Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997). *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(7), 711–720.

4. Ahonen, T., Hadid, A., & Pietikäinen, M. (2006). *Face Description with Local Binary Patterns: Application to Face Recognition.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(12), 2037–2041.

5. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

6. Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). *Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements.* Psychological Science in the Public Interest, 20(1), 1–68.

7. Justin Shenk. *FER: Facial Expression Recognition.* Python library and project documentation.

## Author

**Project:** EmotiVision — Real-Time Face Recognition & Facial Expression Recognition System

**Developed using:** Python, OpenCV, LBPH, FER, and Tkinter

This project was developed for educational purposes to explore computer vision, machine learning, and real-time facial analysis.

---

*EmotiVision is an educational project. Its identity predictions and facial-expression estimates should not be treated as infallible or used as the sole basis for consequential decisions.*
