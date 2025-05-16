## Social Distancing Detection using YOLOv8
This project implements a real-time social distancing violation detection system using the YOLOv8 object detection model. It detects people in a video and calculates pairwise distances to identify instances where individuals are closer than a defined safe threshold.

### Objective
The goal of this project is to automatically monitor public spaces and detect social distancing violations to promote safer environments—especially during health crises like COVID-19.

### Working
**Person Detection**

The system uses YOLOv8 (You Only Look Once, Version 8) to detect all persons in each video frame.

**Centroid Extraction**

For each person detected, the centroid of their bounding box is calculated.

**Distance Calculation**

It calculates the Euclidean distance between all pairs of centroids using SciPy's distance matrix.

**Violation Detection**

If the distance between any two people is less than a defined threshold (default: 50 pixels), a violation is flagged.

**Visualization**

Green bounding boxes indicate safe distance.
Red bounding boxes and dots indicate a social distancing violation.
Violation count is displayed on the frame.

### Technologies Used
**Python**

**YOLOv8 (Ultralytics)**

**OpenCV**

**NumPy**

**SciPy**
