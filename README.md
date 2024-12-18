# Arthropod Taxonomy Orders - Object Detection 
Species identification from images is complex, especially when multiple species are present. While traditional image classification focuses on single-species detection, identifying all species in an image requires a hierarchical approach. The ArTaxOr dataset addresses this challenge for arthropods (insects, spiders, crustaceans, etc.), which include over 1.3 million described species. Instead of creating an exhaustive dataset, the problem is broken into steps: object detection at the order level (>120 orders), followed by family and species classification.

![ArTaxOr.png](/images/ArTaxOr.png)
---

# Dataset

The dataset consists of images of arthropods in jpeg format and object boundary boxes in json format. There are between one and 50 objects per image.

This dataset is actively maintained, and new orders will be added on a regular basis. Currently, the following orders are covered with at least 2000 objects per order:

- Araneae (spiders), adults, juveniles
- Coleoptera (beetles), adults
- Diptera (true flies, including mosquitoes, midges, crane file etc.), adults
- Hemiptera (true bugs, including aphids, cicadas, planthoppers, shield bugs etc.), adults and nymphs
- Hymenoptera (ants, bees, wasps), adults
- Lepidoptera (butterflies, moths), adults
- Odonata (dragonflies, damselflies), adults

The detail of dataset in [dataset kaggle](https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset)

# Approach

Faster R-CNN and YOLO are two popular object detection models widely used in computer vision tasks.

- **Faster R-CNN (Region-based Convolutional Neural Network)**:  
  Faster R-CNN is a two-stage object detection model. It first identifies regions of interest (RoIs) using a Region Proposal Network (RPN) and then classifies and refines these regions in a second stage. Known for its high accuracy, Faster R-CNN is often used in applications requiring precise localization, though it may be slower compared to real-time models.

- **YOLO (You Only Look Once)**:  
  YOLO is a one-stage object detection model that predicts bounding boxes and class probabilities directly from an input image in a single pass. Optimized for speed and efficiency, YOLO is suitable for real-time applications while maintaining good accuracy, making it a popular choice for tasks requiring high-speed object detection.

# Set up

The training stage was implemented in the Kaggle environment. You need to install the required dependencies as shown below to ensure the proper execution of the code. The installation methods for each approach are presented in these individual notebooks. Details:

**YOLOv10**: ```/yolov10/notebooks/yolov10-training.ipynb```

**Faster-RCNN**: ```/faster-rcnn/notebooks/faster-rcnn-training.ipynb```

# Experiment
