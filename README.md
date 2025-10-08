# eWardrobe_Image-Segmentation-of-fashion-dataset

# Introduction
The project we have been working on is also called **eWardrobe**, a playful yet practical experiment that brings fashion into the digital world. The idea is simple: users can digitize their clothing by taking photos, and the app automatically segments each item. Once digitized, clothes can be stored in a personal database and even tried on virtually.

Behind the scenes, this combines deep learning and data science. On the one hand, image segmentation is used to extract the shape and details of each piece of clothing. For this, we trained different segmentation models using the DeepFashion2 dataset. On the other hand, all this information needs to be stored and managed in a structured way, building a digital wardrobe that can be searched, analyzed, and used for different applications. Here, we used a simple JSON-based database that works with pandas DataFrames.

---

# Deep Learning Methods, Results, and Conclusion

## Method

### Starting with YOLO
We began with **YOLO**, a popular model known for its speed and simplicity. Training the model itself is straightforward, but preparing the dataset for YOLO required considerable effort. We first converted the dataset into COCO format and then used the YOLO transformer to make it compatible.

We started small, working with a fraction of the dataset. Before training, we ran a short parameter sweep to find suitable settings, but this turned out to be difficult within the YOLO framework. In the end, we trained a model for several days. The results, however, were disappointing: details like sleeves were often missing in the segmentation. Since segmentation is crucial in fashion tasks, YOLO was not sufficient.

### PyTorch Methods (U-Net & Mask R-CNN)
To improve segmentation quality, we implemented both a custom U-Net for pixel-level semantic segmentation and Mask R-CNN for instance/mask segmentation inside the PyTorch ecosystem. PyTorch provided the flexibility to build custom loaders, control training loops, and run detailed evaluations.

#### Data Preparation
- **Mask R-CNN**: We converted or adapted annotations to COCO-style format (bounding boxes + polygon/mask representations) so the torchvision models could use them.  
- **U-Net**: We used DeepFashion2 JSON polygon annotations to generate binary masks (one mask per garment per image) across train/validation/test splits.  

#### Shared Pipeline Components
- Custom PyTorch DataLoader for efficient batch-wise training and handling variable-size inputs.  
- Modified predictors in maskrcnn_resnet50_fpn_v2 to handle DeepFashion2 categories.  
- Training utilities with checkpointing, COCO evaluation, and mixed precision (AMP).  
- Dataset subsampling for experiments on smaller fractions.  
- Parameter sweeps for learning rate, weight decay, optimizer type, and resource settings (batch size, workers).  
- Data augmentation (flips, rotations, color jitter) to improve generalization.  

#### Model-Specific Details
- **U-Net**: Trained as a pixel-wise segmentation model using masks as ground truth. We used combined loss functions (Dice loss + Binary Cross-Entropy) to balance overlap accuracy and pixel-wise classification. Predictions produced dense binary masks that captured fine contours (sleeves, collars, hems).  
- **Mask R-CNN**: Used `maskrcnn_resnet50_fpn_v2` modified for the 13 categories. The model produced bounding boxes, class scores, and instance masks; evaluation relied on COCO metrics for boxes and masks.  

---

## Results

### YOLO Outcomes
Even after long training runs, YOLO’s results looked crude. Sleeves were missing, and segmentation wasn’t precise enough for fashion tasks.

### Mask R-CNN Outcomes
Mask R-CNN delivered much better results. On the validation set, our best runs reached ~0.49 AP for bounding boxes and ~0.56 AP for segmentation using COCO metrics. Visual inspection confirmed this: masks captured contours of shirts, skirts, and sleeves far more accurately.

### U-Net Outcomes
We did not have final training results for U-Net at the time of writing this blog post.

---

## Issues Faced
Several challenges shaped the project:
- **Training time**: At ~2.5 hours per epoch, experimenting on the full dataset wasn’t practical. Subsampling, despite its costs, was essential.  
- **Dataset adaptation**: DeepFashion2 required adjustments to work smoothly with our tasks.  
- **Annotation handling**: JSON annotations required building robust loaders to parse mask polygons accurately.  

---

## Conclusion
The project underlined the importance of choosing the right model for the task:
- **YOLO** is great for fast bounding boxes, but it struggled with the fine details needed in fashion segmentation.  
- **Mask R-CNN** was slower and more complex to set up but superior in capturing garment outlines.  
- While **U-Net** training is still pending, we hope it will successfully capture fine details like sleeves, skirts, and collars, as well.  

Using a PyTorch pipline gave us the flexibility to add features like mixed precision, COCO evaluation, and parameter sweeps. This made it possible to optimize both accuracy and efficiency, something not as accessible in the YOLO workflow.

---

# Data Science Methods, Results, and Conclusion

## Method
While segmentation of clothing items was handled by deep learning, our responsibility was to build the **feature storage system** that organizes the extracted attributes. The goal was to create a lightweight database where each clothing item could be stored with its properties such as type, color, material, and style.

For this, we designed a simple **JSON-based database** (`wardrobe.json`) and wrote a Python program (`feature_store.py`) that:
- Loads the database into a pandas DataFrame.  
- Allows adding new clothing items (features would normally come from the ML model).  
- Saves updates back to the JSON file.  

This ensures that segmentation outputs can be connected with a structured dataset for further analysis and recommendations.

## Results
The initial database contained two sample items (a blue t-shirt and a black dress). Running the script successfully added a new item (e.g., a red t-shirt) and updated both the DataFrame and the JSON file. This demonstrates how the system can continuously store and update clothing features. The structured format (JSON + pandas) will later serve as input for the recommendation engine.

## Issues Faced
Since no fully automated feature extraction was available yet, we started with manually created examples. In real-world use, detecting properties like color, material, and style automatically will be more challenging. However, with the JSON + DataFrame pipeline ready, ML model outputs can be seamlessly integrated once available.

## Conclusion
Our contribution was to design and implement the **data infrastructure** for storing segmentation outputs and clothing attributes. This component is a crucial step toward building the full eWardrobe pipeline: segmentation identifies the clothing item, feature storage organizes its attributes, and the recommendation system can then suggest new outfits. Together, these components create the foundation for a digital wardrobe assistant that connects computer vision with personalized fashion recommendations.

---

# Prototype
To demonstrate the outcome of the project, a **prototype application** was developed. The app enables users to capture a photo of themselves directly through a webcam interface. This image is then processed by the trained models, and the segmentation results are returned in real time. To provide a direct comparison, both YOLO and Mask R-CNN inference results can be displayed side by side, highlighting differences in detection accuracy and segmentation quality.

As an additional feature, the app allows the segmented clothing to be overlaid directly onto the live video stream from the webcam. This functionality offers a first impression of how a virtual wardrobe could operate.
