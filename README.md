# Amazon Picking Challenge
vision code for amazing picking challenge

### Foreground Segmentation
put background model images in bg_images folder

```
cd fg_segmentation
python background_subtractor.py -i test_image.png
```

### Color Histogram Detector
put object images in objects_training_data folder, 
with each object under their name folder

training the model by:
```
cd color_hist_detector
python color_hist_classifier.py
```

### Detect object and return attributes
see detect.py

### TODO
* Python absolute path vs relative path
* bg model file in fg_segmentation 
* color hist model in color_hist_detector
