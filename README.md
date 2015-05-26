# Amazon Picking Challenge
vision code for amazing picking challenge

### Foreground Segmentation
put background model images in bg_images folder

in show_foreground.py script folder
```
python show_foreground.py -d folder_with_images
```

### Color Histogram Detector
put object images in objects_training_data folder, 
with each object under their name folder

training the model by:
```
python train_detector.py
```

testing the model by:
```
python test_detector.py
```


### Detect object and return attributes
see detect.py

### Python absolute path vs relative path
The data used in the script is relative to the script paht,
so once the script is run other than the script path, it reuslts in error. 

The solutin is to transfer the relative path to absolute path. 
get the script whole path: 
os.path.realpath(\_\_file\_\_) 
further to get the dir path by: 
path, file = os.path.split(os.path.realpath(\_\_file\_\_))
