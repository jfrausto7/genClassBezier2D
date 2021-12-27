# genClassBezier2D
A short "starter project" with the goal of generating a small 2D image dataset of abstract shapes (formed using Bezier curves) and then training a model to classify them.

## What and how?

TODO

## How to run


### To generate datasets
```
python3 main.py --generate
```

## To utilize Holistically-Nested Edge Detection data augmentation
```
python3 main.py --hed
```

### To train (on default dataset)
```
python3 main.py --train --epochs 50
```

### To train (on specific dataset)
```
python3 main.py --train --epochs 50 --dataset colors
```

### To test (on default dataset)
```
python3 main.py --test --weights "./path/to/weights"
```

### To test (on specific dataset)
```
python3 main.py --test --weights "./path/to/weights" --dataset colors
```


## References

[1] S. Xie and Z. Tu, “Holistically-nested edge detection,” 2015 IEEE International Conference on Computer Vision (ICCV), 2015. 

### Credits 

This application uses Open Source components. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: bezier_shapes https://github.com/jviquerat/bezier_shapes
Copyright (c) 2019 Jonathan Viquerat (marc@imadjine.com)
License (MIT) https://github.com/jviquerat/bezier_shapes/blob/master/LICENSE

Project: hed https://github.com/s9xie/hed
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
License https://github.com/s9xie/hed/blob/master/LICENSE
