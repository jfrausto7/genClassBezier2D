# genClassBezier2D
 A short "starter project" with the goal of generating a small dataset of abstract 3D shapes and then training a model to classify them.

## What and how?

TODO

## How to run

### To train
```
python3 main.py --train --epochs
```

### To test
```
python3 main.py --test --weights "./path/to/weights"
```

### To predict on a round of GeoGuessr
```
python3 main.py --predict --weights "./path/to/weights" --game "www./geoguessr-link.com"
```


## References

[1] S. Xie and Z. Tu, “Holistically-nested edge detection,” 2015 IEEE International Conference on Computer Vision (ICCV), 2015. 

### Credits 

This application uses Open Source components. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: bezier_shapes https://github.com/jviquerat/bezier_shapes
Copyright (c) 2019 Jonathan Viquerat (marc@imadjine.com)
License (MIT) https://github.com/jviquerat/bezier_shapes/blob/master/LICENSE
