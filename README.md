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

[1] Habibollah Agh Atabay. “Binary shape classification using Convolutional Neu- ral Networks”. In: IIOAB Journal 7 (Oct. 2016), pp. 332–336.

[2] John Canny. “A Computational Ap- proach to Edge Detection”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-8.6 (1986), pp. 679–698. DOI: 10.1109/TPAMI. 1986.4767851.

[3] Jonathan Viquerat and Elie Hachem. “A supervised neural network for drag prediction of arbitrary 2D shapes in
laminar flows at low Reynolds num-
ber”. In: Computers Fluids 210 (2020),
p. 104645. ISSN: 0045-7930. DOI: https: //doi.org/10.1016/j.compfluid. 2020.104645. URL: https:// www.sciencedirect.com/science/ article/pii/S0045793020302164.

[4] S. Xie and Z. Tu, “Holistically-nested edge detection,” 2015 IEEE International Conference on Computer Vision (ICCV), 2015. 

[5] Chaoyan Zhang et al. “SCN: A Novel Shape Classification Algorithm Based on Convolutional Neural Network”. In: Symmetry 13.3 (2021). ISSN: 2073- 8994. DOI: 10.3390/sym13030499. URL: https://www.mdpi.com/ 2073-8994/13/3/499.

### Credits 

This application uses Open Source components. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: bezier_shapes https://github.com/jviquerat/bezier_shapes
Copyright (c) 2019 Jonathan Viquerat (marc@imadjine.com)
License (MIT) https://github.com/jviquerat/bezier_shapes/blob/master/LICENSE

Project: hed https://github.com/s9xie/hed
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
License https://github.com/s9xie/hed/blob/master/LICENSE
