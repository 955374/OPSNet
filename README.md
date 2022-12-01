# Object Preserving Siamese Network for Single Object Tracking on Point Clouds

## Environment settings
* Create an environment for OPSNet
```
conda create -n OPSNet python=3.7
conda activate OPSNet
```

* Install pytorch and torchvision
```
conda install pytorch==1.7.0 torchvision==0.5.0 cudatoolkit=10.0
```

* Install dependencies.
```
pip install -r requirements.txt
pip install mmcv-full
```

## Data preparation
### [KITTI dataset](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf)
* Download the [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Unzip the downloaded files and place them under the same parent folder.

## Evaluation

Train a new model:
```
python train_tracking.py --data_dir /path/to/data --category_name category_name
```

Test a model:
```
python test_tracking.py --data_dir /path/to/data --category_name category_name
```

## Acknowledgements

- Thank Hui for his implementation of [V2B](https://github.com/fpthink/V2B) and [STNet](https://github.com/fpthink/STNet)
- Thank Zhou for his implementation of [PTTR](https://github.com/Jasonkks/PTTR)
- Thank Qi for his implementation of [P2B](https://github.com/HaozheQi/P2B).

## License
This repository is released under MIT License.

