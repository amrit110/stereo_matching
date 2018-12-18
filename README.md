# stereo_matching

This is a Tensorflow re-implementation of Luo, W., & Schwing, A. G. (n.d.). Efficient Deep Learning for Stereo Matching.
(https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf)


## To run

### Setup data folders

```
data
└───kitti_2015
    │─── training
         |───image_2
             |───000000_10.png
             |───000001_10.png
             |─── ...
         |───image_3
         |───disp_noc_0
         |─── ...
    │─── testing
         |───image_2
         |───image_3
```

### Start training

```bash
python main.py --dataset kitti_2015 --patch-size 37 --disparity-range 201
```

## Results

* After training for 40k iterations.
* Qualitative results on validation set.

### KITTI 2015 Stereo

Example input images

<p float="left">
 <img src="/plots/inputs_sample.png" width="800" />
</p>

Disparity Ground-truth

<p float="left">
 <img src="/plots/disparity_sample.png" width="640" />
</p>


#### Example input patches

<p float="left">
 <img src="/plots/sample_patch_left_1.png" width="50" />
 <img src="/plots/sample_patch_right_1.png" width="320" />
</p>

<p float="left">
 <img src="/plots/sample_patch_left_2.png" width="50" />
 <img src="/plots/sample_patch_right_2.png" width="320" />
</p>

<p float="left">
 <img src="/plots/sample_patch_left_3.png" width="50" />
 <img src="/plots/sample_patch_right_3.png" width="320" />
</p>

#### Qualitative results

<p float="left">
 <img src="/plots/qualitative_sample_1.png" width="800" />
</p>

<p float="left">
 <img src="/plots/qualitative_sample_2.png" width="800" />
</p>

<p float="left">
 <img src="/plots/qualitative_sample_3.png" width="800" />
</p>

### KITTI 2012 Stereo

#### Qualitative results

<p float="left">
 <img src="/plots/qualitative_sample_4.png" width="800" />
</p>

<p float="left">
 <img src="/plots/qualitative_sample_5.png" width="800" />
</p>

<p float="left">
 <img src="/plots/qualitative_sample_6.png" width="800" />
</p>
