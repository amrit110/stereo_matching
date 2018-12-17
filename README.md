# stereo_matching

This is a Tensorflow re-implementation of Luo, W., & Schwing, A. G. (n.d.). Efficient Deep Learning for Stereo Matching.
(https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf)


#### Setup data folders

```
data
└───kitti_2015
    │─── training
         |───image_2
         |───image_3
         |───disp_noc_0
         |─── ...
    │─── testing
         |───image_2
         |───image_3
```

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

##### KITTI 2015

<p float="left">
 <img src="/plots/qualitative_sample_1.png" width="800" />
</p>

<p float="left">
 <img src="/plots/qualitative_sample_2.png" width="800" />
</p>

<p float="left">
 <img src="/plots/qualitative_sample_3.png" width="800" />
</p>
