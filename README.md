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
* 3-pixel error evaluation on validation set.

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

##### Post-processing

* Cost-aggregation

Without cost-aggregation

<p float="left">
 <img src="/plots/qualitative_sample_7.png" width="800" />
</p>


With cost-aggregation

<p float="left">
 <img src="/plots/qualitative_sample_7_CA.png" width="800" />
</p>


A closer look to observe the smoothing of predictions, without cost
aggregation and with respectively:

<p float="left">
 <img src="/plots/disp_sample_zoom.png" width="400" />
 <img src="/plots/disp_sample_zoom_CA.png" width="400" />
</p>


#### Quantitative results

* To compare with results reported in paper, look at Table-5, column `Ours(37)`.


  |                                     | 3-pixel error (%)   |
  |-------------------------------------|:-------------------:|
  | baseline (paper)                    |     7.13            |
  | baseline (re-implementation)        |     7.271           |
  | baseline + CA (paper)               |     6.58            |
  | baseline + CA (re-implementation)   |     6.527           |


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

## Possible next steps

- [ ] Implement post processing to smoothen output.
- [ ] Look into error metrics and do quantitative analysis.
- [ ] Run inference on test video sequences.
- [x] Instead of the batch matrix multiplication during inference, which constructs a `B x H x W x W` tensor, use a loop to compute cost volume over the disparity range. Tensorflow VM might figure out that it should parallelise operations over the loop. 
