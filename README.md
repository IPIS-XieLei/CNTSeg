# README

# **Cranial Nerves Tract Segmentation -- CNTSeg**

---

Due to the slender structure of cranial nerves (CNs) and the complex anatomical environment, single-modality data based on dMRI cannot provide a complete and accurate description. CNs tract segmentation is the strategy of voxel-wise analysis, which directly classifies voxels based on associated fiber bundles, bypassing traditional streamline analysis. CNTSeg has achieved the segmentation of 5 pairs of CNs (i.e. optic nerve *CN II*, oculomotor nerve *CN III*, trigeminal nerve *CN V*, and facial–vestibulocochlear nerve *CN VII/VIII*).

<div align=center>
<img src="figures/cns5.jpg" width="400px">
</div>

CNTSeg is the code for the following papers. Please cite the papers if you use it.

- Xie L, Huang J, Yu J, et al. Cntseg: A multimodal deep-learning-based network for cranial nerves tract segmentation[J]. Medical Image Analysis, 2023, 86: 102766.
- CNTSeg-v2: An Arbitrary-Modal Fusion Network for Cranial Nerves Tract Segmentation. Submitted to IEEE TMI.

  

# **Install**

---
- pytorch >= 2.0.1
- Torchvision >= 0.15.2
- python >= 3.6
- numpy >= 1.20.1
- SimpleITK >= 2.2.1


# **How to use**

---
- For CNTSeg

Crop the HCP data (i.e., T1w and dMRI data) with a spatial resolutionof 145×174×145 voxels to 128×160×128 voxels; normalize the T1w images and dMRI data from 0 to 255.
```
python Pre_data.py
```
Obtaining training and testing data
```
python createmydata.py
```
Find the best weights for single modalities separately
```
python train_mutil_T1.py
python train_mutil_FA.py
python train_mutil_peaks.py
```
Load the best weights of a single modality to train the CNTSeg
```
python train_CNTSeg_V1.py
```
Validate model
```
python predict_CNTSeg_V1.py
```

Note: Use the MRtrix3 to process primary DWI data and get the corresponding FA images and Peaks images; The Human Connectome Project dataset is available at [https://db.humanconnectome.org](https://db.humanconnectome.org/).


- For CNTSeg-v2
  
-- python train_CNTSeg_V2.py
-- predict_CNTSeg_V2.py


# **Concact**

---
Lei Xie, College of Information Engineering, Zhejiang University of Technology

leix@zjut.edu.cn
