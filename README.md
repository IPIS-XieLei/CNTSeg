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
Enter python train_mutil_T1_FA_Peaks.py to run the code. If you are prompted for no packages, enter pip install * * to install dependent packages
- For CNTSeg-v2
Enter python train_mutil_T1_FA_Peaks.py to run the code. If you are prompted for no packages, enter pip install * * to install dependent packages
## **Copyright**

---

After training, you can use the python train_mutil_T1_FA_Peaks.py to validate your model.

# **Concact**

---

Lei Xie, leix@zjut.edu.cn# CNTSeg
