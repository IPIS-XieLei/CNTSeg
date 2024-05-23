# README

# **Cranial Nerves Tract Segmentation -- CNTSeg**

---

Due to the slender structure of cranial nerves (CNs) and the complex anatomical environment, single-modality data based on dMRI cannot provide a complete and accurate description. CNs tract segmentation is the strategy of voxel-wise analysis, which directly classifies voxels based on associated fiber bundles, bypassing traditional streamline analysis. CNTSeg has achieved the segmentation of 5 pairs of CNs (i.e. optic nerve *CN II*, oculomotor nerve *CN III*, trigeminal nerve *CN V*, and facial–vestibulocochlear nerve *CN VII/VIII*).
![Untitled](figures/cns5.jpg)

CNTSeg is the code for the following papers. Please cite the papers if you use it.

- Xie L, Huang J, Yu J, et al. Cntseg: A multimodal deep-learning-based network for cranial nerves tract segmentation[J]. Medical Image Analysis, 2023, 86: 102766.
- Xie L, Zeng Q, Zhou H, et al. Anatomy-guided fiber trajectory distribution estimation for cranial nerves tractography[J]. arXiv preprint arXiv:2402.18856, 2024. (ISBI2024).
- CNTSeg-v2: An Arbitrary-Modal Fusion Network for Cranial Nerves Tract Segmentation. Submitted to IEEE TMI.

  

# **Install**

---

In this work, we propose a novel multimodal deep-learning-based multi-class network for automated cranial nerves tract segmentation without using tractography, ROI placement or clustering, called CNTSeg. Specifically, we introduced T1w images, fractional anisotropy (FA) images, and fiber orientation distribution function (fODF) peaks into the training data set, and design the back-end fusion module which uses the complementary information of the interphase feature fusion to improve the segmentation performance. CNTSeg has achieved the segmentation of 5 pairs of CNs (i.e. optic nerve *CN II*, oculomotor nerve *CN III*, trigeminal nerve *CN V*, and facial–vestibulocochlear nerve *CN VII/VIII*).




# How to use

---

Enter python train_mutil_T1_FA_Peaks.py to run the code. If you are prompted for no packages, enter pip install * * to install dependent packages

## Copyright

---

After training, you can use the python train_mutil_T1_FA_Peaks.py to validate your model.

# Concact

---

Lei Xie, leix@zjut.edu.cn# CNTSeg
