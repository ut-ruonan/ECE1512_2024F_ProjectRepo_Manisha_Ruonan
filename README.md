# ECE1512_2024F_ProjectRepo_Manisha_Ruonan

## Overview

This repository contains the implementation for "Dataset Distillation for Data-Efficient Learning," created as part of the ECE1512H project by Manisha Mahagammulle Gamage and Ruonan Niu. The project investigates dataset distillation techniques, particularly Attention Matching, Prioritize Alignment Distillation (PAD) and Distribution Matching (DM) methods to develop synthetic datasets that replicate the characteristics of MNIST and MHIST datasets, providing a solution for training models with reduced data while maintaining accuracy.

## Project Structure
- **Task 1:** Focuses on dataset distillation using Attention Matching for MNIST and MHIST datasets, evaluating the performance of models trained on these synthetic datasets.

- **Task 2:** Implements Prioritize Alignment Distillation (PAD) and Distribution Matching (DM), state-of-the-art techniques designed to optimize feature alignment and distribution representation in synthetic data, enhancing generalization across different architectures.

Each task includes relevant code, synthetic datasets, images, and evaluation scripts.

## Methodology
- **Attention Matching:** Aligns synthetic datasets with attention maps of real data to capture essential features in a data-efficient manner.
- **Prioritize Alignment Distillation (PAD):** Emphasizes feature alignment across layers to improve generalization and robustness, especially for complex datasets.
- **Distribution Matching (DM):** Utilizes feature distribution alignment to condense important data patterns, ensuring that synthetic datasets retain critical statistical characteristics of the original data.

## Key Contributions

1. **Efficient Training**: The synthetic datasets created with Attention Matching, PAD, and DM allow training with significantly fewer data points.
2. **Cross-Architecture Generalization**: Synthetic datasets are evaluated on different architectures, showcasing their ability to generalize across ConvNet and ResNet models.
3. **Noise Robustness**: Experiments include synthetic datasets with Gaussian noise to analyze model resilience.

## Repository Usage

This repository is intended for reference and educational purposes. The code in this repository demonstrates dataset distillation methods and provides insights into creating synthetic datasets for machine learning.

## Results Summary
- **Accuracy:** Synthetic data achieves comparable accuracy.
- **Efficiency:** Significant reductions in computational cost.
- **Generalization:** Synthetic datasets generalize well across different model architectures.
For further details, consult the documentation within each task's folder.


## References

[1] A. Sajedi, S. Khaki, E. Amjadian, L. Z. Liu, Y. A. Lawryshyn, and K. N. Plataniotis, “DataDAM: Efficient dataset distillation with attention matching,” in *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 17051–17061.

[2] S. Khaki and K. N. Plataniotis, “Project A: Dataset distillation: A data-efficient learning framework,” in *ECE1512H F Digital Image Processing*, 2024, pp. 1–10.

[3] T. Wang, J.-Y. Zhu, A. Torralba, and A. A. Efros, “Dataset distillation,” *arXiv preprint:1811.10959*, 2018.

[4] Z. Li, Z. Guo, W. Zhao, T. Zhang, Z.-Q. Cheng, S. Khaki, K. Zhang, A. Sajedi, K. N. Plataniotis, K. Wang, and Y. You, “Prioritize alignment in dataset distillation,” *arXiv preprint:2408.03360*, 2024.

[5] B. Zhao and H. Bilen, “Dataset condensation with distribution matching,” in *2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2023, pp. 6503–6512.

[6] B. Zhao, K. R. Mopuri, and H. Bilen, “Dataset condensation with gradient matching,” *arXiv preprint:2006.05929*, 2021.
