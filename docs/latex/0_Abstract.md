# 摘要

随着生成式人工智能（如扩散模型 SDXL、Flux 等）的飞速发展，高度逼真的伪造与篡改图像给网络信息安全带来了严峻挑战。现有的图像篡改检测方法大多局限于整图级别的真伪分类，或过度依赖庞大的预训练基础模型提取差异特征，难以在保证计算效率的同时提供极具解释性的像素级篡改定位。此外，在联合训练分类与定位的多任务架构中，往往存在特征优化的“梯度拔河”现象，导致定位特征被全局分类特征稀释，从而引起指标退化。针对上述痛点，本文提出了一种基于多模态特征融合的生成式图像篡改检测与像素级极速定位算法（LaRE）。本文的主要研究工作与创新点如下：

**（1）提出了一种轻量化的空间与频域物理取证特征（SSFR）构建方法。** 针对传统方法提取扩散特征计算开销巨大（如依赖 7GB 的 UNet 模型）的弊端，本文设计了包含去噪残差、JPEG 重压缩伪影、频域相位异常、高斯噪声极值及 VAE 重构误差等 7 通道的物理特征提取模块。该模块实现了全 GPU 向量化加速，显著降低了显存占用与时间开销，为后续模型提供了极速且高区分度的底度篡改痕迹。

**（2）设计了基于结构相似度（SSIM）的细粒度掩码生成与数据流解耦策略。** 针对篡改定位任务缺乏高质量真实标注数据的问题，本文通过 SSIM 算法自动化比对原图与篡改图的结构级差异，高精度且批量化地渲染出对应像修改区域的黑白掩码（Mask）。同时，在工程层面实现了分类特征池与定位监督池的严格正交隔离，从根源上阻断了训练集与评估测试集的数据泄露风险。

**（3）构建了分类与定位双轨并行的多模态解耦网络架构。** 在真伪分类分支，提出基于 LaFT 融合思想的全局检测网络，将高分辨率（512×512）的 RGB 纹理特征与低分辨率（32×32）的 SSFR 物理特征经由 CLIP 骨干网络与空间亮度门控机制进行深度穿插；在篡改定位分支，独立引入 SegFormer 分割架构。针对多任务环境下的特征稀释问题，本文通过解耦 Luma Gate 权重并提出定位损失动态缩放因子（SEG_LOSS_SCALE），成功重建了梯度传输平衡，有效克服了大幅扩充无 Mask 样本所引起的定位性能退化。

**（4）在多源异构据集上进行了详尽的实验分析与架构消融。** 本文构建了包含真实图像（FFHQ, FORLAB 等）与多种主流生成器（SDXL, Flux, BR-Gen, Doubao 等）的混合篡改数据集。实验结果表明，该算法在实现 99.6% 的高精度真伪二分类拦截率的同时，像素级伪造定位的 Dice 性能依然稳固。结合禁用空间错位的数据增强策略，系统展现出了卓越的跨域泛化能力与鲁棒性。

**关键词：** 图像篡改检测；像素级定位；多模态特征融合；物理取证；多任务解耦；SegFormer


---

# Abstract

With the rapid development of generative artificial intelligence (such as diffusion models like SDXL and Flux), highly photorealistic forged images pose severe challenges to network information security. Existing image tampering detection methods are mostly limited to image-level binary classification or over-rely on massive pre-trained foundation models for feature extraction, struggling to provide highly explainable pixel-level tampering localization while maintaining computational efficiency. Furthermore, in multi-task architectures that jointly train classification and localization, a "gradient tug-of-war" phenomenon often occurs, where localization features are diluted by global classification gradients, leading to performance degradation. To address these issues, this paper proposes LaRE, a generative image tampering detection and extremely fast pixel-level localization algorithm based on multimodal feature fusion. The main contributions of this paper are as follows:

**(1) A lightweight spatial and frequency-domain physical forensic feature (SSFR) construction method is proposed.** To overcome the massive computational overhead of traditional diffusion feature extraction (e.g., relying on a 7GB UNet model), this paper designs a 7-channel physical feature extraction module comprising denoising residuals, JPEG recompression artifacts, frequency-domain phase anomalies, Gaussian noise extremes, and VAE reconstruction errors. This module achieves full GPU-vectorized acceleration, significantly reducing VRAM usage and time overhead, providing extremely fast and highly discriminative low-level tampering signatures for subsequent models.

**(2) A fine-grained mask generation and data flow decoupling strategy based on Structural Similarity (SSIM) is designed.** Addressing the lack of high-quality annotated data for tampering localization, this paper utilizes the SSIM algorithm to automatically compare structural differences between original and tampered images, generating high-precision pixel-level binary masks in batches. Meanwhile, at the engineering level, strict orthogonal isolation of the classification feature pool and the localization supervision pool is implemented, eradicating the risk of data leakage between training and evaluation sets.

**(3) A dual-track parallel multimodal decoupled network architecture for classification and localization is constructed.** In the authenticity classification branch, a global detection network based on the LaFT fusion concept is proposed, which deeply interleaves high-resolution (512×512) RGB texture features with low-resolution (32×32) SSFR physical features via a CLIP backbone and spatial luminance gating mechanisms. In the tampering localization branch, the SegFormer segmentation architecture is introduced independently. To tackle the feature dilution problem in multi-task environments, this paper decouples the Luma Gate weights and introduces a dynamic localization loss scaling factor (`SEG_LOSS_SCALE`), successfully reconstructing gradient transmission balance and effectively overcoming localization degradation caused by large-scale expansion of mask-free samples.

**(4) Comprehensive experimental analyses and architectural ablations are conducted on multi-source heterogeneous datasets.** This paper constructs a mixed tampering dataset comprising real images (FFHQ, FORLAB, etc.) and various mainstream generators (SDXL, Flux, BR-Gen, Doubao, etc.). Experimental results indicate that while achieving a high-precision binary classification interception rate of 99.6%, the algorithm maintains robust pixel-level forgery localization Dice performance. Combined with the strategy of disabling spatial-misalignment data augmentations, the system demonstrates excellent cross-domain generalization and robustness.

**Keywords:** Image Tampering Detection; Pixel-level Localization; Multimodal Feature Fusion; Physical Forensics; Multi-task Decoupling; SegFormer