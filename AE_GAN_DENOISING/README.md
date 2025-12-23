# 🧼 Image Denoising bằng Autoencoder & GAN

## 📋 MỤC LỤC
1. [Giới thiệu dự án](#giới-thiệu-dự-án)
2. [Công nghệ sử dụng](#công-nghệ-sử-dụng)
3. [Kiến trúc 2 model](#kiến-trúc-2-model)
4. [Quy trình xử lý dữ liệu](#quy-trình-xử-lý-dữ-liệu)
5. [Chi tiết quá trình huấn luyện](#chi-tiết-quá-trình-huấn-luyện)
6. [Hướng dẫn chạy code](#hướng-dẫn-chạy-code)
7. [Kết quả & Đánh giá](#kết-quả--đánh-giá)

---

## 🎯 Giới Thiệu Dự Án

### Mục đích
Dự án này phát triển **2 mô hình Deep Learning** để **khử nhiễu ảnh (Image Denoising)**:
- **Autoencoder**: Mô hình neural network không giám sát
- **GAN (Generative Adversarial Network)**: Mô hình sinh với 2 network cạnh tranh

### Bài toán
- **Đầu vào**: Ảnh bị nhiễu (Gaussian, Bernoulli, Poisson)
- **Đầu ra**: Ảnh sạch (khử nhiễu)
- **Metrics**: PSNR (Peak Signal-to-Noise Ratio), MSE (Mean Squared Error)

### So sánh 2 Model

| Tiêu chí | Autoencoder | GAN |
|----------|-------------|-----|
| **Kiến trúc** | 1 network (Encoder + Decoder) | 2 networks (Generator + Discriminator) |
| **Loss function** | MSE/L1 Loss | Adversarial + Reconstruction Loss |
| **Đặc điểm** | Tái tạo ảnh trơn mượt | Ảnh sinh sống động, chi tiết tốt |
| **Tốc độ training** | Nhanh | Chậm hơn (2 networks) |
| **Ứng dụng** | Compression, Anomaly Detection | High-quality denoising |

---

## 💾 Công Nghệ Sử Dụng

### Framework & Libraries
```
PyTorch 2.0+           # Deep Learning framework
torchvision            # Bộ công cụ xử lý ảnh
torch.cuda.amp         # Mixed Precision Training (GPU optimization)
numpy / matplotlib     # Xử lý số & visualize
```

### GPU Optimization
- **CUDA & cuDNN**: Tối ưu hoá tính toán trên GPU
- **Mixed Precision (FP16)**: Giảm memory, tăng tốc độ
- **Batch size động**: Tự điều chỉnh theo GPU available
- **Non-blocking GPU transfer**: Data transfer song song

### Nhiễu Hỗ Trợ
1. **Gaussian (Normal) Noise**: $I_{noisy} = I + \sigma \cdot N(0,1)$
2. **Bernoulli (Dropout) Noise**: Random pixels bị xóa
3. **Poisson Noise**: Photon noise trong ảnh thực

---

## 🏗️ Kiến Trúc 2 Model

### 1️⃣ Autoencoder Architecture

```
INPUT (3×128×128)
    ↓
ENCODER (Downsampling):
  Conv2d(3→32, stride=2) + BatchNorm + ReLU     → 32×64×64
  Conv2d(32→64, stride=2) + BatchNorm + ReLU    → 64×32×32
  Conv2d(64→128, stride=2) + BatchNorm + ReLU   → 128×16×16
  Conv2d(128→256, stride=2) + BatchNorm + ReLU  → 256×8×8
    ↓
BOTTLENECK (Compressed Code)
  Dimension: 256×8×8 = 16,384 values
    ↓
DECODER (Upsampling):
  ConvTranspose2d(256→128, stride=2) + BatchNorm + ReLU  → 128×16×16
  ConvTranspose2d(128→64, stride=2) + BatchNorm + ReLU   → 64×32×32
  ConvTranspose2d(64→32, stride=2) + BatchNorm + ReLU    → 32×64×64
  ConvTranspose2d(32→3, stride=2) + Sigmoid              → 3×128×128
    ↓
OUTPUT (3×128×128) - Ảnh khôi phục
```

**Thông số:**
- **Số tham số**: ~2.1M
- **Compression ratio**: ~150x (3×128×128 → 256×8×8)
- **Loss function**: MSE hoặc L1
- **Activation cuối**: Sigmoid (output: [0, 1])

### 2️⃣ GAN Architecture

#### Generator (Sinh ảnh)
```
INPUT: Random noise z ~ N(0, 1)
  Dimension: [batch, 100]
    ↓
Fully Connected + Reshape → [batch, 256, 8, 8]
    ↓
ConvTranspose2d(256→128, stride=2) + BatchNorm + ReLU  → 128×16×16
ConvTranspose2d(128→64, stride=2) + BatchNorm + ReLU   → 64×32×32
ConvTranspose2d(64→32, stride=2) + BatchNorm + ReLU    → 32×64×64
ConvTranspose2d(32→3, stride=2) + Sigmoid              → 3×128×128
    ↓
OUTPUT: Fake image (tạo từ noise)
```

#### Discriminator (Phân biệt thực/giả)
```
INPUT: Ảnh real hoặc fake [batch, 3, 128, 128]
    ↓
Conv2d(3→64, stride=2) + LeakyReLU(0.2)                → 64×64×64
Conv2d(64→128, stride=2) + BatchNorm + LeakyReLU(0.2)  → 128×32×32
Conv2d(128→256, stride=2) + BatchNorm + LeakyReLU(0.2) → 256×16×16
Conv2d(256→512, stride=2) + BatchNorm + LeakyReLU(0.2) → 512×8×8
    ↓
Adaptive Avg Pool + Flatten → [batch, 512]
    ↓
Linear(512 → 1) + Sigmoid → [batch, 1] ∈ [0, 1]
    ↓
OUTPUT: Probability (real=1, fake=0)
```

**Thông số:**
- **Generator param**: ~1.7M
- **Discriminator param**: ~2.5M
- **Total**: ~4.2M

---

## 📊 Quy Trình Xử Lý Dữ Liệu

### Cấu trúc Input
```
Input Dataset/
├── classA/          # Lớp A: Ảnh người hoặc vật thể
│   ├── image1.png
│   └── ...
├── classB/          # Lớp B: Ảnh nền hoặc lớp khác
│   ├── image1.png
│   └── ...
```

**Dữ liệu được chuẩn hoá:**
1. **Resize**: Tất cả ảnh → 128×128
2. **ToTensor**: Convert sang tensor [0, 1]
3. **Train/Val split**: 80% train, 20% validation

### Quá trình Thêm Nhiễu
```
Original Image I
    ↓
Add Noise (loai_nhieu, do_manh_nhieu)
    ├── Gaussian: I_noisy = I + σ·N(0,1)
    ├── Bernoulli: I_noisy = I · Bernoulli(p)
    └── Poisson: I_noisy = Poisson(I·λ)/λ
    ↓
Noisy Image I_noisy (training input)
```

### Data Loading
```python
# Cấu hình tối ưu GPU
batch_size = 32          # GPU có đủ VRAM
num_workers = 4          # Parallel loading
pin_memory = True        # Transfer nhanh hơn
```

---

## 🎓 Chi Tiết Quá Trình Huấn Luyện

### Autoencoder Training

**Target Function:**
$$L_{AE} = \text{MSE}(\hat{I}, I_{target})$$

Với:
- $\hat{I}$: Ảnh tái tạo từ encoder-decoder
- $I_{target}$: Ảnh gốc (hoặc ảnh gốc/nhiễu tùy thuộc mode)

**Quá trình:**
```
Epoch 1 - N:
  For each batch:
    1. Đọc ảnh sạch (input)
    2. Thêm nhiễu → ảnh nhiễu (noisy input)
    3. Forward: noisy_input → Autoencoder → reconstructed
    4. Tính loss: MSE(reconstructed, clean_image)
    5. Backward pass + Optimizer step
    
  Validation:
    - Tính PSNR, MSE trên validation set
    - Lưu checkpoint nếu PSNR tốt nhất
```

**Hyperparameters:**
```
Epochs: 5
Batch size: 32
Learning rate: 1e-3
Optimizer: Adam
Loss: MSE
Mixed Precision: Enabled (nếu GPU capability >= 7)
```

### GAN Training

**Loss Functions:**

Generator loss:
$$L_G = L_{adv} + \lambda \cdot L_{recon}$$

Discriminator loss:
$$L_D = L_{real} + L_{fake}$$

Với:
- $L_{adv}$: Adversarial loss (BCEWithLogitsLoss)
- $L_{recon}$: Reconstruction loss (L1)
- $\lambda$: Weight của reconstruction (thường = 100)

**Quá trình Training (từng iteration):**

```
Step 1: Update Discriminator
  1. Đọc batch ảnh sạch
  2. Thêm nhiễu
  
  3. Forward real images → D
     Loss_real = D_loss(D(clean), label=1)
  
  4. Forward fake images từ Generator → D
     Loss_fake = D_loss(D(G(noisy)), label=0)
  
  5. Total D loss = Loss_real + Loss_fake
  6. Backward + Optimizer step

Step 2: Update Generator
  1. Forward noisy → Generator → fake
  
  2. Adversarial loss:
     G_loss_adv = D_loss(D(fake), label=1)
  
  3. Reconstruction loss:
     G_loss_recon = L1(fake, clean)
  
  4. Total G loss = G_loss_adv + 100 * G_loss_recon
  5. Backward + Optimizer step

Validation:
  - Tính PSNR, MSE trên validation set
  - Lưu checkpoint nếu PSNR tốt nhất
```

**Hyperparameters:**
```
Epochs: 5
Batch size: 32
Learning rate: 2e-4
Optimizer: Adam (β1=0.5, β2=0.999)
Lambda (recon weight): 100.0
Loss: BCEWithLogitsLoss + L1Loss
Mixed Precision: Enabled
```

---

## 🚀 Hướng Dẫn Chạy Code

### Bước 1: Chuẩn Bị Environment
```bash
# Cài thư viện
pip install torch torchvision
pip install pillow numpy matplotlib

# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Bước 2: Chuẩn Bị Dữ Liệu
```
AE_GAN_DENOISING/
├── thumbnails/
│   ├── classA/
│   │   ├── img1.png
│   │   └── ...
│   └── classB/
│       ├── img1.png
│       └── ...
```

### Bước 3: Chạy Notebook

#### Cell 1-4: Setup & Configuration
```python
# Kiểm tra GPU, thiết lập hyperparameters
# Output: Device info, Model architecture
```

#### Cell 5-6: Prepare Data
```python
# Load ImageFolder dataset
# Tạo DataLoaders với num_workers tối ưu
```

#### Cell 7: Evaluate & Visualize Functions
```python
# Định nghĩa PSNR, MSE, hàm thêm nhiễu
# Hàm lưu ảnh minh hoạ
```

#### Cell 8: Train Autoencoder
```python
# Huấn luyện Autoencoder
# Thời gian: ~2-3 phút (GPU)
# Lưu: best_ae_model.pth
```

#### Cell 9: Plot Training History
```python
# Vẽ biểu đồ Loss, PSNR, MSE
```

#### Cell 10: Train GAN
```python
# Huấn luyện Generator + Discriminator
# Thời gian: ~3-5 phút (GPU)
# Lưu: best_gan_generator.pth, best_gan_discriminator.pth
```

#### Cell 11-12: Comparison & Visualization
```python
# So sánh AE vs GAN
# Tạo bảng metrics & visualize kết quả
```

### Bước 4: Xem Kết Quả
```bash
# Output files được lưu trong:
./outputs_denoise/

# Hoặc (trên Kaggle):
/kaggle/working/outputs_denoise/

# Files:
- viz_best_autoencoder.png    # Kết quả AE
- viz_best_gan_generator.png  # Kết quả GAN
- comparison_training.png     # So sánh huấn luyện
- ae_vs_gan_comparison.png    # So sánh chi tiết
```

### Xử Lý Lỗi Thường Gặp

| Lỗi | Giải pháp |
|-----|----------|
| **CUDA out of memory** | Giảm batch_size: 32 → 16 → 8 |
| **Dataset not found** | Kiểm tra đường dẫn `thu_muc_du_lieu` |
| **Module not found** | Chạy: `pip install torch torchvision pillow` |
| **Slow training (CPU)** | Bật GPU hoặc giảm số epoch |

---

## 📈 Kết Quả & Đánh Giá

### Metrics Đánh Giá

**PSNR (Peak Signal-to-Noise Ratio)** - Cao hơn tốt hơn
$$\text{PSNR} = 20 \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right) \text{ dB}$$

**MSE (Mean Squared Error)** - Thấp hơn tốt hơn
$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(I_i - \hat{I}_i)^2$$

### Kết Quả Dự Kiến

| Model | PSNR (dB) | MSE | Loss |
|-------|-----------|-----|------|
| Input (nhiễu) | ~15-20 | ~0.02-0.05 | N/A |
| Autoencoder | ~25-30 | ~0.001-0.003 | 0.0245 |
| GAN | ~26-32 | ~0.0008-0.0025 | Balanced |

### Kết Luận
- **Autoencoder**: Nhanh, ổn định, ảnh mịn
- **GAN**: Ảnh sắc nét, chi tiết tốt, nhưng khó huấn luyện

---

## 🎨 Visualization Outputs

### 1. Training History (comparison_training.png)
- G Loss, D Loss qua epochs
- PSNR, MSE so sánh AE vs GAN

### 2. Denoising Results (ae_vs_gan_comparison.png)
- 4 hàng, mỗi hàng: noisy → clean → AE result → GAN result
- PSNR in dB trên mỗi kết quả

### 3. Model Checkpoints
```
best_ae_model.pth           # Best Autoencoder weights
best_gan_generator.pth      # Best Generator weights
best_gan_discriminator.pth  # Best Discriminator weights
```

---

## 💡 Ứng Dụng Thực Tế

1. **Medical Imaging**: Khử nhiễu từ CT, MRI scans
2. **Astronomy**: Xử lý ảnh từ kính thiên văn
3. **Surveillance**: Cải thiện chất lượng video giám sát
4. **Photography**: Post-processing để tạo ảnh sạch hơn
5. **Data Augmentation**: Sinh ảnh sạch từ ảnh nhiễu
6. **Anomaly Detection** (AE): Phát hiện bất thường dựa trên reconstruction error

---

## 📚 Tài Liệu Tham Khảo

1. **Autoencoder**: Hinton & Salakhutdinov (2006) - "Reducing the Dimensionality of Data with Neural Networks"
2. **GAN**: Goodfellow et al. (2014) - "Generative Adversarial Nets"
3. **DCGAN**: Radford et al. (2015) - "Unsupervised Representation Learning with DCGANs"
4. **PyTorch**: https://pytorch.org/docs/
5. **Image Denoising**: https://en.wikipedia.org/wiki/Image_noise

---

**Author**: Computer Vision Project  
**Date**: December 2025  
**Framework**: PyTorch 2.0+
