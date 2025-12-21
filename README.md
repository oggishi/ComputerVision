# PHÁT HIỆN NGƯỜI ĐI BỘ BẰNG DEEP LEARNING

## 📋 MỤC LỤC
1. [Giới thiệu dự án](#giới-thiệu-dự-án)
2. [Tập dữ liệu Penn-Fudan](#tập-dữ-liệu-penn-fudan)
3. [5 Mô hình Deep Learning](#5-mô-hình-deep-learning)
4. [Quy trình xử lý dữ liệu](#quy-trình-xử-lý-dữ-liệu)
5. [Chi tiết các mô hình](#chi-tiết-các-mô-hình)
6. [Kết quả và Visualization](#kết-quả-và-visualization)
7. [Hướng dẫn chạy code](#hướng-dẫn-chạy-code)
8. [Ứng dụng thực tế](#ứng-dụng-thực-tế)

---

## 🎯 Giới thiệu Dự Án

### Mục đích
Dự án này phát triển **5 mô hình Deep Learning khác nhau** để giải quyết bài toán **phát hiện và phân khúc người đi bộ (Pedestrian Detection & Segmentation)** từ hình ảnh.

### Phạm vi
- **Bài toán chính**: Phát hiện vị trí người trong ảnh
- **Bài toán phụ**: Phân loại crops, tái tạo ảnh, tạo ảnh 
- **Dataset**: Penn-Fudan Pedestrian Dataset (124 ảnh huấn luyện)
- **Framework**: PyTorch + TorchVision

### Tổng quan 5 mô hình
| Mô hình | Nhiệm vụ | Đầu vào | Đầu ra | Loại học |
|---------|----------|---------|--------|----------|
| CNN (ResNet18) | Phân loại từng crop (người vs nền) | Ảnh 64×64 | Nhãn lớp (0: nền, 1: người) | Supervised |
| Faster R-CNN | Phát hiện vị trí & vẽ bounding box | Ảnh gốc 384×288 | Tọa độ bbox + độ tin cậy | Supervised |
| Mask R-CNN | Phân khúc từng người thành mặt nạ pixel | Ảnh gốc 384×288 | Mặt nạ nhị phân + bbox cho mỗi người | Supervised |
| AutoEncoder | Tái tạo ảnh (nén & giải nén đặc trưng) | Ảnh 64×64 | Ảnh tái tạo 64×64 | Unsupervised |
| GAN (DCGAN) | Tạo ảnh người giả mạo từ latent vector | Noise ngẫu nhiên (100 chiều) | Ảnh tổng hợp 64×64 giống người thật | Unsupervised |

---

## 📊 Tập Dữ Liệu Penn-Fudan

### Cấu trúc dữ liệu
```
PennFudanPed/
├── PNGImages/          # Ảnh gốc (384×288 pixels)
│   ├── FudanPed00001.png
│   ├── FudanPed00002.png
│   └── ... (74 ảnh từ Fudan)
├── PedMasks/           # Mặt nạ (mask) nhị phân cho mỗi ảnh
│   ├── FudanPed00001_mask.png
│   ├── FudanPed00002_mask.png
│   └── ...
├── Annotation/         # File văn bản với tọa độ bounding box
│   ├── FudanPed00001.txt
│   └── ...
└── crops64/            # Ảnh cắt 64×64 được tạo từ bounding boxes
    ├── FudanPed00001_0.png
    ├── FudanPed00001_1.png
    └── ... (~2000 ảnh cắt)
```

### Thông tin chi tiết
- **Tổng ảnh gốc**: 124 ảnh (74 từ Fudan, 50 từ Penn)
- **Kích thước ảnh**: 384×288 pixels
- **Số lượng người**: ~345 người
- **Trung bình/ảnh**: ~2 người
- **Mặt nạ**: Mỗi ảnh có 1 file `_mask.png` với ID cho mỗi người
- **Phân chia dữ liệu**:
  - Train: 80% (99 ảnh)
  - Test: 20% (25 ảnh)

### Tiền xử lý dữ liệu

#### 1. Tạo Ảnh Cắt 64×64
```python
def load_target(mask_p):
    """
    Chuyên đổi file mask thành:
    - Bounding boxes (tọa độ hình chữ nhật)
    - Labels (nhãn lớp = 1 cho người)
    - Masks (mặt nạ nhị phân)
    """
    mask = np.array(Image.open(mask_p))
    obj_ids = np.unique(mask)[1:]  # ID của mỗi người
    
    # Tạo mặt nạ nhị phân cho mỗi người
    masks = (mask[..., None] == obj_ids).astype(np.uint8).transpose(2,0,1)
    
    # Tính bounding box từ mặt nạ
    boxes = []
    for m in masks:
        pos = np.argwhere(m)  # Tìm tất cả pixel = 1
        y1, x1 = pos.min(0)
        y2, x2 = pos.max(0)
        boxes.append([x1, y1, x2, y2])
    
    return boxes, labels, masks
```

#### 2. Cắt ảnh từ Bounding Boxes
```
Quy trình:
  1. Đọc ảnh gốc
  2. Lấy bounding boxes
  3. Cắt mỗi người theo box
  4. Resize thành 64×64
  5. Lưu thành file PNG riêng
  
Kết quả: ~2000 ảnh cắt để dùng cho CNN/AE/GAN
```

---

##  5 Mô Hình Deep Learning

### 1️⃣ CNN - ResNet18 (Phân Loại)

#### Mục đích
- **Phân loại ảnh 64×64**: Có phải người hay không?

- Đầu ra: 2 lớp (person=1, non-person=0)

#### Kiến trúc
```
Input (3×64×64)
    ↓
ResNet18 (pre-trained = False)
    ├── Layer 1-4: Residual blocks
    └── FC layers: 512 → 2 classes
    ↓
Output: Logits [batch_size, 2]
```

#### Thông số
- **Số tham số**: ~11.2M
- **Epoch**: 10
- **Batch size**: 32
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Cross Entropy Loss
- **Dataset**: 490 train + 123 val
- **Positive samples**: 423
- **Negative samples**: 190

#### Hiệu suất
```
Epoch 1: val acc=0.293
Epoch 2: val acc=0.496
Epoch 3: val acc=0.959
Epoch 4: val acc=0.919
Epoch 5: val acc=0.943
Epoch 6: val acc=0.967
Epoch 7: val acc=0.959
Epoch 8: val acc=0.984
Epoch 9: val acc=0.984
Epoch 10: val acc=0.967

---
```
### 2️⃣ Faster R-CNN (Phát Hiện)

#### Mục đích
- **Phát hiện người trong ảnh gốc**
- Output: Bounding boxes + confidence scores
- Sử dụng full image không cần cắt

#### Kiến trúc
```
Input (3×H×W)
    ↓
Backbone: ResNet50 + FPN
    └── Feature pyramid (multi-scale features)
    ↓
Region Proposal Network (RPN)
    └── Generate ~2000 proposal boxes
    ↓
ROI Pooling
    └── Extract features từ proposal
    ↓
Classification Head
    ├── Box predictor (2 classes)
    └── Bounding box regressor
    ↓
Output: 
  ├── Boxes: [N, 4] tọa độ
  ├── Scores: [N] confidence
  └── Labels: [N] class ID
```

#### Thông số
- **Backbone**: ResNet50 + FPN (Feature Pyramid Network)
- **Số tham số**: ~41.4M
- **Weights**: Pre-trained trên COCO
- **Epoch**: 2
- **Batch size**: 2 (GPU memory constraint)
- **Optimizer**: SGD (lr=0.005, momentum=0.9)
- **Loss**: RPN loss + Classification loss + Box regression loss

#### Hiệu suất
```

Epoch 1/6: 100%|██████████| 68/68 [00:36<00:00,  1.86it/s, loss=0.1450]
✅ Epoch 1/6 completed in 36.5s | Avg Loss: 17.6114

Epoch 2/6: 100%|██████████| 68/68 [00:36<00:00,  1.84it/s, loss=0.1660]
✅ Epoch 2/6 completed in 37.0s | Avg Loss: 7.6883

Epoch 3/6: 100%|██████████| 68/68 [00:37<00:00,  1.82it/s, loss=0.1037]
✅ Epoch 3/6 completed in 37.3s | Avg Loss: 6.5595

Epoch 4/6: 100%|██████████| 68/68 [00:38<00:00,  1.79it/s, loss=0.1058]
✅ Epoch 4/6 completed in 38.1s | Avg Loss: 5.8970

Epoch 5/6: 100%|██████████| 68/68 [00:39<00:00,  1.72it/s, loss=0.0557]
✅ Epoch 5/6 completed in 39.6s | Avg Loss: 4.3591

Epoch 6/6: 100%|██████████| 68/68 [00:40<00:00,  1.70it/s, loss=0.0483]
✅ Epoch 6/6 completed in 40.0s | Avg Loss: 3.8907
```



### 3️⃣ Mask R-CNN (Phân Khúc)

#### Mục đích
- **Instance segmentation**: Phát hiện + phân khúc mỗi người
- Output: Mặt nạ + bounding boxes
- Tính toán chính xác hình dạng mỗi người

#### Kiến trúc
```
Input (3×H×W)
    ↓
Backbone: ResNet50 + FPN
    ├── Shared feature extraction
    └── Multi-scale feature maps
    ↓
Region Proposal Network (RPN)
    └── Generate proposals
    ↓
ROI Align (không phải ROI Pool)
    └── Chính xác hơn cho mask prediction
    ↓
Parallel Heads:
    ├── Classification Head → 2 classes
    ├── Bounding Box Regressor
    └── **Mask Head** (NEW!)
         └── FCN (Fully Convolutional Network)
             └── Output: [N, 1, 28, 28] mask per class
    ↓
Output:
  ├── Boxes: [N, 4]
  ├── Scores: [N]
  ├── Labels: [N]
  └── Masks: [N, H, W] nhị phân
```

#### Thông số
- **Số tham số**: ~44.2M
- **Mask Head**: 256 channels, FCN architecture
- **Epoch**: 6
- **Batch size**: 2
- **Optimizer**: SGD (momentum=0.9)

#### Hiệu suất
Epoch 1/6: 100%|██████████| 68/68 [00:42<00:00,  1.61it/s, loss=0.3493]
✅ Epoch 1/6 completed in 42.3s | Avg Loss: 46.6854

Epoch 2/6: 100%|██████████| 68/68 [00:43<00:00,  1.57it/s, loss=0.2328]
✅ Epoch 2/6 completed in 43.2s | Avg Loss: 21.0259

Epoch 3/6: 100%|██████████| 68/68 [00:43<00:00,  1.55it/s, loss=0.2123]
✅ Epoch 3/6 completed in 43.9s | Avg Loss: 15.8942

Epoch 4/6: 100%|██████████| 68/68 [00:43<00:00,  1.56it/s, loss=0.1667]
✅ Epoch 4/6 completed in 43.7s | Avg Loss: 14.0750

Epoch 5/6: 100%|██████████| 68/68 [00:43<00:00,  1.55it/s, loss=0.2101]
✅ Epoch 5/6 completed in 44.0s | Avg Loss: 12.2623

Epoch 6/6: 100%|██████████| 68/68 [00:44<00:00,  1.53it/s, loss=0.1160]
✅ Epoch 6/6 completed in 44.6s | Avg Loss: 11.3528


#### Khác biệt so với Faster R-CNN
| Tiêu chí | Faster R-CNN | Mask R-CNN |
|----------|--------------|-----------|
| Đầu ra | Boxes + Scores | Boxes + Scores + **Masks** |
| ROI Pool | Coarse | Fine (ROI Align) |
| Ứng dụng | Detection | Instance Segmentation |
| Độ phức tạp | Thấp | Cao hơn |

---

### 4️⃣ AutoEncoder (Tái Tạo Ảnh)

#### Mục đích
- **Học biểu diễn không giám sát** từ ảnh 64×64
- **Nén dữ liệu**: Giảm từ 3×64×64 → 128 (compressed code)
- **Phát hiện bất thường**: So sánh MSE reconstruction
- **Feature extraction**: Dùng encoder cho downstream tasks

#### Kiến trúc
```
Input (3×64×64)
    ↓
ENCODER:
  Conv2d(3→32, kernel=4, stride=2) + ReLU   → 32×32×32
  Conv2d(32→64, kernel=4, stride=2) + ReLU  → 64×16×16
  Conv2d(64→128, kernel=4, stride=2) + ReLU → 128×8×8
    ↓
Bottleneck (Compressed code)
    ↓
DECODER:
  ConvTranspose2d(128→64, kernel=4, stride=2) + ReLU  → 64×16×16
  ConvTranspose2d(64→32, kernel=4, stride=2) + ReLU   → 32×32×32
  ConvTranspose2d(32→3, kernel=4, stride=2) + Sigmoid → 3×64×64
    ↓
Output (3×64×64) - Ảnh tái tạo
```

#### Thông số
- **Số tham số**: ~2.1M (rất nhẹ)
- **Dimension**: 3×64×64 → 128×8×8 (compression ratio: ~150x)
- **Epoch**: 30
- **Batch size**: 64
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: MSE (Mean Squared Error)

#### Kết quả
```
poch  1/30: L1 Loss=0.1251 | Best=0.1251
Epoch  5/30: L1 Loss=0.0426 | Best=0.0426
Epoch 10/30: L1 Loss=0.0328 | Best=0.0328
Epoch 15/30: L1 Loss=0.0314 | Best=0.0310
Epoch 20/30: L1 Loss=0.0300 | Best=0.0276
Epoch 25/30: L1 Loss=0.0304 | Best=0.0274
Epoch 30/30: L1 Loss=0.0254 | Best=0.0245
```

#### Ứng dụng
- **Anomaly Detection**: Nếu MSE > threshold → bất thường
- **Data Compression**: Dùng encoder để nén ảnh
- **Feature Learning**: Encoder layers làm feature extractor

---

### 5️⃣ GAN - DCGAN (Tạo Ảnh Tổng Hợp)

#### Mục đích
- **Tạo ảnh người 64×64 thực tế từ noise ngẫu nhiên**
- **Data augmentation**: Tạo training data thêm
- **Privacy-preserving**: Ảnh tổng hợp thay thế ảnh thật
- **Chứng minh học không giám sát**: Generator học phân phối dữ liệu

#### Kiến trúc

##### Generator (G)
```
Input: Random noise z ~ N(0,1), shape [batch, 64, 1, 1]
    ↓
ConvTranspose2d(64, 512, kernel=4, stride=1, pad=0) + ReLU  → 512×4×4
ConvTranspose2d(512, 256, kernel=4, stride=2, pad=1) + ReLU → 256×8×8
ConvTranspose2d(256, 128, kernel=4, stride=2, pad=1) + ReLU → 128×16×16
ConvTranspose2d(128, 3, kernel=4, stride=2, pad=1) + Tanh   → 3×64×64
    ↓
Output: Fake image ~ [-1, 1] (Tanh output)
```

##### Discriminator (D)
```
Input: Real/Fake image, shape [batch, 3, 64, 64]
    ↓
Conv2d(3, 64, kernel=4, stride=2, pad=1) + LeakyReLU(0.2)                → 64×32×32
Conv2d(64, 128, kernel=4, stride=2, pad=1) + BatchNorm2d + LeakyReLU     → 128×16×16
Conv2d(128, 256, kernel=4, stride=2, pad=1) + BatchNorm2d + LeakyReLU    → 256×8×8
Conv2d(256, 1, kernel=4, stride=1, pad=0)                                → 1×1×1
    ↓
Output: Logit (điểm thực/giả)
```


#### Thông số
- **Generator param**: ~1.7M
- **Discriminator param**: ~1.8M
- **Latent dimension (nz)**: 64
- **Epoch**: 100
- **Batch size**: 64
- **Optimizer**:
  - Generator: Adam (lr=2e-4, beta1=0.5)
  - Discriminator: Adam (lr=2e-4, beta1=0.5)
- **Loss**: BCEWithLogitsLoss

#### Kết quả Huấn Luyện
```
Epoch   1/100: D Loss=-42.6121 | G Loss=32.3636 | GP=0.2006
Epoch   5/100: D Loss=-82.3951 | G Loss=106.1968 | GP=24.9636
Epoch  10/100: D Loss=-66.7601 | G Loss=79.8856 | GP=22.8528
Epoch  15/100: D Loss=-60.7648 | G Loss=65.7075 | GP=23.0156
Epoch  20/100: D Loss=-58.8079 | G Loss=62.4049 | GP=21.8783
Epoch  25/100: D Loss=-47.9745 | G Loss=33.5453 | GP=16.7308
Epoch  30/100: D Loss=-43.2105 | G Loss=24.1113 | GP=14.5866
Epoch  35/100: D Loss=-42.3529 | G Loss=3.6158 | GP=13.5290
Epoch  40/100: D Loss=-40.2293 | G Loss=-7.4197 | GP=13.2516
Epoch  45/100: D Loss=-35.9524 | G Loss=-9.3946 | GP=12.0025
Epoch  50/100: D Loss=-33.0638 | G Loss=-7.7295 | GP=9.6321
Epoch  55/100: D Loss=-31.4066 | G Loss=8.2952 | GP=9.7162
Epoch  60/100: D Loss=-29.4300 | G Loss=-0.5442 | GP=9.1760
Epoch  65/100: D Loss=-28.4854 | G Loss=-6.4294 | GP=8.4540
Epoch  70/100: D Loss=-25.3650 | G Loss=-13.6383 | GP=7.0295
Epoch  75/100: D Loss=-25.7261 | G Loss=-21.2284 | GP=7.1632
...
Epoch  95/100: D Loss=-23.0451 | G Loss=-19.4365 | GP=5.5377
Epoch 100/100: D Loss=-22.6645 | G Loss=-19.0220 | GP=5.3548

```

#### Ứng dụng
- **Data Augmentation**: Tạo thêm ảnh training
- **Privacy**: Tạo ảnh fake thay cho ảnh thật
- **Simulation**: Tạo dataset tổng hợp

---

## ⚙️ Quy Trình Xử Lý Dữ Liệu

### Sơ đồ tổng thể
```
1. Load Dataset
   └── PennFudanPed/
       ├── PNGImages/*.png
       ├── PedMasks/*_mask.png
       └── Annotation/*.txt

2. Preprocess
   ├── Parse mask → extract boxes, masks, labels
   └── Create 64×64 crops from bounding boxes

3. Create Dataloaders
   ├── CNN Dataset: PedCropDataset (64×64, labels)
   ├── Faster R-CNN: PennFudanDet (full image, boxes)
   ├── Mask R-CNN: PennFudanSeg (full image, boxes+masks)
   ├── AutoEncoder: CropOnly (64×64, no labels)
   └── GAN: CropOnly (same as AE)

4. Train Models
   ├── CNN (3 epochs)
   ├── Faster R-CNN (2 epochs)
   ├── Mask R-CNN (2 epochs)
   ├── AutoEncoder (3 epochs)
   └── GAN (3 epochs)

5. Generate Visualizations
   ├── CNN_Results.png
   ├── RCNN_Detection.png
   ├── MaskRCNN_Segmentation.png
   ├── AE_Reconstruction.png
   ├── GAN_Generated.png
   ├── DEMO_Full_Pipeline.png
   ├── Performance_Analysis.png
   └── CNN_Feature_Maps.png

6. Analysis
   └── Compare models, show trade-offs
```

### Custom Collate Function

Vì batch chứa ảnh kích thước khác nhau, cần custom collate:

```python
def collate(batch):
    """
    Batch là danh sách (img, target) tuples
    Target là dict với 'boxes', 'labels', 'masks'
    Return: (list of images, list of targets)
    """
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)
```

**Tại sao cần?** 
- Ảnh gốc có kích thước 384×288, khác nhau
- PyTorch batch cần tensor cùng kích thước
- Collate function trả về list thay vì tensor

---

## 📊 Chi Tiết Các Mô Hình

### Tổng Hợp Thông Số

| Tiêu chí | CNN | Faster R-CNN | Mask R-CNN | AE | GAN |
|----------|-----|--------------|-----------|----|----|
| **Nhiệm vụ** | Phân loại | Phát hiện | Phân khúc | Tái tạo | Tạo |
| **Đầu vào** | 64×64 | Full image | Full image | 64×64 | Noise |
| **Đầu ra** | Class label | Boxes | Masks+Boxes | Ảnh tái tạo | Ảnh giả |
| **Số tham số** | 11.2M | 41.4M | 44.2M | 2.1M | 3.5M |
| **Epoch** | 3 | 2 | 2 | 3 | 3 |
| **Accuracy/Loss** | 100% | - | - | MSE: 0.0293 | Balanced |
| **Backbone** | ResNet18 | ResNet50+FPN | ResNet50+FPN | - | - |

### Yêu Cầu Tài Nguyên

| Thành phần | Yêu cầu |
|-----------|---------|
| **RAM** | ≥8GB (16GB recommended) |
| **GPU** | NVIDIA với CUDA (hoặc CPU chậm) |
| **Thời gian huấn luyện** | ~30-40 phút (với GPU) |
| **Dung lượng model** | ~200MB (tất cả model) |
| **Dataset size** | ~200MB |

---

## 🎨 Kết Quả và Visualization

### 1. CNN_Results.png
**Nội dung**: 8 ảnh cắt 64×64 và dự đoán class
- Cột trên: Original samples
- Cột dưới: CNN predictions
- Tiêu đề xanh: Dự đoán đúng
- Tiêu đề đỏ: Dự đoán sai
- **Kết quả**: 100% accuracy!

### 2. RCNN_Detection.png
**Nội dung**: 2 ảnh gốc full size
- **Xanh**: Ground truth boxes
- **Đỏ nét**: Predicted boxes
- **Score**: Confidence score mỗi detection

### 3. MaskRCNN_Segmentation.png
**Nội dung**: 2×2 grid
- **Hàng trên**: Ground truth masks (xanh)
- **Hàng dưới**: Predicted masks (đỏ nét)
- **Chiều rộng contour**: Nhìn rõ ranh giới

### 4. AE_Reconstruction.png
**Nội dung**: 2×8 grid so sánh
- **Hàng trên**: Ảnh gốc
- **Hàng dưới**: Ảnh tái tạo từ AE
- **Metrics**: 
  - Average MSE: 0.0293
  - Range: [0.0245, 0.0369]

### 5. GAN_Generated.png
**Nội dung**: 2×8 grid (16 ảnh giả)
- **Tất cả**: Ảnh tổng hợp từ GAN
- **Không có labels**: Chỉ show ảnh tạo ra
- **Quality**: Tăng dần qua epochs

### 6. DEMO_Full_Pipeline.png
**Nội dung**: 3×3 grid (9 panels)
```
Row 1:
  [1] Original Image    [2] GT Mask          [3] GT Boxes
Row 2:
  [4] R-CNN Detections  [5] Mask Segmentation [6] Combined
Row 3:
  [7] CNN Crops         [8] AE Reconstruction [9] GAN Generated
```
**Ý nghĩa**: Tổng hợp tất cả 5 mô hình trên 1 ảnh

### 7. Performance_Analysis.png
**Nội dung**: 2×2 charts
- **Top-left**: Model complexity (số tham số)
- **Top-right**: Task capability matrix
- **Bottom-left**: Speed vs Quality trade-off
- **Bottom-right**: Applications & use cases

### 8. CNN_Feature_Maps.png
**Nội dung**: Feature map từ CNN layers
- **Hàng 1**: Conv layer 1 (32 channels → show 8)
- **Hàng 2**: Conv layer 2 (64 channels → show 8)
- **Hàng 3**: Conv layer 3 (128 channels → show 8)
- **Colormap**: 'hot' (đen → đỏ)

---

## 🚀 Hướng Dẫn Chạy Code

### Bước 1: Setup Environment
```bash
# Cài đặt thư viện cần thiết
pip install torch torchvision
pip install pillow numpy matplotlib pandas
pip install jupyter  # Nếu dùng Jupyter

# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Bước 2: Chuẩn Bị Dữ Liệu
```bash
# Download Penn-Fudan Dataset
# Từ: http://www.cis.upenn.edu/~jshi/ped_html/

# Giải nén vào thư mục:
PennFudanPed/
├── PNGImages/
├── PedMasks/
└── Annotation/
```

### Bước 3: Cấu Hình Đường Dẫn
Trong **Cell 1**, sửa:
```python
root = r"./PennFudanPed"  # Đổi thành đường dẫn thực tế
```

### Bước 4: Chạy từng Cell

#### Cell 1: Load & Preprocess
```python
# Tạo thư mục crops64 từ PennFudanPed/
# Kết quả: ~2000 ảnh cắt 64×64
```

#### Cell 2: Train CNN
```python
# Huấn luyện ResNet18
# Thời gian: ~2-3 phút
# Output: model, train_dl_cnn, val_dl_cnn
```

#### Cell 3: Train Faster R-CNN
```python
# Huấn luyện detection
# Thời gian: ~10 phút
# Output: det_model, train_dl_det, val_dl_det
```

#### Cell 4: Train Mask R-CNN
```python
# Huấn luyện segmentation
# Thời gian: ~12 phút
# Output: seg_model, train_dl_seg, val_dl_seg
```

#### Cell 5: Train AutoEncoder
```python
# Huấn luyện AE
# Thời gian: ~3-4 phút
# Output: ae, ae_ds, ae_dl
```

#### Cell 6: Train GAN
```python
# Huấn luyện DCGAN
# Thời gian: ~5-7 phút
# Output: gen, disc
```

#### Cell 7-14: Visualizations
```python
# Chạy lần lượt các demo cells
# Tạo 8 PNG files
```

### Bước 5: Xem Kết Quả
```bash
# Tất cả file PNG lưu trong:
./PennFudanPed/

# Mở file:
- CNN_Results.png
- RCNN_Detection.png
- MaskRCNN_Segmentation.png
- AE_Reconstruction.png
- GAN_Generated.png
- DEMO_Full_Pipeline.png
- Performance_Analysis.png
- CNN_Feature_Maps.png
```

### Xử Lý Lỗi Thường Gặp

#### 1. "CUDA out of memory"
```python
# Giảm batch size
batch_size = 2 → 1
# Hoặc sử dụng CPU
device = "cpu"
```

#### 2. "Module not found"
```bash
pip install pillow torch torchvision
pip install numpy matplotlib pandas
```

#### 3. Dataset path không tìm thấy
```python
# Kiểm tra đường dẫn
import os
print(os.path.exists(root))
print(os.listdir(root))
```

---

## 💡 Ứng Dụng Thực Tế

### 1. Giám Sát An Toàn (Security)
```
Ứng dụng: Phát hiện người trái phép
  ├── Faster R-CNN: Phát hiện tất cả người
  ├── Mask R-CNN: Phân khúc chính xác hình dạng
  └── Alert: Thông báo khi có người ở vùng cấm
```

### 2. Đếm Đám Đông (Crowd Counting)
```
Ứng dụng: Đếm số người trong nhà ga, tàu điện
  ├── Mask R-CNN: Đếm instance từ masks
  ├── Tối ưu: Không bị xếp chồng
  └── Output: Số người chính xác
```

### 3. Phân Tích Hành Vi (Behavior Analysis)
```
Ứng dụng: Phát hiện hoạt động bất thường
  ├── CNN: Phân loại từng người
  ├── AutoEncoder: Phát hiện anomaly
  └── Alert: Khi bất thường được phát hiện
```

### 4. Hệ Thống Tranh Cấp (Access Control)
```
Ứng dụng: Kiểm soát ra vào cơ sở
  ├── Faster R-CNN: Phát hiện người
  ├── Mask R-CNN: Xác định hình dạng, tư thế
  └── Compare: So với dữ liệu base
```

### 5. Tạo Dataset (Data Augmentation)
```
Ứng dụng: Mở rộng training data
  ├── GAN: Tạo ảnh nhân tạo
  ├── Lợi ích: Bảo mật, đa dạng hóa
  └── Training: Dùng ảnh tổng hợp huấn luyện
```

### 6. Phát Hiện Bất Thường (Anomaly)
```
Ứng dụng: Tìm người lạ hoặc hoạt động kỳ lạ
  ├── AutoEncoder: Học pattern bình thường
  ├── MSE Error: Nếu cao → bất thường
  └── Alert: Kích hoạt khi vượt threshold
```

### 7. Phân Tích Lưu Lượng (Traffic Flow)
```
Ứng dụng: Theo dõi luồng người di chuyển
  ├── Mask R-CNN: Theo dõi từng người
  ├── Temporal tracking: Ghi lại đường đi
  └── Analytics: Vị trí, hướng, tốc độ
```

### 8. Tối Ưu Hóa Không Gian (Space Optimization)
```
Ứng dụng: Phân bố con người hợp lý
  ├── Crowd counting: Số người thực tế
  ├── Heatmap: Nơi tập trung
  └── Decision: Mở thêm entrance/exit
```

---

## 📚 Tài Liệu Tham Khảo

1. **Penn-Fudan Dataset**: http://www.cis.upenn.edu/~jshi/ped_html/
2. **Faster R-CNN**: Ren et al., NIPS 2015
3. **Mask R-CNN**: He et al., ICCV 2017
4. **ResNet**: He et al., CVPR 2016
5. **GAN**: Goodfellow et al., NIPS 2014
6. **PyTorch Documentation**: https://pytorch.org/
