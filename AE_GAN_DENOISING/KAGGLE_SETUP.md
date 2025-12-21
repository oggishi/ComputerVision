# 🚀 Hướng dẫn chạy Notebook trên Kaggle

## 📌 Tóm tắt thay đổi cho Kaggle

Notebook đã được chỉnh sửa để chạy tối ưu trên Kaggle Notebooks với các tính năng:

### ✅ Tính năng tự động điều chỉnh:
- **Detect môi trường**: Notebook tự động phát hiện nếu chạy trên Kaggle
- **Đường dẫn tự động**: Sử dụng `/kaggle/working/` cho output
- **Batch size tự động**: Giảm từ 16 → 8 trên Kaggle để tiết kiệm RAM
- **Num workers tự động**: Sử dụng `num_workers=0` trên Kaggle (bắt buộc)
- **Pin memory tự động**: Chỉ kích hoạt khi sử dụng GPU

---

## 📁 Cấu trúc dữ liệu yêu cầu

Dữ liệu phải theo format **ImageFolder** của PyTorch:

```
thumbnails/
├── classA/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── classB/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...
```

---

## 🔧 Cách 1: Upload Dataset riêng (RECOMMENDED)

### Bước 1: Chuẩn bị dữ liệu
```
thumbnails/
├── classA/    (ảnh class A)
└── classB/    (ảnh class B)
```

### Bước 2: Tạo Kaggle Dataset
1. Đăng nhập vào [Kaggle.com](https://kaggle.com)
2. Vào "Your Work" → "Datasets"
3. Ấn "Create new dataset"
4. Upload thư mục `thumbnails/`

### Bước 3: Tạo Kaggle Notebook
1. Vào dataset vừa tạo
2. Ấn "New notebook"
3. Mở notebook này

### Bước 4: Gắn Dataset vào Notebook
1. Ấn "Add data" → "Kaggle Datasets"
2. Tìm và chọn dataset vừa tạo
3. Ấn "Add"

### Bước 5: Chỉnh sửa đường dẫn
Trong cell **"CẤU HÌNH CHO KAGGLE"**, sửa:

```python
# Thay YOUR-DATASET-NAME bằng tên dataset của bạn
thu_muc_du_lieu = "/kaggle/input/YOUR-DATASET-NAME/thumbnails"
```

---

## 🌐 Cách 2: Sử dụng Dataset Kaggle công cộng

Nếu dataset bạn đã được public trên Kaggle:

```python
# Ví dụ
thu_muc_du_lieu = "/kaggle/input/imagenet-object-localization-challenge/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train"
```

---

## ⚙️ Các thay đổi chính

### 1. **Tự động detect Kaggle**
```python
IN_KAGGLE = os.path.exists('/kaggle')
```

### 2. **Đường dẫn thích ứng**
```python
if IN_KAGGLE:
    thu_muc_du_lieu = "/kaggle/input/..."
    thu_muc_ket_qua = "/kaggle/working/outputs_denoise"
else:
    thu_muc_du_lieu = "./thumbnails"
    thu_muc_ket_qua = "./outputs_denoise"
```

### 3. **Batch size tối ưu**
```python
batch_size = 16 if not IN_KAGGLE else 8  # Giảm trên Kaggle
```

### 4. **Num workers hợp lệ**
```python
so_worker = 2 if not IN_KAGGLE else 0  # 0 trên Kaggle
```

### 5. **Checkpoint paths**
```python
duong_dan_checkpoint_ae = "/kaggle/working/best_ae_model.pth"
duong_dan_checkpoint_gan_g = "/kaggle/working/best_gan_generator.pth"
duong_dan_checkpoint_gan_d = "/kaggle/working/best_gan_discriminator.pth"
```

---

## 📊 Output

Tất cả output sẽ được lưu vào `/kaggle/working/`:

```
outputs_denoise/
├── viz_epoch_00.png
├── viz_epoch_01.png
├── ...
├── best_ae_model.pth              # Checkpoint Autoencoder
├── best_gan_generator.pth         # Checkpoint GAN Generator
├── best_gan_discriminator.pth     # Checkpoint GAN Discriminator
├── comparison_training.png        # Đồ thị so sánh
└── ae_vs_gan_comparison.png       # Hình ảnh so sánh kết quả
```

---

## 🎯 Tham số chủ yếu

| Tham số | Local | Kaggle | Mô tả |
|--------|-------|--------|-------|
| `batch_size` | 16 | 8 | Số ảnh trong mỗi batch |
| `so_epoch` | 20 | 20 | Số epoch huấn luyện |
| `kich_thuoc_anh` | 128 | 128 | Kích thước ảnh input |
| `so_worker` | 2 | 0 | Số worker load dữ liệu |
| `learning_rate` | 1e-3 | 1e-3 | Tốc độ học |

---

## ❓ Troubleshooting

### ❌ "Lỗi: Thư mục không tồn tại"
**Giải pháp**: 
- Kiểm tra dataset đã được gắn vào notebook chưa
- Kiểm tra tên dataset trong URL gắn dữ liệu
- Chỉnh sửa đường dẫn `thu_muc_du_lieu` chính xác

### ❌ "Out of Memory" (OOM)
**Giải pháp**:
- Giảm `batch_size` từ 8 → 4
- Giảm `so_epoch` từ 20 → 10
- Giảm `kich_thuoc_anh` từ 128 → 64

### ❌ "DataLoader lỗi với num_workers"
**Giải pháp**: Notebook sẽ tự động sử dụng `num_workers=0` trên Kaggle ✅

### ❌ "Checkpoint không được lưu"
**Giải pháp**: 
- Kiểm tra folder `/kaggle/working/` tồn tại
- Notebook sẽ tự động tạo folder này ✅

---

## 💡 Mẹo

1. **Download output**: Ấn "Save Version" → "Save" để lưu notebook, sau đó download các file từ `/kaggle/working/`
2. **Chạy nhanh**: Đặt `so_epoch = 5` để test trước, sau đó tăng lên 20
3. **GPU**: Chọn "Accelerator: GPU P100" trong Settings để tăng tốc độ
4. **Monitor RAM**: Watch RAM usage với `nvidia-smi` (GPU) hoặc `free -h` (CPU)

---

## 📚 Tài liệu tham khảo

- [Kaggle Notebooks Docs](https://www.kaggle.com/docs/notebooks)
- [Kaggle API](https://www.kaggle.com/docs/api)
- [PyTorch ImageFolder](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder)

---

**Happy training on Kaggle! 🚀**
