# 🧠 Brain Tumor MRI Classification using CNN (Multi-Class)

This project is a multi-class classification task using Convolutional Neural Networks (CNNs) to detect and classify brain tumors from MRI images. The four classes include:

- `glioma_tumor`
- `meningioma_tumor`
- `pituitary_tumor`
- `no_tumor`

---

## 📂 Dataset

The dataset used in this project is the [Brain Tumor MRI Images Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).  
We selected **5,000 images** from the full dataset for initial experimentation, split into:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

The images were preprocessed and resized to a uniform shape.

---

## ⚙️ Project Structure


---

## 🧪 Preprocessing & Augmentation

- Converted images to grayscale (optional)
- Resized to 150x150
- Normalized pixel values between 0 and 1
- Applied image augmentation using `ImageDataGenerator` for training

---

## 🧠 Model Architecture

We used a custom CNN architecture with the following layers:

- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten → Dense (128) → ReLU
- Dropout (0.5)
- Output: Dense (4) → Softmax

**Loss Function:** `categorical_crossentropy`  
**Optimizer:** `Adam`  
**Metrics:** `accuracy`

---

## 🏋️ Model Training

```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)
Trained for 10 epochs

Achieved good convergence

Validation accuracy: ~77% in initial runs

Test set accuracy varies (latest: 28%, possibly due to imbalance or preprocessing mismatch)
