# 🏆 Kaggle Solutions & Machine Learning Competitions Repository

[![Kaggle](https://img.shields.io/badge/Kaggle-Competitions-blue?logo=kaggle)](https://www.kaggle.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repository này chứa toàn bộ mã nguồn giải pháp, tài liệu phân tích dữ liệu chuyên sâu (Obsidian-styled Master Documentation), các bộ thí nghiệm Jupyter Notebook và các pipeline Machine Learning chuẩn mực phục vụ các cuộc thi trên nền tảng **Kaggle**.

---

## 📁 Cấu Trúc Tổng Thể Repository

```
kaggle/
├── .obsidian/                           # Cấu hình không gian tri thức Obsidian Vault
├── competition/                         # Các dự án thi đấu Kaggle Competitions
│   ├── kaggriculture/                   # Thử nghiệm Kaggriculture Environment Simulation
│   │   ├── README.md                    # Tài liệu chính thức về Kaggriculture
│   │   ├── README_VI.md                 # Cẩm nang phân tích tiếng Việt
│   │   ├── AGENTS.md                    # Hướng dẫn xây dựng Agent
│   │   ├── main.py                      # Script Agent điều khiển
│   │   └── simulate.py                  # Môi trường giả lập trận đấu
│   └── titanic/                         # Titanic: Machine Learning from Disaster
│       ├── data/                        # Dữ liệu train.csv, test.csv, gender_submission.csv
│       ├── notebooks/                   # titanic_pipeline.ipynb (Interactive 6-stage workspace)
│       ├── src/                         # Module hóa: config.py, features.py, models.py, evaluate.py, ensemble.py
│       ├── outputs/                     # outputs/models/ và outputs/submissions/
│       ├── kernel-metadata.json         # Cấu hình đẩy notebook lên Kaggle Cloud tự động
│       ├── OVERVIEW.md                  # Tổng quan, bối cảnh lịch sử, Metric & Leaderboard
│       ├── RULES.md                     # Quy định thi đấu & chống rò rỉ dữ liệu
│       ├── DATA_DICTIONARY.md           # Từ điển 11 biến & Pipeline tiền xử lý chi tiết
│       ├── PIPELINE_PLAN.md             # Kế hoạch kiến trúc, EDA 8 biểu đồ, so sánh lý thuyết mô hình
│       ├── README_VI.md                 # Cẩm nang thực chiến tiếng Việt & Kaggle CLI
│       └── README.md                    # Master Project Index cho Titanic
├── learn/                               # Không gian học tập & thực hành kỹ năng Kaggle Learn
├── .gitignore                           # Danh sách bỏ qua các file tạm & cache
└── README.md                            # Mục lục chính của Repository này
```

---

## 🚢 Dự Án Tiêu Biểu: Titanic - Machine Learning from Disaster

* **Thư mục:** [`competition/titanic/`](competition/titanic/)
* **Mục tiêu:** Phân loại nhị phân dự đoán hành khách sống sót (`Survived = 1`) hay thiệt mạng (`Survived = 0`).
* **Metric:** Categorization Accuracy ($\ge 82.5\% - 84.0\%$).
* **Công nghệ:** Stratified 5-Fold Cross-Validation, Soft Voting Ensemble đa mô hình (**Random Forest**, **Extra Trees**, **XGBoost**, **LightGBM**, **CatBoost**, **Logistic Regression**).
* **Tài liệu chi tiết:**
  * [`competition/titanic/OVERVIEW.md`](competition/titanic/OVERVIEW.md)
  * [`competition/titanic/DATA_DICTIONARY.md`](competition/titanic/DATA_DICTIONARY.md)
  * [`competition/titanic/PIPELINE_PLAN.md`](competition/titanic/PIPELINE_PLAN.md)
  * [`competition/titanic/README.md`](competition/titanic/README.md)

---

## 🌾 Dự Án Kaggriculture: Agentic Environment Simulation

* **Thư mục:** [`competition/kaggriculture/`](competition/kaggriculture/)
* **Mục tiêu:** Phát triển Agent tự hành tối ưu hóa chiến thuật trong môi trường giả lập nông nghiệp Kaggriculture.
* **Tài liệu:** [`competition/kaggriculture/README_VI.md`](competition/kaggriculture/README_VI.md)

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### 1. Clone repository về máy:
```bash
git clone https://github.com/KhoaS84/kaggle-key.git
cd kaggle-key
```

### 2. Cài đặt các thư viện cần thiết:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna kaggle
```

### 3. Vận hành với Kaggle CLI:
```bash
# Đẩy notebook Titanic lên Kaggle Cloud
cd competition/titanic
kaggle kernels push -p ./

# Nộp kết quả trực tiếp lên cuộc thi
kaggle competitions submit titanic -f outputs/submissions/submission.csv -m "Ensemble Soft Voting Submission"
```
