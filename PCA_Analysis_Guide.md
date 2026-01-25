# 주성분 분석(PCA: Principal Component Analysis) 완벽 가이드

## 목차
1. [PCA 개요](#1-pca-개요)
2. [PCA 수학적 원리](#2-pca-수학적-원리)
3. [USArrests 데이터 PCA 분석 (R)](#3-usarrests-데이터-pca-분석-r)
4. [USArrests 데이터 PCA 분석 (Python)](#4-usarrests-데이터-pca-분석-python)
5. [Biplot 해석 방법](#5-biplot-해석-방법)
6. [주성분 수 결정 방법](#6-주성분-수-결정-방법)
7. [PCA 활용 분야](#7-pca-활용-분야)
8. [빅데이터분석기사 핵심 포인트](#8-빅데이터분석기사-핵심-포인트)

---

## 1. PCA 개요

### 1.1 정의
주성분 분석(PCA)은 고차원 데이터를 **저차원으로 축소**하면서 데이터의 **분산(정보)을 최대한 보존**하는 비지도 학습 기법입니다.

### 1.2 목적
- **차원 축소 (Dimensionality Reduction)**: 변수 수 감소
- **다중공선성 제거**: 상관된 변수들을 독립적인 주성분으로 변환
- **데이터 시각화**: 고차원 데이터를 2D/3D로 시각화
- **노이즈 제거**: 분산이 작은 성분 제거로 노이즈 감소
- **특성 추출 (Feature Extraction)**: 새로운 특성 생성

### 1.3 PCA의 특징
| 특징 | 설명 |
|------|------|
| 비지도 학습 | 레이블(y) 없이 X만으로 분석 |
| 선형 변환 | 원래 변수들의 선형 결합 |
| 직교성 | 주성분들은 서로 직교(상관관계 0) |
| 분산 최대화 | 첫 번째 주성분이 가장 큰 분산 설명 |

---

## 2. PCA 수학적 원리

### 2.1 기본 개념

#### 공분산 행렬 (Covariance Matrix)
```
Σ = (1/(n-1)) * X^T * X  (X는 중심화된 데이터)
```

#### 고유값 분해 (Eigendecomposition)
```
Σv = λv

- Σ: 공분산 행렬 (p × p)
- λ: 고유값 (eigenvalue) - 해당 주성분의 분산
- v: 고유벡터 (eigenvector) - 주성분의 방향 (loadings)
```

### 2.2 PCA 계산 절차

```
1. 데이터 표준화 (평균 0, 분산 1)
   Z = (X - μ) / σ

2. 공분산 행렬 계산
   Σ = Z^T * Z / (n-1)

3. 고유값, 고유벡터 계산
   Σ = VΛV^T

4. 고유값 크기 순으로 정렬

5. 상위 k개 고유벡터 선택

6. 데이터 변환
   PC = Z * V_k
```

### 2.3 주요 용어

| 용어 | 영문 | 의미 |
|------|------|------|
| 주성분 | Principal Component | 새로운 축(변환된 변수) |
| 고유값 | Eigenvalue | 각 주성분이 설명하는 분산 |
| 고유벡터 | Eigenvector | 주성분의 방향 |
| 로딩 | Loadings | 원래 변수가 주성분에 기여하는 가중치 |
| 스코어 | Scores | 변환된 데이터 좌표 |
| 설명분산비율 | Proportion of Variance | 각 PC가 설명하는 분산 비율 |
| 누적분산비율 | Cumulative Proportion | 누적 설명 분산 |

---

## 3. USArrests 데이터 PCA 분석 (R)

### 3.1 데이터 탐색

```r
# 데이터 로드
data(USArrests)

# 데이터 구조 확인
str(USArrests)
# 'data.frame':	50 obs. of  4 variables:
#  $ Murder  : num  13.2 10 8.1 8.8 9 7.9 3.3 5.9 15.4 17.4 ...
#  $ Assault : int  236 263 294 190 276 204 110 238 335 211 ...
#  $ UrbanPop: int  58 48 80 50 91 78 77 72 80 60 ...
#  $ Rape    : num  21.2 44.5 31 19.5 40.6 38.7 11.1 15.8 31.9 25.8 ...

# 기술통계
summary(USArrests)

# 변수 간 상관관계
cor(USArrests)
#              Murder   Assault   UrbanPop      Rape
# Murder   1.0000000 0.8018733 0.06957262 0.5635788
# Assault  0.8018733 1.0000000 0.25887170 0.6652412
# UrbanPop 0.06957262 0.25887170 1.00000000 0.4113412
# Rape     0.5635788 0.6652412 0.41134124 1.0000000
```

### 3.2 PCA 수행

```r
# PCA 분석 (scale = TRUE: 표준화)
pca_result <- prcomp(USArrests, scale = TRUE, center = TRUE)

# 결과 요약
summary(pca_result)
# Importance of components:
#                           PC1    PC2     PC3     PC4
# Standard deviation     1.5749 0.9949 0.59713 0.41645
# Proportion of Variance 0.6201 0.2474 0.08914 0.04336
# Cumulative Proportion  0.6201 0.8675 0.95664 1.00000
```

### 3.3 결과 해석

```r
# 1. 고유값 (분산)
pca_result$sdev^2
# [1] 2.4802416 0.9897652 0.3565632 0.1734301

# 2. 로딩 (고유벡터) - 변수 기여도
pca_result$rotation
#                 PC1        PC2        PC3         PC4
# Murder   -0.5358995 -0.4181809  0.3412327  0.64922780
# Assault  -0.5831836 -0.1879856 -0.2681484 -0.74340748
# UrbanPop -0.2781909  0.8728062  0.3780158  0.13387773
# Rape     -0.5434321  0.1673186 -0.8177779  0.08902432

# 3. 주성분 스코어 (변환된 데이터)
head(pca_result$x)
#                   PC1        PC2         PC3          PC4
# Alabama    -0.9756604 -1.1220012  0.43980366  0.154696581
# Alaska     -1.9305379 -1.0624269 -2.01950027 -0.434175454
# Arizona    -1.7454429  0.7384595 -0.05423025 -0.826264240
# Arkansas    0.1399989 -1.1085423 -0.11342217 -0.180973554
# California -2.4986128  1.5274267 -0.59254100 -0.338559240
# Colorado   -1.4993407  0.9776297 -1.08400162  0.001450164
```

### 3.4 시각화

```r
# 1. Scree Plot (스크리 도표)
screeplot(pca_result, type = "lines", main = "Scree Plot")

# 또는 ggplot2 사용
library(ggplot2)
var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
qplot(x = 1:4, y = var_explained, geom = "line") +
  geom_point() +
  labs(x = "Principal Component", y = "Variance Explained",
       title = "Scree Plot") +
  theme_minimal()

# 2. Biplot
biplot(pca_result, scale = 0, cex = 0.7)

# 3. ggbiplot 패키지 사용 (더 예쁜 시각화)
# install.packages("devtools")
# devtools::install_github("vqv/ggbiplot")
library(ggbiplot)
ggbiplot(pca_result, labels = rownames(USArrests)) +
  theme_minimal() +
  ggtitle("PCA Biplot of USArrests")
```

### 3.5 분산 비율 시각화

```r
# 각 주성분의 분산 비율
var_prop <- summary(pca_result)$importance[2,]
cum_prop <- summary(pca_result)$importance[3,]

# 막대 그래프
barplot(var_prop, 
        names.arg = paste0("PC", 1:4),
        main = "Variance Explained by Each PC",
        ylab = "Proportion of Variance",
        col = "steelblue")

# 누적 분산 그래프
plot(cum_prop, type = "b", 
     xlab = "Number of Components",
     ylab = "Cumulative Proportion",
     main = "Cumulative Variance Explained")
abline(h = 0.8, col = "red", lty = 2)  # 80% 기준선
```

---

## 4. USArrests 데이터 PCA 분석 (Python)

### 4.1 라이브러리 및 데이터 준비

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# USArrests 데이터 로드
# R의 USArrests와 동일한 데이터
url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/USArrests.csv"
df = pd.read_csv(url, index_col=0)

print(df.head())
print(df.describe())
```

### 4.2 데이터 표준화

```python
# 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# DataFrame으로 변환
X_scaled_df = pd.DataFrame(X_scaled, 
                           index=df.index, 
                           columns=df.columns)
```

### 4.3 PCA 수행

```python
# PCA 모델 생성 (모든 주성분 유지)
pca = PCA()
pca_scores = pca.fit_transform(X_scaled)

# 결과를 DataFrame으로
pca_df = pd.DataFrame(pca_scores, 
                      index=df.index,
                      columns=['PC1', 'PC2', 'PC3', 'PC4'])
```

### 4.4 결과 확인

```python
# 1. 설명 분산 비율
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
# [0.62006039 0.24744129 0.0891408  0.04335752]

print("\nCumulative Variance Ratio:")
print(np.cumsum(pca.explained_variance_ratio_))
# [0.62006039 0.86750168 0.95664248 1.        ]

# 2. 고유값
print("\nEigenvalues:")
print(pca.explained_variance_)
# [2.48024165 0.98976528 0.35656321 0.17343006]

# 3. 로딩 (주성분 계수)
loadings = pd.DataFrame(pca.components_.T,
                        index=df.columns,
                        columns=['PC1', 'PC2', 'PC3', 'PC4'])
print("\nLoadings:")
print(loadings)
```

### 4.5 시각화

```python
# 1. Scree Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 개별 분산
axes[0].bar(range(1, 5), pca.explained_variance_ratio_, color='steelblue')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained')
axes[0].set_title('Scree Plot')
axes[0].set_xticks(range(1, 5))

# 누적 분산
axes[1].plot(range(1, 5), np.cumsum(pca.explained_variance_ratio_), 
             'bo-', linewidth=2)
axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained')
axes[1].set_title('Cumulative Variance Explained')
axes[1].set_xticks(range(1, 5))
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 4.6 Biplot 구현

```python
def biplot(score, coeff, labels=None, var_names=None):
    """
    PCA Biplot 그리기
    
    Parameters:
    -----------
    score : array-like, shape (n_samples, 2)
        PCA 스코어 (PC1, PC2)
    coeff : array-like, shape (n_features, 2)
        PCA 로딩 (loadings)
    labels : list, optional
        샘플 이름
    var_names : list, optional
        변수 이름
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    
    # 스케일 조정
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    
    # 데이터 포인트 (주)
    ax.scatter(xs * scalex, ys * scaley, c='steelblue', alpha=0.6)
    
    # 레이블 추가
    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (xs[i] * scalex, ys[i] * scaley),
                       fontsize=8, alpha=0.7)
    
    # 로딩 벡터 (화살표)
    for i in range(n):
        ax.arrow(0, 0, coeff[i, 0], coeff[i, 1],
                color='red', alpha=0.8, head_width=0.03)
        if var_names is not None:
            ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                   var_names[i], color='red', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('PC1 ({:.1%})'.format(pca.explained_variance_ratio_[0]))
    ax.set_ylabel('PC2 ({:.1%})'.format(pca.explained_variance_ratio_[1]))
    ax.set_title('PCA Biplot - USArrests')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Biplot 그리기
biplot(pca_scores[:, :2], 
       pca.components_[:2].T,
       labels=df.index.tolist(),
       var_names=df.columns.tolist())
```

### 4.7 Seaborn을 활용한 고급 시각화

```python
# 히트맵으로 로딩 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0,
            fmt='.3f', linewidths=0.5)
plt.title('PCA Loadings Heatmap')
plt.show()

# PC1 vs PC2 산점도
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, s=50)
for i, state in enumerate(pca_df.index):
    plt.annotate(state, (pca_df['PC1'][i], pca_df['PC2'][i]),
                fontsize=8, alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA Score Plot - USArrests')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5. Biplot 해석 방법

### 5.1 Biplot의 구성 요소

Biplot은 두 가지 정보를 하나의 그래프에 표시합니다:

| 요소 | 표현 | 의미 |
|------|------|------|
| **점 (Points)** | 관측치 (주/샘플) | PC 공간에서의 좌표 |
| **화살표 (Arrows)** | 변수 (Loadings) | 변수가 PC에 기여하는 방향과 크기 |

### 5.2 화살표(Loading Vector) 해석

#### 5.2.1 화살표 방향 = 변수의 기여 방향

```
                    PC2 (+)
                      ↑
                      |    ↗ UrbanPop
                      |   
         ←————————————+————————————→ PC1 (+)
                      |
            Murder ↙  |  ↘ Assault
              Rape ↙  |
                      ↓
                    PC2 (-)
```

USArrests 데이터에서:
- **Murder, Assault, Rape**: PC1 방향으로 강하게 기여 → **PC1 = "전반적 범죄율"**
- **UrbanPop**: PC2 방향으로 기여 → **PC2 = "도시화 정도"**

#### 5.2.2 화살표 길이 = 변수의 설명력

- 화살표가 **길수록** 해당 변수가 주성분을 **잘 설명**
- 화살표가 **짧으면** 해당 변수는 선택된 PC에서 잘 표현되지 않음

#### 5.2.3 화살표 간 각도 = 변수 간 상관관계

| 각도 | 의미 | 예시 |
|------|------|------|
| **0° (같은 방향)** | 강한 양의 상관 (+1) | Murder ↔ Assault |
| **90° (직각)** | 상관 없음 (0) | Assault ↔ UrbanPop |
| **180° (반대 방향)** | 강한 음의 상관 (-1) | - |

### 5.3 점과 화살표의 관계

#### 투영 (Projection) 해석법

관측치(점)를 변수 화살표 위에 **수직으로 투영**하면 해당 변수의 대략적인 값을 알 수 있습니다.

```
        Murder 화살표
        ←——————————————————→
              ↑
        North |  (투영점: 왼쪽 → 낮은 Murder)
        Dakota|
              |
              |        Florida
              |          ↓
              |    (투영점: 오른쪽 → 높은 Murder)
```

#### 해석 예시

| 주(State) | 위치 | 해석 |
|-----------|------|------|
| Florida, Nevada | Murder, Rape 화살표 방향 | 범죄율 높음 |
| California, New York | UrbanPop 화살표 방향 | 도시화율 높음 |
| North Dakota, Vermont | 화살표 반대 방향 | 범죄율 낮음 |
| Mississippi | PC2 음의 방향 | 도시화율 낮음 (농촌) |

### 5.4 USArrests Biplot 종합 해석

```
                         높은 도시화율
                              ↑
                              | California
                  UrbanPop ↗  |  New York
                              |
    낮은 범죄율 ←——————————————+——————————————→ 높은 범죄율
    (안전한 주)               |                (위험한 주)
    Vermont                   |               Florida
    North Dakota    Murder ↙  |  ↘ Assault   Nevada
                      Rape ↙  |
                              |
                              ↓
                         낮은 도시화율
                         (농촌 지역)
                         Mississippi
```

**핵심 발견:**
1. Murder, Assault, Rape는 서로 강한 양의 상관관계 (비슷한 방향)
2. UrbanPop은 범죄 변수들과 거의 직각 → 도시화와 범죄율은 독립적인 별개 차원
3. PC1과 PC2가 전체 분산의 약 87% 설명 → 2차원으로 충분히 표현 가능

---

## 6. 주성분 수 결정 방법

### 6.1 Scree Plot (스크리 도표)

고유값을 순서대로 그린 그래프에서 **Elbow Point(팔꿈치 점)**를 찾습니다.

```r
# R
screeplot(pca_result, type = "lines")
```

```python
# Python
plt.plot(range(1, len(pca.explained_variance_) + 1), 
         pca.explained_variance_, 'bo-')
plt.xlabel('Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.show()
```

### 6.2 Kaiser 규칙 (고유값 > 1)

- **표준화된 데이터**에서 고유값이 1보다 큰 주성분만 선택
- 고유값 1 = 원래 변수 하나의 분산
- 원래 변수보다 더 많은 정보를 담고 있는 PC만 유지

```r
# R - Kaiser 규칙 적용
pca_result$sdev^2  # 고유값
# [1] 2.4802416 0.9897652 0.3565632 0.1734301
# → PC1만 1보다 큼 (엄격한 기준)
```

### 6.3 누적 설명 분산 비율

일반적으로 **70~90%** 설명하는 주성분 수를 선택합니다.

```r
# R
summary(pca_result)$importance[3,]  # Cumulative Proportion
# PC1: 62.0% → PC2: 86.7% → PC3: 95.7% → PC4: 100%
```

| 기준 | 선택되는 PC 수 |
|------|----------------|
| 70% 이상 | PC1 + PC2 = 2개 |
| 80% 이상 | PC1 + PC2 = 2개 |
| 90% 이상 | PC1 + PC2 + PC3 = 3개 |
| 95% 이상 | PC1 + PC2 + PC3 = 3개 |

### 6.4 Parallel Analysis (병렬 분석)

랜덤 데이터에서 얻은 고유값과 비교하여 유의미한 주성분을 선택합니다.

```r
# R - paran 패키지 사용
library(paran)
paran(USArrests, iterations = 1000)
```

### 6.5 주성분 수 결정 요약

| 방법 | 기준 | USArrests 결과 |
|------|------|----------------|
| Scree Plot | Elbow point | 2개 |
| Kaiser | 고유값 > 1 | 1개 |
| 누적분산 80% | ≥ 80% | 2개 |
| 누적분산 90% | ≥ 90% | 3개 |

**USArrests의 경우**: PC1, PC2 두 개가 적절 (86.7% 설명)

---

## 7. PCA 활용 분야

### 7.1 차원 축소 후 머신러닝

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# PCA로 차원 축소 후 분류
pipeline = Pipeline([
    ('pca', PCA(n_components=0.95)),  # 95% 분산 유지
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

### 7.2 다중공선성 해결

회귀 분석에서 독립변수 간 높은 상관관계가 있을 때:

```python
# 원래 변수 → PCA 변환 → 회귀 분석
from sklearn.linear_model import LinearRegression

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

model = LinearRegression()
model.fit(X_pca, y)
```

### 7.3 이상치 탐지

```python
# 주성분 공간에서 중심으로부터의 거리로 이상치 탐지
from scipy.spatial.distance import mahalanobis

# PC 스코어의 공분산 행렬
cov_matrix = np.cov(pca_scores.T)
mean = np.mean(pca_scores, axis=0)

# 마할라노비스 거리 계산
distances = [mahalanobis(x, mean, np.linalg.inv(cov_matrix)) 
             for x in pca_scores]
```

### 7.4 이미지 압축/노이즈 제거

```python
from sklearn.datasets import load_digits

# 손글씨 숫자 데이터
digits = load_digits()
X = digits.data

# PCA로 차원 축소
pca = PCA(n_components=30)  # 64 → 30
X_reduced = pca.fit_transform(X)

# 복원
X_reconstructed = pca.inverse_transform(X_reduced)
```

### 7.5 데이터 시각화

```python
# t-SNE, UMAP 전처리로 PCA 사용
from sklearn.manifold import TSNE

# 먼저 PCA로 50차원으로 축소 (t-SNE 속도 향상)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# 그 다음 t-SNE로 2차원 시각화
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_pca)
```

---

## 8. 빅데이터분석기사 핵심 포인트

### 8.1 자주 출제되는 개념

#### ✅ 반드시 알아야 할 것

| 개념 | 핵심 내용 |
|------|-----------|
| PCA 목적 | 차원 축소, 분산 최대화 |
| 주성분 특징 | 서로 직교(독립), 첫 PC가 최대 분산 |
| 고유값 의미 | 해당 주성분이 설명하는 분산 |
| 로딩 의미 | 원래 변수가 PC에 기여하는 가중치 |
| 스코어 의미 | 변환된 데이터 좌표 |
| 표준화 필요성 | 변수 스케일이 다르면 반드시 표준화 |

#### ✅ 주성분 수 결정

| 방법 | 기준 |
|------|------|
| Kaiser 규칙 | 고유값 > 1 |
| Scree Plot | Elbow point |
| 누적분산비율 | 70~90% |

#### ✅ Biplot 해석

| 요소 | 의미 |
|------|------|
| 화살표 방향 | 변수가 PC에 기여하는 방향 |
| 화살표 길이 | 변수의 설명력 |
| 화살표 간 각도 | 변수 간 상관관계 |
| 점의 위치 | 관측치의 PC 좌표 |

### 8.2 R 함수 정리

```r
# 기본 PCA
prcomp(data, scale = TRUE, center = TRUE)

# 결과 접근
pca$sdev       # 표준편차 (제곱 = 고유값)
pca$rotation   # 로딩 (고유벡터)
pca$x          # 스코어 (변환된 데이터)
pca$center     # 중심화에 사용된 평균
pca$scale      # 스케일링에 사용된 표준편차

# 요약
summary(pca)   # 설명분산비율, 누적분산비율

# 시각화
biplot(pca)
screeplot(pca)
```

### 8.3 Python 함수 정리

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=k)  # k: 주성분 수 또는 분산비율
pca.fit(X_scaled)

# 결과 접근
pca.components_           # 로딩 (고유벡터) - shape: (n_components, n_features)
pca.explained_variance_   # 고유값
pca.explained_variance_ratio_  # 설명분산비율
pca.n_components_         # 선택된 주성분 수

# 변환
pca.transform(X_scaled)        # 스코어 계산
pca.fit_transform(X_scaled)    # fit + transform
pca.inverse_transform(scores)  # 원래 공간으로 복원
```

### 8.4 예상 문제 유형

#### 유형 1: 개념 문제
> PCA에 대한 설명으로 옳지 않은 것은?
> 1. 비지도 학습 기법이다
> 2. 첫 번째 주성분이 가장 큰 분산을 설명한다
> 3. 주성분들은 서로 상관관계가 있다 ❌
> 4. 차원 축소에 활용된다

#### 유형 2: 결과 해석
> PCA 분석 결과 PC1의 분산비율이 60%, PC2가 25%일 때, 두 주성분의 누적분산비율은?
> → **85%**

#### 유형 3: 코드 해석
```r
pca <- prcomp(data, scale = TRUE)
summary(pca)
```
> scale = TRUE의 의미는?
> → **데이터를 표준화하여 분석**

#### 유형 4: Biplot 해석
> Biplot에서 두 변수의 화살표가 거의 같은 방향을 가리킬 때 의미는?
> → **두 변수 간 강한 양의 상관관계**

---

## 참고 자료

1. Jolliffe, I.T. (2002). Principal Component Analysis. Springer.
2. James, G. et al. (2013). An Introduction to Statistical Learning.
3. scikit-learn PCA Documentation: https://scikit-learn.org/stable/modules/decomposition.html#pca
4. R prcomp Documentation: https://www.rdocumentation.org/packages/stats/functions/prcomp
