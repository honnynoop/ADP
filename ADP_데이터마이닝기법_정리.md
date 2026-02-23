# ADP 데이터마이닝 기법 종합 정리

## 목차
1. [분류(Classification) 기법](#1-분류classification-기법)
2. [회귀(Regression) 기법](#2-회귀regression-기법)
3. [군집화(Clustering) 기법](#3-군집화clustering-기법)
4. [연관규칙(Association Rules) 기법](#4-연관규칙association-rules-기법)
5. [차원축소(Dimensionality Reduction) 기법](#5-차원축소dimensionality-reduction-기법)
6. [이상탐지(Anomaly Detection) 기법](#6-이상탐지anomaly-detection-기법)
7. [앙상블(Ensemble) 기법](#7-앙상블ensemble-기법)
8. [시계열 분석 기법](#8-시계열-분석-기법)
9. [텍스트 마이닝 기법](#9-텍스트-마이닝-기법)
10. [기법 비교 및 선택 가이드](#10-기법-비교-및-선택-가이드)

---

## 1. 분류(Classification) 기법

### 1.1 의사결정나무 (Decision Tree)

#### 개념 및 원리
- **정의**: 트리 구조를 이용하여 데이터를 분류하는 지도학습 기법
- **작동원리**: 데이터를 순도가 높아지는 방향으로 반복적으로 분할
- **분할 기준**: 정보이득(Information Gain), 지니지수(Gini Index), 카이제곱 통계량

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **알고리즘 종류** | - CART (Classification and Regression Trees)<br>- C4.5 / C5.0<br>- ID3<br>- CHAID |
| **분할 기준** | **정보이득**: Gain = Entropy(부모) - Σ(가중평균 Entropy(자식))<br>**지니지수**: Gini = 1 - Σ(p_i)²<br>**엔트로피**: Entropy = -Σ(p_i × log₂(p_i)) |
| **장점** | ✓ 해석이 매우 용이 (화이트박스 모델)<br>✓ 비선형 관계 포착 가능<br>✓ 결측값 처리 자동<br>✓ 변수 스케일링 불필요<br>✓ 범주형/연속형 변수 모두 처리<br>✓ 변수 중요도 파악 용이 |
| **단점** | ✗ 과적합 경향 높음<br>✗ 데이터 변화에 불안정<br>✗ 편향된 트리 생성 가능<br>✗ XOR 같은 문제 처리 어려움<br>✗ 연속형 변수 경계 결정 부정확 |
| **적용 시나리오** | - 신용평가, 대출 승인/거절<br>- 의료 진단 (해석 중요)<br>- 마케팅 캠페인 타겟팅<br>- 고객 이탈 예측<br>- 규칙 기반 시스템 구축 |
| **주요 하이퍼파라미터** | - `max_depth`: 트리 최대 깊이 (과적합 방지)<br>- `min_samples_split`: 분할 최소 샘플 수<br>- `min_samples_leaf`: 리프노드 최소 샘플 수<br>- `max_features`: 분할 시 고려할 최대 특징 수<br>- `criterion`: 분할 기준 (gini, entropy) |
| **주의사항** | ⚠ 가지치기(Pruning) 필수<br>⚠ 깊이 제한 설정<br>⚠ 앙상블과 함께 사용 권장<br>⚠ 불균형 데이터 시 클래스 가중치 조정<br>⚠ 범주형 변수 범주 수 많으면 편향 발생 |
| **평가지표** | - 정확도(Accuracy)<br>- 정밀도(Precision), 재현율(Recall)<br>- F1-Score<br>- AUC-ROC<br>- 혼동행렬(Confusion Matrix) |

#### Python 코드 예시

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습
dt_model = DecisionTreeClassifier(
    criterion='gini',           # 분할 기준
    max_depth=5,                # 최대 깊이
    min_samples_split=20,       # 분할 최소 샘플
    min_samples_leaf=10,        # 리프 최소 샘플
    random_state=42
)
dt_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = dt_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 트리 시각화
plt.figure(figsize=(20,10))
tree.plot_tree(dt_model, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()

# 변수 중요도
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)
```

---

### 1.2 로지스틱 회귀 (Logistic Regression)

#### 개념 및 원리
- **정의**: 선형 회귀에 시그모이드 함수를 적용하여 확률을 예측하는 분류 기법
- **수식**: P(Y=1|X) = 1 / (1 + e^(-z)), z = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
- **목적함수**: 로그손실(Log Loss) 최소화

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **모델 유형** | - 이진 분류(Binary)<br>- 다중 분류(Multinomial): One-vs-Rest, Softmax<br>- 순서형 로지스틱 회귀(Ordinal) |
| **주요 수식** | **시그모이드**: σ(z) = 1/(1+e^(-z))<br>**로그손실**: L = -Σ[y·log(p) + (1-y)·log(1-p)]<br>**오즈비(Odds Ratio)**: OR = e^β |
| **장점** | ✓ 확률값 직접 출력<br>✓ 계수 해석 용이 (오즈비)<br>✓ 계산 효율적<br>✓ 선형 관계 가정 시 안정적<br>✓ 정규화를 통한 과적합 방지<br>✓ 온라인 학습 가능 |
| **단점** | ✗ 선형 결정 경계만 가능<br>✗ 비선형 관계 포착 어려움<br>✗ 다중공선성에 민감<br>✗ 이상값에 영향 받음<br>✗ 클래스 불균형 시 성능 저하 |
| **적용 시나리오** | - 질병 발생 여부 예측<br>- 스팸 메일 분류<br>- 고객 구매 여부 예측<br>- 클릭률(CTR) 예측<br>- 신용 위험 평가 |
| **주요 하이퍼파라미터** | - `penalty`: 정규화 유형 (l1, l2, elasticnet, none)<br>- `C`: 정규화 강도의 역수 (작을수록 강한 정규화)<br>- `solver`: 최적화 알고리즘 (lbfgs, liblinear, saga)<br>- `max_iter`: 최대 반복 횟수<br>- `class_weight`: 클래스 가중치 (balanced) |
| **주의사항** | ⚠ 변수 간 독립성 확인 (다중공선성 제거)<br>⚠ 연속형 변수 스케일링 필수<br>⚠ 범주형 변수 인코딩 (One-Hot)<br>⚠ 클래스 불균형 시 SMOTE 등 적용<br>⚠ 정규화 파라미터 튜닝 |
| **평가지표** | - AUC-ROC, AUC-PR<br>- Log Loss<br>- Brier Score<br>- Calibration Curve |

#### Python 코드 예시

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

# 스케일링 (중요!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 생성 및 학습
lr_model = LogisticRegression(
    penalty='l2',               # L2 정규화
    C=1.0,                      # 정규화 강도
    solver='lbfgs',             # 최적화 알고리즘
    max_iter=1000,              # 최대 반복
    class_weight='balanced',    # 불균형 데이터 처리
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

# 확률 예측
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# AUC-ROC 평가
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {auc_score:.4f}")

# 계수 해석 (오즈비)
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr_model.coef_[0],
    'odds_ratio': np.exp(lr_model.coef_[0])
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df)
```

---

### 1.3 나이브 베이즈 (Naive Bayes)

#### 개념 및 원리
- **정의**: 베이즈 정리와 특징 독립 가정을 이용한 확률적 분류 기법
- **베이즈 정리**: P(C|X) = P(X|C) × P(C) / P(X)
- **독립 가정**: 모든 특징이 서로 독립적

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **알고리즘 종류** | - **Gaussian NB**: 연속형 변수, 정규분포 가정<br>- **Multinomial NB**: 텍스트 분류, 카운트 데이터<br>- **Bernoulli NB**: 이진 데이터<br>- **Complement NB**: 불균형 데이터셋 |
| **주요 수식** | **사후확률**: P(C|X) ∝ P(C) × Π P(x_i|C)<br>**가우시안**: P(x|C) = (1/√(2πσ²)) × e^(-(x-μ)²/2σ²)<br>**다항분포**: P(x|C) = (Σx_i)! / Π(x_i!) × Π p_i^x_i |
| **장점** | ✓ 학습 및 예측 속도 매우 빠름<br>✓ 고차원 데이터에 효과적<br>✓ 작은 데이터셋에서도 잘 작동<br>✓ 확률값 직접 제공<br>✓ 다중 클래스 분류에 자연스럽게 확장<br>✓ 실시간 예측에 적합 |
| **단점** | ✗ 독립 가정이 현실적이지 않음<br>✗ 상관관계 있는 변수 처리 약함<br>✗ 연속형 변수의 분포 가정 필요<br>✗ Zero Probability 문제 |
| **적용 시나리오** | - 텍스트 분류 (스팸 필터링)<br>- 문서 범주화<br>- 감성 분석<br>- 추천 시스템<br>- 실시간 예측 시스템<br>- 의료 진단 (사전확률 활용) |
| **주요 하이퍼파라미터** | - `alpha` (Laplace Smoothing): 평활화 파라미터 (기본값 1.0)<br>- `fit_prior`: 사전확률 학습 여부<br>- `class_prior`: 사전확률 지정<br>- `var_smoothing` (Gaussian NB): 분산 평활화 |
| **주의사항** | ⚠ 변수 독립성 검토<br>⚠ Zero Probability 방지 (Laplace Smoothing)<br>⚠ 적절한 NB 유형 선택 (데이터 특성에 맞게)<br>⚠ 연속형 변수 분포 확인<br>⚠ 불균형 데이터 시 사전확률 조정 |
| **평가지표** | - 정확도, F1-Score<br>- 로그 손실<br>- 확률 보정 성능 |

#### Python 코드 예시

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 연속형 데이터 - Gaussian NB
gnb_model = GaussianNB(var_smoothing=1e-9)
gnb_model.fit(X_train, y_train)
y_pred = gnb_model.predict(X_test)

# 텍스트 데이터 - Multinomial NB
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(text_train)
X_test_counts = vectorizer.transform(text_test)

mnb_model = MultinomialNB(alpha=1.0)  # Laplace smoothing
mnb_model.fit(X_train_counts, y_train)
y_pred = mnb_model.predict(X_test_counts)

# 확률 예측
y_pred_proba = mnb_model.predict_proba(X_test_counts)
```

---

### 1.4 k-최근접 이웃 (k-Nearest Neighbors, KNN)

#### 개념 및 원리
- **정의**: 가장 가까운 k개의 이웃을 찾아 다수결로 분류하는 게으른 학습 기법
- **거리 측정**: 유클리디안, 맨해튼, 민코프스키 거리
- **특징**: Instance-based learning, Lazy learning

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **거리 측정 방법** | **유클리디안**: d = √(Σ(x_i - y_i)²)<br>**맨해튼**: d = Σ\|x_i - y_i\|<br>**민코프스키**: d = (Σ\|x_i - y_i\|^p)^(1/p)<br>**코사인 유사도**: cos(θ) = (A·B) / (\|\|A\|\| × \|\|B\|\|) |
| **장점** | ✓ 알고리즘 단순하고 직관적<br>✓ 비선형 데이터 분류 가능<br>✓ 이론적 배경 불필요<br>✓ 다중 클래스 분류 자연스럽게 처리<br>✓ 추가 학습 데이터 반영 쉬움 |
| **단점** | ✗ 예측 시간 느림 (모든 데이터 비교)<br>✗ 메모리 사용량 많음<br>✗ 차원의 저주에 취약<br>✗ 불균형 데이터 처리 어려움<br>✗ 최적 k 값 찾기 어려움<br>✗ 이상값에 민감 |
| **적용 시나리오** | - 추천 시스템 (유사 사용자/상품)<br>- 패턴 인식<br>- 이미지 분류<br>- 이상 탐지<br>- 손글씨 인식<br>- 소규모 데이터셋 |
| **주요 하이퍼파라미터** | - `n_neighbors`: 이웃 수 k (일반적으로 √n)<br>- `weights`: 가중치 ('uniform', 'distance')<br>- `metric`: 거리 측정 방법<br>- `algorithm`: 탐색 알고리즘 ('auto', 'ball_tree', 'kd_tree', 'brute')<br>- `p`: 민코프스키 거리의 p 값 |
| **주의사항** | ⚠ 반드시 데이터 정규화/스케일링<br>⚠ k 값은 홀수 권장 (이진 분류 시)<br>⚠ k 값 교차검증으로 최적화<br>⚠ 차원축소 선행 (PCA 등)<br>⚠ 대용량 데이터엔 부적합<br>⚠ 거리 가중치 옵션 고려 |
| **평가지표** | - 정확도<br>- 혼동행렬<br>- k 값에 따른 성능 곡선 |

#### Python 코드 예시

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# 스케일링 필수!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 최적 k 찾기
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
print(f"최적 k: {optimal_k}")

# 모델 학습
knn_model = KNeighborsClassifier(
    n_neighbors=optimal_k,
    weights='distance',      # 거리 가중치
    metric='euclidean',      # 유클리디안 거리
    algorithm='auto'         # 자동 선택
)
knn_model.fit(X_train_scaled, y_train)

# 예측
y_pred = knn_model.predict(X_test_scaled)
```

---

### 1.5 서포트 벡터 머신 (Support Vector Machine, SVM)

#### 개념 및 원리
- **정의**: 최대 마진을 갖는 초평면(hyperplane)을 찾아 데이터를 분류
- **목표**: 클래스 간 마진을 최대화하는 결정 경계 찾기
- **커널 트릭**: 비선형 데이터를 고차원 공간으로 매핑하여 선형 분리 가능하게 변환

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **커널 함수** | **Linear**: K(x, y) = x^T · y<br>**Polynomial**: K(x, y) = (γx^T·y + r)^d<br>**RBF (Gaussian)**: K(x, y) = e^(-γ\|\|x-y\|\|²)<br>**Sigmoid**: K(x, y) = tanh(γx^T·y + r) |
| **주요 개념** | - **Support Vector**: 결정 경계에 가장 가까운 데이터 포인트<br>- **Margin**: 결정 경계와 Support Vector 간 거리<br>- **Soft Margin**: 일부 오분류 허용 (C 파라미터)<br>- **Kernel Trick**: 비선형 변환 |
| **장점** | ✓ 고차원 데이터에 효과적<br>✓ 메모리 효율적 (support vector만 사용)<br>✓ 과적합 제어 가능 (C, γ 파라미터)<br>✓ 다양한 커널 함수 활용<br>✓ 이상값에 상대적으로 강건<br>✓ 이론적 배경 탄탄 |
| **단점** | ✗ 대용량 데이터셋에서 느림 (O(n²~n³))<br>✗ 확률값 직접 제공 안 함<br>✗ 하이퍼파라미터 튜닝 까다로움<br>✗ 해석이 어려움 (블랙박스)<br>✗ 다중 클래스 분류 시 복잡<br>✗ 불균형 데이터 처리 어려움 |
| **적용 시나리오** | - 텍스트 분류 (고차원)<br>- 이미지 인식<br>- 생명정보학 (유전자 분류)<br>- 필기체 인식<br>- 얼굴 감지<br>- 중소규모 데이터셋 |
| **주요 하이퍼파라미터** | - `C`: 정규화 파라미터 (작을수록 강한 정규화, 넓은 마진)<br>- `kernel`: 커널 유형 ('linear', 'rbf', 'poly', 'sigmoid')<br>- `gamma`: RBF 커널 파라미터 (작을수록 넓은 영향 범위)<br>- `degree`: 다항식 커널의 차수<br>- `class_weight`: 클래스 가중치 |
| **주의사항** | ⚠ 반드시 스케일링 수행<br>⚠ C와 gamma 그리드 서치로 최적화<br>⚠ 대용량 데이터는 선형 SVM 또는 SGDClassifier 사용<br>⚠ 커널 선택 중요 (선형 → RBF 순으로 시도)<br>⚠ 클래스 불균형 시 가중치 조정<br>⚠ 확률 예측 필요 시 `probability=True` 설정 |
| **평가지표** | - 정확도, F1-Score<br>- Support Vector 개수<br>- 결정 함수 값 분포 |

#### Python 코드 예시

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# 스케일링 필수!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 그리드 서치로 하이퍼파라미터 최적화
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

svm_model = SVC(random_state=42, probability=True)
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 예측
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
y_pred_proba = best_svm.predict_proba(X_test_scaled)

# Support Vector 확인
print(f"Support Vectors 개수: {best_svm.n_support_}")
```

---

### 1.6 신경망 (Neural Network)

#### 개념 및 원리
- **정의**: 인간의 뇌 구조를 모방한 다층 퍼셉트론(MLP) 기반 학습 모델
- **구조**: 입력층 - 은닉층(여러 개 가능) - 출력층
- **학습**: 역전파(Backpropagation) 알고리즘을 통한 가중치 업데이트

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **주요 구성요소** | - **뉴런(Neuron)**: 입력 신호 처리 단위<br>- **가중치(Weight)**: 연결 강도<br>- **편향(Bias)**: 활성화 임계값 조정<br>- **활성화 함수**: 비선형성 도입 |
| **활성화 함수** | **Sigmoid**: σ(x) = 1/(1+e^(-x))<br>**Tanh**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))<br>**ReLU**: f(x) = max(0, x)<br>**Leaky ReLU**: f(x) = max(0.01x, x)<br>**Softmax**: σ(x_i) = e^(x_i) / Σe^(x_j) |
| **장점** | ✓ 복잡한 비선형 관계 학습<br>✓ 범용적 근사기 (Universal Approximator)<br>✓ 다양한 문제에 적용 가능<br>✓ 대용량 데이터 처리 가능<br>✓ 피처 엔지니어링 자동화<br>✓ 전이학습 가능 |
| **단점** | ✗ 학습 시간 오래 걸림<br>✗ 하이퍼파라미터 많음<br>✗ 해석 어려움 (블랙박스)<br>✗ 과적합 위험<br>✗ 지역 최소값 문제<br>✗ 많은 데이터 필요 |
| **적용 시나리오** | - 이미지 인식 (CNN)<br>- 자연어 처리 (RNN, Transformer)<br>- 음성 인식<br>- 시계열 예측<br>- 복잡한 패턴 인식<br>- 추천 시스템 |
| **주요 하이퍼파라미터** | - `hidden_layer_sizes`: 은닉층 구조 (예: (100, 50))<br>- `activation`: 활성화 함수 ('relu', 'tanh', 'logistic')<br>- `solver`: 최적화 알고리즘 ('adam', 'sgd', 'lbfgs')<br>- `alpha`: L2 정규화 파라미터<br>- `learning_rate_init`: 초기 학습률<br>- `batch_size`: 배치 크기<br>- `max_iter`: 최대 반복 횟수 |
| **주의사항** | ⚠ 데이터 정규화/스케일링 필수<br>⚠ 적절한 층 수와 뉴런 수 선택<br>⚠ 과적합 방지 (Dropout, Early Stopping)<br>⚠ 학습률 조정 중요<br>⚠ 배치 정규화 고려<br>⚠ 충분한 데이터 확보<br>⚠ GPU 활용 권장 |
| **정규화 기법** | - Dropout: 일부 뉴런 무작위 비활성화<br>- L1/L2 Regularization<br>- Batch Normalization<br>- Early Stopping |

#### Python 코드 예시

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 생성 및 학습
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),  # 3개 은닉층
    activation='relu',                  # ReLU 활성화
    solver='adam',                      # Adam 옵티마이저
    alpha=0.0001,                       # L2 정규화
    batch_size='auto',                  # 배치 크기
    learning_rate='adaptive',           # 적응형 학습률
    learning_rate_init=0.001,           # 초기 학습률
    max_iter=200,                       # 최대 에포크
    early_stopping=True,                # 조기 종료
    validation_fraction=0.1,            # 검증 데이터 비율
    n_iter_no_change=10,                # 조기 종료 patience
    random_state=42,
    verbose=True
)

mlp_model.fit(X_train_scaled, y_train)

# 학습 곡선 시각화
plt.plot(mlp_model.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# 예측
y_pred = mlp_model.predict(X_test_scaled)
```

---

## 2. 회귀(Regression) 기법

### 2.1 선형 회귀 (Linear Regression)

#### 개념 및 원리
- **정의**: 독립변수와 종속변수 간의 선형 관계를 모델링
- **수식**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
- **목적함수**: 잔차제곱합(RSS) 최소화

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **회귀 유형** | - **단순 선형 회귀**: 독립변수 1개<br>- **다중 선형 회귀**: 독립변수 2개 이상<br>- **다항 회귀**: 독립변수의 거듭제곱 포함<br>- **정규화 회귀**: Ridge, Lasso, ElasticNet |
| **주요 가정** | 1. **선형성**: X와 Y는 선형 관계<br>2. **독립성**: 오차항이 서로 독립<br>3. **등분산성**: 오차의 분산이 일정<br>4. **정규성**: 오차항이 정규분포<br>5. **다중공선성 없음**: 독립변수 간 상관관계 낮음 |
| **손실함수** | **MSE**: (1/n)Σ(y_i - ŷ_i)²<br>**RMSE**: √MSE<br>**MAE**: (1/n)Σ\|y_i - ŷ_i\|<br>**R²**: 1 - (SSR/SST) |
| **장점** | ✓ 해석 용이 (계수의 의미 명확)<br>✓ 계산 효율적 (최소제곱법)<br>✓ 통계적 유의성 검정 가능<br>✓ 신뢰구간/예측구간 제공<br>✓ 기본적이면서도 강력<br>✓ 외삽(extrapolation) 가능 |
| **단점** | ✗ 비선형 관계 포착 불가<br>✗ 이상값에 민감<br>✗ 다중공선성 문제<br>✗ 가정 충족 필요<br>✗ 과적합 위험 (변수 많을 때) |
| **적용 시나리오** | - 주택 가격 예측<br>- 매출 예측<br>- 수요 예측<br>- 인과관계 분석<br>- 추세 분석<br>- 벤치마크 모델 |
| **주요 파라미터** | - `fit_intercept`: 절편 포함 여부<br>- `normalize`: 정규화 여부 (deprecated)<br>- `copy_X`: X 복사 여부 |
| **주의사항** | ⚠ 가정 검증 필수 (선형성, 등분산성, 정규성)<br>⚠ 다중공선성 확인 (VIF)<br>⚠ 이상값 처리<br>⚠ 변수 선택/제거<br>⚠ 스케일링 (정규화 회귀 시)<br>⚠ 잔차 분석 수행 |
| **평가지표** | - **R² (결정계수)**: 설명력<br>- **Adjusted R²**: 변수 수 고려<br>- **RMSE**: 예측 오차<br>- **MAE**: 평균 절대 오차<br>- **MAPE**: 평균 절대 백분율 오차 |

#### Python 코드 예시

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats

# 모델 학습
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 예측
y_pred = lr_model.predict(X_test)

# 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# 계수 해석
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(f"절편: {lr_model.intercept_:.4f}")
print(coef_df)

# 잔차 분석
residuals = y_test - y_pred

# 정규성 검정 (Shapiro-Wilk)
stat, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk p-value: {p_value:.4f}")

# 잔차 플롯
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.subplot(1, 3, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.show()
```

---

### 2.2 정규화 회귀 (Regularized Regression)

#### Ridge, Lasso, ElasticNet 비교

| 특성 | Ridge (L2) | Lasso (L1) | ElasticNet |
|------|-----------|-----------|------------|
| **정규화 항** | λΣβ²_j | λΣ\|β_j\| | λ₁Σβ²_j + λ₂Σ\|β_j\| |
| **계수 특징** | 0에 가깝게 축소 | 정확히 0으로 만듦 | 두 방식 결합 |
| **변수 선택** | 불가능 | 가능 (자동) | 가능 |
| **다중공선성** | 잘 처리함 | 하나만 선택 경향 | 균형잡힌 처리 |
| **장점** | - 안정적<br>- 모든 변수 유지<br>- 다중공선성 해결 | - 변수 선택 자동<br>- 해석 용이<br>- 희소 모델 | - 두 장점 결합<br>- 유연함<br>- 상관 변수 그룹 선택 |
| **단점** | - 변수 선택 불가<br>- 해석 복잡 | - 불안정할 수 있음<br>- 그룹 변수 중 하나만 선택 | - 파라미터 2개<br>- 계산 복잡 |
| **사용 시기** | 변수 많고<br>다중공선성 있을 때 | 변수 선택 필요하고<br>희소 모델 원할 때 | 두 장점 필요하고<br>상관 변수 많을 때 |
| **주요 파라미터** | `alpha`: 정규화 강도 | `alpha`: 정규화 강도 | `alpha`: 전체 정규화 강도<br>`l1_ratio`: L1 비율 (0~1) |

#### Python 코드 예시

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 스케일링 (정규화 회귀에 필수!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge 회귀
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Lasso 회귀 - 하이퍼파라미터 튜닝
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
lasso = Lasso(max_iter=10000)
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
lasso_model = grid_search.best_estimator_

# ElasticNet
elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
elasticnet_model.fit(X_train_scaled, y_train)

# 계수 비교
coef_comparison = pd.DataFrame({
    'feature': feature_names,
    'Ridge': ridge_model.coef_,
    'Lasso': lasso_model.coef_,
    'ElasticNet': elasticnet_model.coef_
})

# Lasso로 선택된 변수 확인
selected_features = coef_comparison[coef_comparison['Lasso'] != 0]['feature'].tolist()
print(f"Lasso가 선택한 변수 수: {len(selected_features)}/{len(feature_names)}")
```

---

### 2.3 회귀 트리 및 앙상블

| 기법 | 설명 | 장점 | 단점 | 사용 시기 |
|------|------|------|------|-----------|
| **Decision Tree Regressor** | 트리 구조로 연속값 예측 | 비선형 관계 포착<br>해석 용이 | 과적합 경향<br>불안정 | 해석 중요,<br>비선형 관계 |
| **Random Forest** | 다수의 회귀 트리 앙상블 | 과적합 방지<br>안정적<br>변수 중요도 | 해석 어려움<br>느린 예측 | 대부분의 경우<br>벤치마크 |
| **Gradient Boosting** | 순차적 오차 보정 | 높은 정확도<br>유연함 | 학습 느림<br>하이퍼파라미터 많음 | 성능 최우선 |
| **XGBoost** | 최적화된 그래디언트 부스팅 | 매우 빠름<br>정규화 내장<br>결측값 자동 처리 | 메모리 사용량<br>해석 어려움 | 구조적 데이터<br>고성능 필요 |

---

## 3. 군집화(Clustering) 기법

### 3.1 K-평균 군집화 (K-Means Clustering)

#### 개념 및 원리
- **정의**: 데이터를 k개의 군집으로 분할하는 비지도 학습 기법
- **목표**: 군집 내 분산(Within-Cluster Sum of Squares, WCSS) 최소화
- **알고리즘**: 중심점 초기화 → 할당 → 중심점 업데이트 → 수렴까지 반복

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **알고리즘 단계** | 1. k개 중심점 무작위 초기화<br>2. 각 데이터를 가장 가까운 중심점에 할당<br>3. 각 군집의 평균으로 중심점 업데이트<br>4. 중심점 변화 없을 때까지 2-3 반복 |
| **목적함수** | **WCSS**: Σᵏᵢ₌₁ Σₓ∈Cᵢ \|\|x - μᵢ\|\|²<br>(각 군집 내 데이터와 중심점 간 거리 제곱합) |
| **장점** | ✓ 알고리즘 단순하고 빠름 (O(nkt))<br>✓ 대용량 데이터 처리 가능<br>✓ 구현 쉬움<br>✓ 확장성 좋음 |
| **단점** | ✗ k 값 사전 지정 필요<br>✗ 초기값에 민감<br>✗ 구형 군집만 잘 찾음<br>✗ 이상값에 민감<br>✗ 크기가 다른 군집 처리 어려움<br>✗ 지역 최적해 문제 |
| **적용 시나리오** | - 고객 세분화<br>- 이미지 압축<br>- 문서 클러스터링<br>- 이상 탐지 (군집에서 멀리 떨어진 점)<br>- 데이터 요약 |
| **주요 하이퍼파라미터** | - `n_clusters`: 군집 수 k<br>- `init`: 초기화 방법 ('k-means++', 'random')<br>- `n_init`: 다른 초기값으로 실행 횟수<br>- `max_iter`: 최대 반복 횟수<br>- `random_state`: 난수 시드 |
| **최적 k 선택** | - **Elbow Method**: WCSS 그래프에서 꺾이는 지점<br>- **Silhouette Score**: 군집 응집도와 분리도<br>- **Gap Statistic**: 실제 vs 참조 데이터 비교<br>- **도메인 지식**: 비즈니스 요구사항 |
| **주의사항** | ⚠ 반드시 스케일링 수행<br>⚠ k-means++ 초기화 사용 권장<br>⚠ n_init을 10 이상으로 설정<br>⚠ 최적 k 값 다양한 방법으로 검증<br>⚠ 결과 해석 시 도메인 지식 활용<br>⚠ 이상값 제거 고려 |
| **평가지표** | - **Silhouette Score**: -1 ~ 1 (높을수록 좋음)<br>- **Davies-Bouldin Index**: 낮을수록 좋음<br>- **Calinski-Harabasz Index**: 높을수록 좋음<br>- **Inertia (WCSS)**: 낮을수록 좋음 |

#### Python 코드 예시

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# 스케일링 필수!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method로 최적 k 찾기
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Elbow 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.show()

# 최적 k로 모델 학습
optimal_k = 4  # Elbow와 Silhouette 결과 기반
kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
clusters = kmeans_model.fit_predict(X_scaled)

# 평가
silhouette_avg = silhouette_score(X_scaled, clusters)
db_index = davies_bouldin_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")

# 군집별 특성 분석
cluster_analysis = pd.DataFrame(X)
cluster_analysis['Cluster'] = clusters
print(cluster_analysis.groupby('Cluster').mean())
```

---

### 3.2 계층적 군집화 (Hierarchical Clustering)

#### 개념 및 원리
- **정의**: 덴드로그램 형태의 트리 구조로 군집을 형성
- **유형**: 응집형(Agglomerative, 상향식), 분할형(Divisive, 하향식)
- **결과**: 덴드로그램을 통해 다양한 수준의 군집 확인 가능

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **연결 방법** | **Single Linkage**: 가장 가까운 점 간 거리<br>**Complete Linkage**: 가장 먼 점 간 거리<br>**Average Linkage**: 평균 거리<br>**Ward**: 분산 증가 최소화 (가장 많이 사용) |
| **장점** | ✓ k 값 사전 지정 불필요<br>✓ 덴드로그램으로 시각화 가능<br>✓ 계층 구조 파악 용이<br>✓ 다양한 모양의 군집 탐지<br>✓ 결정론적 (초기값 영향 없음) |
| **단점** | ✗ 계산 복잡도 높음 (O(n²) ~ O(n³))<br>✗ 대용량 데이터 부적합<br>✗ 한 번 병합하면 되돌릴 수 없음<br>✗ 이상값에 민감<br>✗ 메모리 사용량 많음 |
| **적용 시나리오** | - 생물학적 분류 (계통수)<br>- 소셜 네트워크 분석<br>- 텍스트 문서 계층화<br>- 중소규모 데이터셋<br>- 군집 계층 구조 파악 필요 시 |
| **주요 하이퍼파라미터** | - `n_clusters`: 군집 수 (덴드로그램 절단 위치)<br>- `linkage`: 연결 방법 ('ward', 'complete', 'average', 'single')<br>- `metric`: 거리 측정 방법<br>- `distance_threshold`: 거리 임계값 |
| **주의사항** | ⚠ 데이터 스케일링 필수<br>⚠ Ward는 유클리디안 거리만 가능<br>⚠ 대용량 데이터는 샘플링 후 적용<br>⚠ 덴드로그램 해석 주의<br>⚠ 연결 방법에 따라 결과 크게 달라짐 |

#### Python 코드 예시

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 덴드로그램 생성
linkage_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# 모델 학습
hc_model = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward',
    metric='euclidean'
)
clusters = hc_model.fit_predict(X_scaled)

# 평가
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")
```

---

### 3.3 DBSCAN (Density-Based Spatial Clustering)

#### 개념 및 원리
- **정의**: 밀도 기반 군집화로 임의 모양의 군집을 찾고 노이즈 탐지
- **핵심 개념**: ε(epsilon) 반경 내 최소 점 수(min_samples)
- **점 유형**: 핵심점(Core), 경계점(Border), 노이즈점(Noise)

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **주요 개념** | - **ε (epsilon)**: 이웃을 정의하는 반경<br>- **MinPts**: 핵심점이 되기 위한 최소 이웃 수<br>- **밀도 도달 가능**: 핵심점에서 ε 이내<br>- **밀도 연결**: 공통 핵심점을 통해 연결 |
| **장점** | ✓ k 값 지정 불필요<br>✓ 임의 모양 군집 탐지<br>✓ 노이즈/이상값 자동 탐지<br>✓ 군집 크기 제약 없음<br>✓ 밀도가 다양한 데이터 처리 가능 |
| **단점** | ✗ ε와 MinPts 설정 어려움<br>✗ 밀도가 크게 다른 군집 처리 어려움<br>✗ 고차원 데이터에서 성능 저하<br>✗ 경계 점 할당 모호할 수 있음 |
| **적용 시나리오** | - 지리 데이터 분석<br>- 이상 탐지<br>- 불규칙한 모양의 군집<br>- 노이즈가 많은 데이터<br>- 실시간 데이터 스트림 |
| **주요 하이퍼파라미터** | - `eps`: 이웃 반경 ε<br>- `min_samples`: 핵심점 최소 이웃 수<br>- `metric`: 거리 측정 방법<br>- `algorithm`: 탐색 알고리즘 ('auto', 'ball_tree', 'kd_tree', 'brute') |
| **파라미터 선택** | - **eps**: k-거리 그래프의 elbow 지점<br>- **min_samples**: 일반적으로 차원 수 × 2 또는 4<br>- **NearestNeighbors** 활용한 최적값 탐색 |
| **주의사항** | ⚠ 데이터 스케일링 필수<br>⚠ eps와 min_samples 신중히 선택<br>⚠ 고차원 데이터는 차원 축소 선행<br>⚠ 군집 레이블 -1은 노이즈<br>⚠ 대용량 데이터는 샘플링 고려 |

#### Python 코드 예시

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# eps 값 찾기 위한 k-거리 그래프
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:, -1], axis=0)
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel('5th Nearest Neighbor Distance')
plt.title('k-distance Graph')
plt.axhline(y=0.3, color='r', linestyle='--', label='eps candidate')
plt.legend()
plt.show()

# DBSCAN 모델
dbscan_model = DBSCAN(
    eps=0.3,           # k-거리 그래프에서 결정
    min_samples=5,     # 일반적으로 차원 수 × 2
    metric='euclidean'
)
clusters = dbscan_model.fit_predict(X_scaled)

# 결과 분석
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f"군집 수: {n_clusters}")
print(f"노이즈 점 수: {n_noise}")

# 노이즈가 아닌 점들에 대한 실루엣 점수
if n_clusters > 1:
    mask = clusters != -1
    if mask.sum() > 0:
        silhouette_avg = silhouette_score(X_scaled[mask], clusters[mask])
        print(f"Silhouette Score: {silhouette_avg:.4f}")
```

---

### 3.4 군집화 기법 비교

| 기법 | 장점 | 단점 | 적합한 상황 | 복잡도 |
|------|------|------|-------------|--------|
| **K-Means** | 빠르고 간단<br>대용량 처리 | k 지정 필요<br>구형 군집만 | 대용량 데이터<br>빠른 처리 필요 | O(nkt) |
| **Hierarchical** | k 불필요<br>계층 구조 | 느림<br>메모리 많이 사용 | 소규모 데이터<br>계층 구조 중요 | O(n²) ~ O(n³) |
| **DBSCAN** | 임의 모양<br>노이즈 탐지 | 파라미터 설정 어려움<br>밀도 차이 처리 약함 | 노이즈 많음<br>불규칙 모양 | O(n log n) |
| **GMM** | 확률 기반<br>소프트 군집화 | EM 알고리즘 느림<br>초기값 민감 | 확률 필요<br>겹치는 군집 | O(nkt) |
| **Mean Shift** | 파라미터 적음<br>임의 모양 | 매우 느림<br>대역폭 선택 | 소규모<br>자동화 | O(n²) |

---

## 4. 연관규칙(Association Rules) 기법

### 4.1 Apriori 알고리즘

#### 개념 및 원리
- **정의**: 장바구니 분석을 위한 빈발 항목집합(Frequent Itemset) 발견 알고리즘
- **원리**: Apriori 속성 - 빈발 항목집합의 부분집합은 모두 빈발
- **단계**: 빈발 1-항목집합 → 2-항목집합 → ... → k-항목집합

#### 주요 측도

| 측도 | 수식 | 의미 | 해석 |
|------|------|------|------|
| **지지도 (Support)** | P(A ∩ B) = count(A,B) / N | A와 B가 함께 나타나는 비율 | 규칙의 유용성<br>(낮으면 우연일 수 있음) |
| **신뢰도 (Confidence)** | P(B\|A) = P(A∩B) / P(A) | A 구매 시 B 구매 확률 | 규칙의 정확도<br>(A→B의 신뢰성) |
| **향상도 (Lift)** | P(B\|A) / P(B) = Confidence / P(B) | A와 B의 독립성 대비 증가 정도 | Lift > 1: 양의 상관<br>Lift = 1: 독립<br>Lift < 1: 음의 상관 |
| **확신도 (Conviction)** | [1-P(B)] / [1-P(B\|A)] | 규칙이 틀릴 확률의 비 | 인과관계 강도 |
| **레버리지 (Leverage)** | P(A∩B) - P(A)×P(B) | 독립 가정 대비 차이 | 실제 효과 크기 |

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **알고리즘 단계** | 1. 최소 지지도 이상인 1-항목집합 찾기<br>2. k-항목집합에서 (k+1)-항목집합 생성<br>3. 최소 지지도 미달 항목 제거<br>4. 더 이상 빈발 항목집합이 없을 때까지 반복<br>5. 연관규칙 생성 및 신뢰도/향상도 계산 |
| **장점** | ✓ 원리 단순하고 이해 쉬움<br>✓ 후보 항목집합 효율적 생성<br>✓ 해석 용이<br>✓ 다양한 분야 적용 가능 |
| **단점** | ✗ 대용량 데이터에서 느림<br>✗ 메모리 사용량 많음<br>✗ 다중 스캔 필요<br>✗ 희소 데이터에서 비효율적 |
| **적용 시나리오** | - 장바구니 분석 (상품 추천)<br>- 교차판매/상향판매<br>- 웹 사용 패턴 분석<br>- 의료 진단 (증상-질병)<br>- 클릭스트림 분석 |
| **주요 하이퍼파라미터** | - `min_support`: 최소 지지도 (0.01 ~ 0.1 일반적)<br>- `min_confidence`: 최소 신뢰도 (0.5 이상 권장)<br>- `min_lift`: 최소 향상도 (1.0 이상)<br>- `max_length`: 최대 항목 수 |
| **주의사항** | ⚠ 최소 지지도 너무 낮으면 조합 폭발<br>⚠ 최소 지지도 너무 높으면 규칙 발견 못함<br>⚠ 향상도 1 이상인 규칙에 집중<br>⚠ 인과관계와 상관관계 구분<br>⚠ 도메인 지식으로 규칙 검증<br>⚠ 너무 많은 규칙 생성 시 필터링 필요 |
| **평가 기준** | - 지지도: 실용성<br>- 신뢰도: 정확성<br>- 향상도: 유의미성<br>- 비즈니스 가치 |

#### Python 코드 예시

```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# 트랜잭션 데이터를 원-핫 인코딩 형식으로 변환
# transactions: [['우유', '빵'], ['우유', '기저귀', '맥주'], ...]
# -> DataFrame with binary values

# One-hot encoding
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori 알고리즘으로 빈발 항목집합 찾기
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
print(f"빈발 항목집합 수: {len(frequent_itemsets)}")
print(frequent_itemsets.sort_values('support', ascending=False).head(10))

# 연관규칙 생성
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# 신뢰도 필터링
rules = rules[rules['confidence'] >= 0.5]

# 결과 정렬 및 출력
rules_sorted = rules.sort_values(['lift', 'confidence'], ascending=False)
print(f"\n생성된 규칙 수: {len(rules_sorted)}")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# 규칙 해석 예시
for idx, row in rules_sorted.head(5).iterrows():
    antecedent = ', '.join(list(row['antecedents']))
    consequent = ', '.join(list(row['consequents']))
    print(f"\n규칙: {antecedent} → {consequent}")
    print(f"  지지도: {row['support']:.4f} (전체의 {row['support']*100:.2f}%)")
    print(f"  신뢰도: {row['confidence']:.4f} ({antecedent} 구매 시 {row['confidence']*100:.2f}% 확률로 {consequent} 구매)")
    print(f"  향상도: {row['lift']:.4f} (무작위 대비 {row['lift']:.2f}배)")
```

---

### 4.2 FP-Growth 알고리즘

#### 개념 및 원리
- **정의**: FP-Tree 자료구조를 사용하여 Apriori보다 효율적으로 빈발 항목집합 발견
- **장점**: 데이터베이스 2회만 스캔, 후보 생성 없음
- **구조**: FP-Tree (Frequent Pattern Tree) + Header Table

#### Apriori vs FP-Growth 비교

| 특성 | Apriori | FP-Growth |
|------|---------|-----------|
| **스캔 횟수** | k+1회 (k는 최대 항목집합 크기) | 2회 |
| **후보 생성** | 필요 (많은 메모리) | 불필요 |
| **속도** | 느림 | 빠름 (10~100배) |
| **메모리** | 후보 저장 시 많음 | FP-Tree 구조 저장 |
| **적합 데이터** | 중소규모, 밀집 데이터 | 대용량, 희소 데이터 |
| **구현 복잡도** | 단순 | 복잡 |

#### Python 코드 예시

```python
from mlxtend.frequent_patterns import fpgrowth, association_rules

# FP-Growth 알고리즘
frequent_itemsets_fpg = fpgrowth(df, min_support=0.01, use_colnames=True)

# 연관규칙 생성 (Apriori와 동일한 방식)
rules_fpg = association_rules(frequent_itemsets_fpg, metric="lift", min_threshold=1.0)
rules_fpg = rules_fpg[rules_fpg['confidence'] >= 0.5]

# 성능 비교
print(f"FP-Growth 빈발 항목집합 수: {len(frequent_itemsets_fpg)}")
print(f"FP-Growth 규칙 수: {len(rules_fpg)}")
```

---

## 5. 차원축소(Dimensionality Reduction) 기법

### 5.1 주성분 분석 (PCA, Principal Component Analysis)

#### 개념 및 원리
- **정의**: 데이터의 분산을 최대화하는 축을 찾아 차원을 축소하는 선형 변환 기법
- **목표**: 원본 데이터의 정보를 최대한 보존하면서 차원 축소
- **방법**: 공분산 행렬의 고유벡터(eigenvector)를 주성분으로 사용

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **수학적 원리** | 1. 데이터 중심화 (평균 0)<br>2. 공분산 행렬 계산<br>3. 고유값/고유벡터 계산<br>4. 고유값 큰 순으로 정렬<br>5. 상위 k개 고유벡터로 변환 |
| **주성분 의미** | - **PC1**: 가장 큰 분산 방향<br>- **PC2**: PC1에 직교하며 두 번째로 큰 분산 방향<br>- **PC3, ...**: 순차적으로 직교하며 분산 큰 방향 |
| **장점** | ✓ 차원의 저주 완화<br>✓ 노이즈 제거<br>✓ 시각화 가능 (2D/3D)<br>✓ 다중공선성 제거<br>✓ 계산 효율성 향상<br>✓ 과적합 방지 |
| **단점** | ✗ 해석 어려움 (원본 변수 조합)<br>✗ 선형 관계만 포착<br>✗ 이상값에 민감<br>✗ 스케일링 필수<br>✗ 정보 손실 가능 |
| **적용 시나리오** | - 고차원 데이터 시각화<br>- 전처리 단계 (차원 축소)<br>- 이미지 압축<br>- 얼굴 인식 (Eigenface)<br>- 다중공선성 제거<br>- 노이즈 제거 |
| **주요 하이퍼파라미터** | - `n_components`: 주성분 개수 (또는 설명 분산 비율)<br>- `whiten`: 백색화 여부 (분산 1로 스케일링)<br>- `svd_solver`: SVD 알고리즘 ('auto', 'full', 'arpack', 'randomized') |
| **성분 수 선택** | - **Elbow Method**: Scree plot에서 꺾이는 지점<br>- **설명 분산**: 누적 85-95% 보존<br>- **Kaiser 기준**: 고유값 > 1<br>- **교차검증**: 모델 성능 기반 |
| **주의사항** | ⚠ 반드시 표준화/정규화<br>⚠ 이상값 제거 선행<br>⚠ 설명 분산 비율 확인<br>⚠ 원본 데이터 보관 (역변환 위해)<br>⚠ 범주형 변수 인코딩 후 적용<br>⚠ 로딩(loading) 해석 시 주의 |
| **평가 방법** | - 설명 분산 비율 (Explained Variance Ratio)<br>- Scree Plot<br>- 누적 설명 분산<br>- 재구성 오차 |

#### Python 코드 예시

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 스케일링 필수!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 수행 (모든 성분)
pca_full = PCA()
pca_full.fit(X_scaled)

# Scree Plot - 성분 수 결정
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca_full.explained_variance_) + 1), 
         pca_full.explained_variance_, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')

# 누적 설명 분산
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.tight_layout()
plt.show()

# 95% 분산 설명하는 성분 수 찾기
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
print(f"95% 분산 설명하는 성분 수: {n_components_95}/{X.shape[1]}")

# 최종 PCA 적용
pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)

print(f"원본 차원: {X.shape[1]}")
print(f"축소 차원: {X_pca.shape[1]}")
print(f"설명 분산 비율: {pca.explained_variance_ratio_}")
print(f"총 설명 분산: {pca.explained_variance_ratio_.sum():.4f}")

# 주성분 로딩 (원본 변수와의 상관관계)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_df = pd.DataFrame(
    loadings[:, :3],  # 상위 3개 성분
    columns=['PC1', 'PC2', 'PC3'],
    index=feature_names
)
print("\n주요 변수 로딩 (상위 3개 성분):")
print(loading_df.abs().sort_values('PC1', ascending=False).head(5))

# 2D 시각화 (분류 문제인 경우)
if 'y' in locals():
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA - First Two Components')
    plt.colorbar(scatter)
    plt.show()
```

---

### 5.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

#### 개념 및 원리
- **정의**: 고차원 데이터의 국소적 구조를 보존하며 저차원(주로 2D/3D)으로 시각화하는 비선형 기법
- **목표**: 고차원에서 가까운 점들은 저차원에서도 가깝게, 먼 점들은 멀게 유지
- **특징**: 시각화에 특화, 군집 구조 잘 드러냄

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **작동 원리** | 1. 고차원에서 점 간 유사도를 가우시안 분포로 계산<br>2. 저차원에서 유사도를 t-분포로 계산<br>3. KL divergence 최소화하며 점 배치 최적화 |
| **PCA vs t-SNE** | **PCA**: 전역 구조 보존, 선형, 빠름, 재현 가능<br>**t-SNE**: 국소 구조 보존, 비선형, 느림, 확률적 |
| **장점** | ✓ 복잡한 비선형 구조 시각화<br>✓ 군집 분리 매우 잘됨<br>✓ 고차원 데이터 패턴 발견<br>✓ 시각적으로 직관적 |
| **단점** | ✗ 계산 매우 느림 (O(n²))<br>✗ 새로운 데이터 변환 불가<br>✗ 하이퍼파라미터에 민감<br>✗ 전역 구조 왜곡 가능<br>✗ 거리 해석 주의 필요<br>✗ 결과 매번 달라짐 (확률적) |
| **적용 시나리오** | - 고차원 데이터 시각화<br>- 군집 구조 탐색<br>- 이미지 데이터 시각화<br>- 단어 임베딩 시각화<br>- 탐색적 데이터 분석 (EDA) |
| **주요 하이퍼파라미터** | - `n_components`: 목표 차원 (보통 2 또는 3)<br>- `perplexity`: 이웃 수 고려 (5~50, 기본 30)<br>- `learning_rate`: 학습률 (10~1000)<br>- `n_iter`: 반복 횟수 (최소 250, 권장 1000)<br>- `random_state`: 재현성 위한 시드 |
| **Perplexity 의미** | - 각 점의 유효 이웃 수<br>- 작은 값: 국소 구조 강조<br>- 큰 값: 전역 구조 강조<br>- 데이터 크기에 따라 조정 (보통 5~50) |
| **주의사항** | ⚠ 대용량 데이터는 PCA 선행 (50~100차원)<br>⚠ Perplexity 여러 값 시도<br>⚠ 충분한 반복 횟수 (1000+)<br>⚠ 거리 정량적 해석 금지<br>⚠ 여러 번 실행하여 일관성 확인<br>⚠ 시각화 목적으로만 사용<br>⚠ 모델 학습에는 부적합 |
| **사용 팁** | - 샘플 수 < 10,000 권장<br>- 그 이상은 샘플링 또는 다른 방법<br>- Perplexity는 데이터 수의 1~5%<br>- 여러 perplexity 값 비교 |

#### Python 코드 예시

```python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# 대용량 데이터는 샘플링
if X.shape[0] > 10000:
    idx = np.random.choice(X.shape[0], 10000, replace=False)
    X_sample = X[idx]
    y_sample = y[idx] if 'y' in locals() else None
else:
    X_sample = X
    y_sample = y if 'y' in locals() else None

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# 고차원 데이터는 PCA 선행 (50차원으로)
if X_scaled.shape[1] > 50:
    pca = PCA(n_components=50)
    X_scaled = pca.fit_transform(X_scaled)
    print(f"PCA로 {X.shape[1]}차원 → 50차원 축소")

# 여러 perplexity 값으로 실험
perplexities = [5, 30, 50]
fig, axes = plt.subplots(1, len(perplexities), figsize=(18, 5))

for idx, perplexity in enumerate(perplexities):
    print(f"\nPerplexity = {perplexity} 실행 중...")
    start_time = time.time()
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        n_iter=1000,
        random_state=42,
        verbose=0
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    elapsed = time.time() - start_time
    print(f"완료 ({elapsed:.2f}초)")
    
    # 시각화
    ax = axes[idx]
    if y_sample is not None:
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                           cmap='viridis', alpha=0.6, s=5)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=5)
    
    ax.set_title(f't-SNE (perplexity={perplexity})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

# 최적 perplexity 선택 후 최종 시각화
tsne_final = TSNE(n_components=2, perplexity=30, learning_rate=200, 
                  n_iter=1000, random_state=42)
X_tsne_final = tsne_final.fit_transform(X_scaled)
```

---

### 5.3 UMAP (Uniform Manifold Approximation and Projection)

#### 개념 및 원리
- **정의**: t-SNE의 개선 버전, 더 빠르고 전역 구조도 보존
- **특징**: 수학적으로 더 엄밀, 새로운 데이터 변환 가능, t-SNE보다 빠름

#### t-SNE vs UMAP 비교

| 특성 | t-SNE | UMAP |
|------|-------|------|
| **속도** | 느림 (O(n²)) | 빠름 (O(n log n)) |
| **전역 구조** | 약함 | 강함 |
| **새 데이터** | 변환 불가 | 변환 가능 |
| **파라미터 민감도** | 높음 | 중간 |
| **용도** | 시각화 전용 | 시각화 + 전처리 |
| **군집 분리** | 매우 좋음 | 좋음 |

#### Python 코드 예시

```python
import umap

# UMAP 적용
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,      # t-SNE의 perplexity와 유사
    min_dist=0.1,        # 점 간 최소 거리
    metric='euclidean',
    random_state=42
)
X_umap = umap_model.fit_transform(X_scaled)

# 시각화
plt.figure(figsize=(8, 6))
if 'y' in locals():
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
else:
    plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP Projection')
plt.show()

# 새로운 데이터 변환 (UMAP의 장점)
X_new_umap = umap_model.transform(X_new_scaled)
```

---

### 5.4 차원축소 기법 비교

| 기법 | 유형 | 속도 | 전역 구조 | 국소 구조 | 용도 | 새 데이터 | 파라미터 |
|------|------|------|-----------|-----------|------|-----------|----------|
| **PCA** | 선형 | 매우 빠름 | 강함 | 약함 | 전처리, 시각화 | O | 적음 |
| **t-SNE** | 비선형 | 느림 | 약함 | 매우 강함 | 시각화 | X | 중간 |
| **UMAP** | 비선형 | 빠름 | 중간 | 강함 | 둘 다 | O | 중간 |
| **LDA** | 선형 | 빠름 | 강함 | 약함 | 분류 전처리 | O | 적음 |
| **Autoencoder** | 비선형 | 중간 | 중간 | 중간 | 둘 다 | O | 많음 |

---

## 6. 이상탐지(Anomaly Detection) 기법

### 6.1 Isolation Forest

#### 개념 및 원리
- **정의**: 이상값을 격리(isolate)하는 데 필요한 분할 횟수가 적다는 원리 활용
- **가정**: 이상값은 적고(few), 다르다(different)
- **방법**: 랜덤 포레스트와 유사하나 격리 트리(isolation tree) 사용

#### 상세 정보

| 항목 | 내용 |
|------|------|
| **작동 원리** | 1. 특징과 분할값 무작위 선택<br>2. 재귀적으로 데이터 분할<br>3. 격리까지 필요한 경로 길이 측정<br>4. 여러 트리 평균으로 이상 점수 계산<br>5. 점수 높은 점을 이상값으로 판단 |
| **이상 점수** | **경로 길이**: 루트에서 리프까지 깊이<br>**이상값**: 짧은 경로 (빨리 격리됨)<br>**정상값**: 긴 경로 (격리 어려움) |
| **장점** | ✓ 빠른 학습 및 예측<br>✓ 메모리 효율적<br>✓ 고차원 데이터 처리 가능<br>✓ 레이블 불필요 (비지도)<br>✓ 파라미터 적음<br>✓ 여러 이상값 유형 탐지 |
| **단점** | ✗ 해석 어려움<br>✗ 경계 이상값 놓칠 수 있음<br>✗ 정상 데이터 비율 가정 필요<br>✗ 노이즈 많으면 성능 저하 |
| **적용 시나리오** | - 금융 사기 탐지<br>- 네트워크 침입 탐지<br>- 제조 결함 탐지<br>- 의료 이상 진단<br>- 시스템 장애 탐지 |
| **주요 하이퍼파라미터** | - `n_estimators`: 트리 개수 (100~500)<br>- `max_samples`: 샘플 수 (256 권장)<br>- `contamination`: 이상값 비율 (0.1 = 10%)<br>- `max_features`: 사용 특징 수<br>- `random_state`: 재현성 |
| **주의사항** | ⚠ contamination 값 신중히 설정<br>⚠ 도메인 지식으로 임계값 조정<br>⚠ 스케일링 권장 (필수는 아님)<br>⚠ 이상값 레이블 있으면 검증<br>⚠ 시각화로 결과 확인 |
| **평가지표** | - Precision, Recall, F1-Score (레이블 있을 때)<br>- ROC-AUC<br>- 수동 검증 |

#### Python 코드 예시

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 스케일링 (권장)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest 모델
iso_forest = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.1,  # 10%를 이상값으로 가정
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)

# 이상 점수 및 예측
anomaly_scores = iso_forest.score_samples(X_scaled)  # 낮을수록 이상
predictions = iso_forest.predict(X_scaled)  # 1: 정상, -1: 이상

# 이상값 인덱스
anomalies = np.where(predictions == -1)[0]
print(f"탐지된 이상값 수: {len(anomalies)} ({len(anomalies)/len(X)*100:.2f}%)")

# 이상 점수 분포 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(anomaly_scores, bins=50, edgecolor='black')
plt.axvline(x=iso_forest.offset_, color='r', linestyle='--', label='Threshold')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Distribution of Anomaly Scores')
plt.legend()

# 2D 시각화 (PCA 사용)
if X.shape[1] > 2:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
else:
    X_2d = X_scaled

plt.subplot(1, 2, 2)
plt.scatter(X_2d[predictions == 1, 0], X_2d[predictions == 1, 1], 
           c='blue', label='Normal', alpha=0.5, s=20)
plt.scatter(X_2d[predictions == -1, 0], X_2d[predictions == -1, 1], 
           c='red', label='Anomaly', alpha=0.8, s=50, marker='x')
plt.xlabel('First Component')
plt.ylabel('Second Component')
plt.title('Anomaly Detection Results')
plt.legend()
plt.tight_layout()
plt.show()

# 이상값 분석
if anomalies.shape[0] > 0:
    anomaly_df = pd.DataFrame(X[anomalies], columns=feature_names)
    print("\n이상값 샘플 (상위 5개):")
    print(anomaly_df.head())
    
    print("\n이상값 통계:")
    print(anomaly_df.describe())
```

---

### 6.2 One-Class SVM

#### 개념 및 원리
- **정의**: 정상 데이터만 학습하여 결정 경계를 만들고, 경계 밖을 이상값으로 판단
- **목표**: 정상 데이터를 포함하는 최소 부피의 초구 찾기
- **특징**: 커널을 사용하여 비선형 경계 학습 가능

| 항목 | 내용 |
|------|------|
| **장점** | ✓ 비선형 경계 학습<br>✓ 커널 트릭 활용<br>✓ 이론적 배경 탄탄 |
| **단점** | ✗ 대용량 데이터에 느림<br>✗ 파라미터 튜닝 어려움<br>✗ 메모리 사용량 많음 |
| **주요 파라미터** | - `nu`: 이상값 비율 상한 (0.1 = 최대 10%)<br>- `kernel`: 'rbf', 'linear', 'poly'<br>- `gamma`: RBF 커널 파라미터 |

#### Python 코드 예시

```python
from sklearn.svm import OneClassSVM

# One-Class SVM
oc_svm = OneClassSVM(
    kernel='rbf',
    gamma='auto',
    nu=0.1  # 최대 10% 이상값 허용
)
oc_svm.fit(X_scaled)

# 예측 (-1: 이상, 1: 정상)
predictions = oc_svm.predict(X_scaled)
anomalies = np.where(predictions == -1)[0]
print(f"탐지된 이상값 수: {len(anomalies)}")
```

---

### 6.3 LOF (Local Outlier Factor)

#### 개념 및 원리
- **정의**: 데이터 포인트의 국소 밀도를 주변 이웃과 비교하여 이상값 판단
- **핵심**: 이웃보다 밀도가 현저히 낮으면 이상값
- **LOF 점수**: > 1이면 이상값, ≈1이면 정상

| 항목 | 내용 |
|------|------|
| **장점** | ✓ 국소적 이상값 탐지<br>✓ 밀도 기반<br>✓ 다양한 밀도 처리 |
| **단점** | ✗ 파라미터에 민감<br>✗ 고차원에서 성능 저하<br>✗ 계산 비용 높음 |
| **주요 파라미터** | - `n_neighbors`: 이웃 수 (10~50)<br>- `contamination`: 이상값 비율<br>- `metric`: 거리 측정 방법 |

#### Python 코드 예시

```python
from sklearn.neighbors import LocalOutlierFactor

# LOF
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=False  # 학습 데이터에 대한 예측
)
predictions = lof.fit_predict(X_scaled)

# 이상 점수 (음수가 이상값)
lof_scores = lof.negative_outlier_factor_
anomalies = np.where(predictions == -1)[0]
print(f"탐지된 이상값 수: {len(anomalies)}")
```

---

### 6.4 이상탐지 기법 비교

| 기법 | 유형 | 속도 | 고차원 | 국소/전역 | 레이블 필요 | 새 데이터 | 적합 상황 |
|------|------|------|--------|-----------|------------|-----------|-----------|
| **Isolation Forest** | 앙상블 | 빠름 | 좋음 | 전역 | 불필요 | O | 대부분의 경우 |
| **One-Class SVM** | 커널 | 느림 | 중간 | 전역 | 불필요 | O | 비선형 경계 |
| **LOF** | 밀도 기반 | 중간 | 약함 | 국소 | 불필요 | X | 밀도 변화 큼 |
| **Autoencoder** | 신경망 | 중간 | 좋음 | 전역 | 불필요 | O | 복잡한 패턴 |
| **Statistical** | 통계 | 매우 빠름 | 약함 | 전역 | 불필요 | O | 단순한 경우 |

---

## 7. 앙상블(Ensemble) 기법

### 7.1 배깅 (Bagging) - Random Forest

#### 개념 및 원리
- **정의**: Bootstrap Aggregating, 부트스트랩 샘플링 + 집계
- **방법**: 여러 모델을 독립적으로 학습하여 투표/평균
- **대표**: Random Forest

#### Random Forest 상세

| 항목 | 내용 |
|------|------|
| **작동 원리** | 1. 부트스트랩 샘플링 (중복 허용 샘플링)<br>2. 각 샘플로 의사결정나무 학습<br>3. 각 노드에서 무작위 특징 부분집합만 고려<br>4. 분류: 다수결, 회귀: 평균 |
| **장점** | ✓ 과적합 방지<br>✓ 높은 정확도<br>✓ 변수 중요도 제공<br>✓ 결측값 처리<br>✓ 안정적<br>✓ 병렬 처리 가능<br>✓ 이상값에 강건 |
| **단점** | ✗ 해석 어려움<br>✗ 예측 느림<br>✗ 메모리 많이 사용<br>✗ 외삽 약함 |
| **주요 파라미터** | - `n_estimators`: 트리 개수 (100~500)<br>- `max_depth`: 트리 최대 깊이<br>- `max_features`: 분할 시 고려 특징 수 ('sqrt', 'log2')<br>- `min_samples_split/leaf`: 분할/리프 최소 샘플<br>- `bootstrap`: 부트스트랩 사용 여부<br>- `oob_score`: OOB 점수 계산 여부 |
| **특징 중요도** | - MDI (Mean Decrease in Impurity)<br>- Permutation Importance<br>- SHAP values |

#### Python 코드 예시

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# OOB Score (학습 데이터로 검증)
print(f"OOB Score: {rf_model.oob_score_:.4f}")

# 예측
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# 변수 중요도
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"최적 파라미터: {grid_search.best_params_}")
```

---

### 7.2 부스팅 (Boosting)

#### 7.2.1 AdaBoost (Adaptive Boosting)

| 항목 | 내용 |
|------|------|
| **원리** | 순차적으로 약한 학습기를 학습하며 이전 오류에 가중치 부여 |
| **가중치 업데이트** | 오분류된 샘플에 높은 가중치, 정분류 샘플에 낮은 가중치 |
| **장점** | 간단하고 효과적, 파라미터 적음 |
| **단점** | 노이즈와 이상값에 민감, 과적합 위험 |
| **주요 파라미터** | `n_estimators`, `learning_rate`, `base_estimator` |

#### 7.2.2 Gradient Boosting

| 항목 | 내용 |
|------|------|
| **원리** | 이전 모델의 잔차(residual)를 학습하여 순차적으로 모델 개선 |
| **목적함수** | 손실 함수의 그래디언트 방향으로 최적화 |
| **장점** | 매우 높은 정확도, 유연한 손실 함수 |
| **단점** | 학습 느림, 하이퍼파라미터 많음, 과적합 위험 |
| **주요 파라미터** | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

#### 7.2.3 XGBoost (Extreme Gradient Boosting)

| 항목 | 내용 |
|------|------|
| **특징** | Gradient Boosting의 최적화 버전, 속도와 성능 개선 |
| **핵심 기술** | - 정규화 (L1, L2)<br>- 병렬 처리<br>- 트리 가지치기<br>- 결측값 자동 처리<br>- 조기 종료 |
| **장점** | ✓ 매우 빠름<br>✓ 높은 정확도<br>✓ 정규화 내장<br>✓ 결측값 처리<br>✓ 조기 종료<br>✓ 변수 중요도 |
| **단점** | ✗ 하이퍼파라미터 많음<br>✗ 해석 어려움<br>✗ 메모리 사용 |
| **주요 파라미터** | - `n_estimators`: 트리 개수<br>- `learning_rate` (eta): 학습률 (0.01~0.3)<br>- `max_depth`: 트리 깊이 (3~10)<br>- `subsample`: 샘플 비율 (0.8~1.0)<br>- `colsample_bytree`: 특징 비율<br>- `gamma`: 분할 최소 손실 감소<br>- `reg_alpha` (L1), `reg_lambda` (L2): 정규화 |

#### Python 코드 예시

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# 조기 종료를 위한 검증 세트
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=10,
    verbose=False
)

# 예측
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# 변수 중요도
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title('XGBoost Feature Importance')
plt.show()

# 학습 곡선
results = xgb_model.evals_result()
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(results['validation_0']['logloss'], label='Train')
plt.plot(results['validation_1']['logloss'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.show()
```

#### 7.2.4 LightGBM

| 항목 | 내용 |
|------|------|
| **특징** | Microsoft 개발, XGBoost보다 빠르고 메모리 효율적 |
| **핵심 기술** | - Leaf-wise 트리 성장<br>- Histogram 기반 알고리즘<br>- GOSS (Gradient-based One-Side Sampling)<br>- EFB (Exclusive Feature Bundling) |
| **장점** | XGBoost보다 빠름, 메모리 효율적, 범주형 변수 자동 처리 |
| **단점** | 작은 데이터셋에서 과적합, 파라미터 튜닝 중요 |

#### 7.2.5 CatBoost

| 항목 | 내용 |
|------|------|
| **특징** | Yandex 개발, 범주형 변수에 특화 |
| **핵심 기술** | - Ordered Boosting<br>- 범주형 변수 자동 인코딩<br>- 과적합 방지 메커니즘 |
| **장점** | 범주형 변수 전처리 불필요, 과적합 방지, 빠른 학습 |

---

### 7.3 스태킹 (Stacking)

#### 개념 및 원리
- **정의**: 여러 모델의 예측을 새로운 모델의 입력으로 사용
- **구조**: Base Learners (1층) → Meta Learner (2층)
- **방법**: 교차검증으로 base 모델 학습 → 예측값으로 meta 모델 학습

| 항목 | 내용 |
|------|------|
| **장점** | 다양한 모델의 강점 결합, 높은 성능 |
| **단점** | 계산 비용 높음, 과적합 위험, 복잡함 |
| **추천 조합** | Base: {RandomForest, XGBoost, SVM, Logistic}<br>Meta: Logistic Regression, Ridge |

#### Python 코드 예시

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)),
    ('svm', SVC(probability=True, random_state=42))
]

# Meta learner
meta_learner = LogisticRegression()

# Stacking
stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
```

---

### 7.4 앙상블 기법 비교

| 기법 | 학습 방식 | 다양성 확보 | 속도 | 정확도 | 과적합 | 해석성 |
|------|-----------|-------------|------|--------|--------|--------|
| **Bagging (RF)** | 병렬 | 샘플링 + 특징 | 빠름 | 높음 | 낮음 | 낮음 |
| **AdaBoost** | 순차 | 가중치 조정 | 중간 | 중간 | 중간 | 낮음 |
| **GBM** | 순차 | 잔차 학습 | 느림 | 매우 높음 | 높음 | 낮음 |
| **XGBoost** | 순차 | 잔차 + 정규화 | 빠름 | 매우 높음 | 중간 | 낮음 |
| **LightGBM** | 순차 | Leaf-wise | 매우 빠름 | 매우 높음 | 중간 | 낮음 |
| **Stacking** | 계층적 | 모델 다양성 | 느림 | 매우 높음 | 높음 | 매우 낮음 |

---

## 8. 시계열 분석 기법

### 8.1 ARIMA (AutoRegressive Integrated Moving Average)

#### 개념 및 구성요소

| 구성 | 기호 | 의미 | 설명 |
|------|------|------|------|
| **AR (자기회귀)** | p | 과거 값의 선형 조합 | Y_t = c + φ₁Y_(t-1) + ... + φ_pY_(t-p) + ε_t |
| **I (차분)** | d | 정상성 확보 위한 차분 횟수 | 비정상 → 정상 시계열 변환 |
| **MA (이동평균)** | q | 과거 오차의 선형 조합 | Y_t = μ + ε_t + θ₁ε_(t-1) + ... + θ_qε_(t-q) |

#### 모델 선택 및 진단

| 단계 | 방법 | 도구 |
|------|------|------|
| **정상성 검정** | ADF Test, KPSS Test | p-value < 0.05 → 정상 |
| **차분 차수 (d)** | ACF 플롯, 단위근 검정 | 일반적으로 d=0, 1, 2 |
| **AR 차수 (p)** | PACF 플롯 | PACF가 절단되는 지점 |
| **MA 차수 (q)** | ACF 플롯 | ACF가 절단되는 지점 |
| **모델 비교** | AIC, BIC | 낮을수록 좋음 |

#### Python 코드 예시

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# 정상성 검정 (ADF Test)
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    if result[1] <= 0.05:
        print("정상 시계열 (귀무가설 기각)")
    else:
        print("비정상 시계열 (차분 필요)")

adf_test(time_series)

# ACF, PACF 플롯
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(time_series, lags=40, ax=axes[0])
plot_pacf(time_series, lags=40, ax=axes[1])
plt.show()

# ARIMA 모델
model = ARIMA(time_series, order=(1, 1, 1))  # (p, d, q)
fitted_model = model.fit()
print(fitted_model.summary())

# 예측
forecast = fitted_model.forecast(steps=10)
print(forecast)

# 잔차 진단
residuals = fitted_model.resid
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
residuals.plot(ax=axes[0, 0], title='Residuals')
residuals.plot(kind='kde', ax=axes[0, 1], title='Density')
plot_acf(residuals, lags=40, ax=axes[1, 0])
axes[1, 1].scatter(fitted_model.fittedvalues, residuals)
axes[1, 1].set_xlabel('Fitted values')
axes[1, 1].set_ylabel('Residuals')
plt.tight_layout()
plt.show()
```

### 8.2 계절성 모델 (SARIMA, Prophet, LSTM)

| 모델 | 특징 | 장점 | 단점 | 사용 시기 |
|------|------|------|------|-----------|
| **SARIMA** | 계절성 ARIMA | 계절 패턴 포착<br>통계적 기반 | 파라미터 많음<br>복잡 | 명확한 계절성 |
| **Prophet** | Facebook 개발 | 사용 쉬움<br>결측값 처리<br>휴일 효과 | 긴 예측 부정확 | 일별/주별 데이터 |
| **LSTM** | 딥러닝 | 복잡한 패턴<br>비선형 관계 | 많은 데이터 필요<br>느림 | 대용량, 복잡 패턴 |

---

## 9. 텍스트 마이닝 기법

### 9.1 텍스트 전처리

| 단계 | 기법 | 설명 | Python |
|------|------|------|--------|
| **정규화** | 소문자 변환 | 대소문자 통일 | `text.lower()` |
| **토큰화** | 단어 분리 | 문장 → 단어 리스트 | `nltk.word_tokenize()` |
| **불용어 제거** | Stopwords | 의미 없는 단어 제거 | `nltk.corpus.stopwords` |
| **어간 추출** | Stemming | 어간만 추출 | `PorterStemmer()` |
| **표제어 추출** | Lemmatization | 원형으로 변환 | `WordNetLemmatizer()` |

### 9.2 텍스트 벡터화

| 기법 | 설명 | 장점 | 단점 | 사용 시기 |
|------|------|------|------|-----------|
| **Bag of Words** | 단어 빈도 카운트 | 단순, 빠름 | 순서 무시, 희소 | 기본 분류 |
| **TF-IDF** | 문서별 중요도 | 차별적 단어 강조 | 순서 무시 | 문서 분류/검색 |
| **Word2Vec** | 단어 임베딩 | 의미 포착, 밀집 | 학습 필요 | 유사도, 분류 |
| **Doc2Vec** | 문서 임베딩 | 문서 의미 표현 | 학습 필요 | 문서 유사도 |
| **BERT** | 문맥 임베딩 | 최고 성능 | 느림, 무거움 | 고성능 필요 시 |

### 9.3 주요 텍스트 마이닝 작업

| 작업 | 설명 | 주요 기법 |
|------|------|-----------|
| **감성 분석** | 긍정/부정 분류 | Naive Bayes, LSTM, BERT |
| **토픽 모델링** | 주제 추출 | LDA, NMF |
| **개체명 인식** | 고유명사 추출 | CRF, BiLSTM-CRF |
| **문서 분류** | 카테고리 분류 | SVM, CNN, BERT |
| **요약** | 핵심 내용 추출 | TextRank, Seq2Seq |

---

## 10. 기법 비교 및 선택 가이드

### 10.1 문제 유형별 추천 기법

| 문제 유형 | 1순위 | 2순위 | 3순위 | 고려사항 |
|-----------|-------|-------|-------|----------|
| **이진 분류** | XGBoost, LightGBM | Random Forest | Logistic Regression | 데이터 크기, 해석 필요성 |
| **다중 분류** | XGBoost | Random Forest | Neural Network | 클래스 수, 균형 |
| **회귀** | XGBoost | Random Forest | Ridge/Lasso | 선형성, 해석 필요성 |
| **군집화** | K-Means | DBSCAN | Hierarchical | 클러스터 모양, 크기 |
| **차원축소** | PCA | UMAP | t-SNE | 목적 (전처리 vs 시각화) |
| **이상탐지** | Isolation Forest | One-Class SVM | LOF | 데이터 크기, 이상값 유형 |
| **시계열** | Prophet | ARIMA | LSTM | 계절성, 데이터 길이 |
| **텍스트** | BERT | TF-IDF + SVM | Word2Vec + LSTM | 성능 vs 속도 |
| **추천** | Collaborative Filtering | Matrix Factorization | Neural CF | Cold start 여부 |

### 10.2 데이터 특성별 기법 선택

| 데이터 특성 | 적합 기법 | 부적합 기법 | 이유 |
|-------------|-----------|-------------|------|
| **소규모 (<1,000)** | Logistic, SVM, Naive Bayes | Deep Learning, XGBoost | 과적합 위험 |
| **대규모 (>100만)** | XGBoost, LightGBM, SGD | KNN, SVM | 계산 효율성 |
| **고차원 (>100 features)** | Random Forest, PCA | KNN, Naive Bayes | 차원의 저주 |
| **불균형 클래스** | SMOTE + RF, XGBoost(scale_pos_weight) | 기본 분류기 | 클래스 가중치 필요 |
| **결측값 많음** | XGBoost, Random Forest | SVM, KNN | 자동 처리 |
| **범주형 많음** | CatBoost, LightGBM | Linear 모델 | 인코딩 자동화 |
| **비선형 관계** | XGBoost, Neural Network, SVM(RBF) | Linear Regression, Logistic | 복잡한 패턴 |
| **해석 중요** | Decision Tree, Linear | XGBoost, Deep Learning | 설명 가능성 |

### 10.3 성능 vs 해석성 트레이드오프

```
해석성 높음 ←―――――――――――――――――――――――――――――→ 성능 높음

Linear Regression    Logistic Regression    Decision Tree    Random Forest    XGBoost    Neural Network
        │                     │                    │               │             │              │
    Ridge/Lasso         Naive Bayes          CART/C4.5      Bagging         GBM      Deep Learning
```

### 10.4 프로젝트 단계별 기법 활용

| 단계 | 목적 | 추천 기법 | 이유 |
|------|------|-----------|------|
| **탐색 (EDA)** | 패턴 발견 | PCA, t-SNE, 군집화 | 시각화, 구조 파악 |
| **베이스라인** | 빠른 검증 | Logistic, Decision Tree | 단순, 빠름, 해석 용이 |
| **성능 개선** | 정확도 향상 | XGBoost, Random Forest, Stacking | 높은 성능 |
| **운영 배포** | 안정성/속도 | LightGBM, Logistic(간단한 경우) | 빠른 추론, 안정적 |
| **설명/보고** | 인사이트 도출 | SHAP, Decision Tree, Linear | 해석 가능 |

---

## 11. 평가지표 종합

### 11.1 분류 평가지표

| 지표 | 수식 | 의미 | 사용 시기 |
|------|------|------|-----------|
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | 전체 정확도 | 균형 데이터 |
| **Precision** | TP / (TP+FP) | 양성 예측 정확도 | FP 비용 높을 때 |
| **Recall** | TP / (TP+FN) | 실제 양성 탐지율 | FN 비용 높을 때 |
| **F1-Score** | 2 × (P×R) / (P+R) | 정밀도-재현율 조화평균 | 불균형 데이터 |
| **AUC-ROC** | ROC 곡선 아래 면적 | 임계값 무관 성능 | 확률 예측 평가 |
| **AUC-PR** | PR 곡선 아래 면적 | 불균형 데이터 평가 | 극심한 불균형 |
| **Log Loss** | -Σ(y log(p) + (1-y)log(1-p)) | 확률 예측 오차 | 확률 보정 중요 |

### 11.2 회귀 평가지표

| 지표 | 수식 | 의미 | 특징 |
|------|------|------|------|
| **MAE** | (1/n)Σ\|y-ŷ\| | 평균 절대 오차 | 이상값에 강건 |
| **MSE** | (1/n)Σ(y-ŷ)² | 평균 제곱 오차 | 큰 오차에 민감 |
| **RMSE** | √MSE | MSE의 제곱근 | 원래 단위, 해석 용이 |
| **R²** | 1 - (SSR/SST) | 설명력 | 0~1, 높을수록 좋음 |
| **MAPE** | (1/n)Σ\|((y-ŷ)/y)\|×100 | 평균 절대 백분율 오차 | 비율로 해석 |

### 11.3 군집화 평가지표

| 지표 | 범위 | 의미 | 레이블 필요 |
|------|------|------|------------|
| **Silhouette** | -1 ~ 1 | 응집도 & 분리도 | X |
| **Davies-Bouldin** | 0 ~ ∞ | 낮을수록 좋음 | X |
| **Calinski-Harabasz** | 0 ~ ∞ | 높을수록 좋음 | X |
| **ARI** | -1 ~ 1 | 조정 랜드 지수 | O |
| **NMI** | 0 ~ 1 | 정규화 상호정보 | O |

---

## 12. 실전 체크리스트

### 12.1 기법 선택 체크리스트

- [ ] 문제 유형 확인 (분류/회귀/군집/추천 등)
- [ ] 데이터 크기 확인 (샘플 수, 피처 수)
- [ ] 데이터 품질 평가 (결측값, 이상값, 불균형)
- [ ] 성능 vs 해석성 요구사항 확인
- [ ] 계산 자원 확인 (시간, 메모리, GPU)
- [ ] 배포 환경 고려 (실시간/배치, 모델 크기)

### 12.2 모델 개발 체크리스트

- [ ] 데이터 전처리 (스케일링, 인코딩)
- [ ] Train/Validation/Test 분할
- [ ] 베이스라인 모델 구축
- [ ] 하이퍼파라미터 튜닝
- [ ] 교차검증 수행
- [ ] 앙상블 고려
- [ ] 모델 해석 (SHAP, Permutation Importance)
- [ ] 성능 평가 (적절한 지표)
- [ ] 과적합 검증

### 12.3 일반적인 실수와 해결책

| 실수 | 문제 | 해결책 |
|------|------|--------|
| **스케일링 누락** | 거리 기반 알고리즘 성능 저하 | StandardScaler, MinMaxScaler 적용 |
| **데이터 누수** | 과도하게 높은 성능 | Train/Test 분리 후 전처리 |
| **잘못된 평가지표** | 불균형 데이터에 Accuracy 사용 | F1-Score, AUC 사용 |
| **과적합** | Train 성능 ≫ Test 성능 | 정규화, 교차검증, 조기종료 |
| **하이퍼파라미터 기본값** | 준최적 성능 | GridSearch, RandomSearch |
| **클래스 불균형 무시** | 소수 클래스 무시 | SMOTE, class_weight 조정 |
| **시간 순서 무시** | 시계열 데이터 랜덤 분할 | TimeSeriesSplit 사용 |

---

## 부록: 알고리즘 복잡도

| 알고리즘 | 학습 복잡도 | 예측 복잡도 | 메모리 |
|----------|-------------|-------------|--------|
| Logistic Regression | O(nd) | O(d) | O(d) |
| Decision Tree | O(nd log n) | O(log n) | O(n) |
| Random Forest | O(mnd log n) | O(m log n) | O(mn) |
| SVM | O(n²~n³) | O(nsv × d) | O(nsv × d) |
| KNN | O(1) | O(nd) | O(nd) |
| Neural Network | O(nde) | O(de) | O(de) |
| XGBoost | O(ndk log n) | O(k log n) | O(n) |
| K-Means | O(nkt) | O(k) | O(k) |
| DBSCAN | O(n log n) | - | O(n) |
| PCA | O(min(nd², n²d)) | O(d²) | O(d²) |

*n: 샘플 수, d: 피처 수, m: 트리 수, k: 클러스터/트리 수, t: 반복 횟수, e: 에포크, nsv: support vector 수*

---

**작성일**: 2024년  
**목적**: ADP 자격시험 대비 및 실무 프로젝트 가이드  
**업데이트**: 최신 라이브러리 버전 및 기법 반영

