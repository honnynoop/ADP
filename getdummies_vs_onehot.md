# `pd.get_dummies` vs `OneHotEncoder` 완벽 비교 가이드

> 범주형 데이터를 수치형으로 변환하는 두 가지 대표 방법의 차이, 장단점, 사용 시점을 정리한 레퍼런스

---

## 1. 개요 비교

| 항목 | `pd.get_dummies` | `OneHotEncoder` |
|------|-----------------|-----------------|
| 출처 | **pandas** 라이브러리 | **scikit-learn** (`preprocessing`) |
| 주 용도 | 탐색적 분석, 간단한 변환 | ML 파이프라인, 모델 학습 |
| 반환 타입 | `DataFrame` | `numpy array` / `sparse matrix` |
| fit/transform 분리 | ❌ 없음 | ✅ 있음 (`fit`, `transform`, `fit_transform`) |
| 학습 데이터 기억 | ❌ 불가 | ✅ 가능 (학습 카테고리 저장) |
| 미학습 카테고리 처리 | 자동 새 열 생성 (불일치 위험) | `handle_unknown` 옵션으로 제어 |
| 희소 행렬(Sparse) | ❌ 미지원 | ✅ 지원 (`sparse_output=True`) |
| Pipeline 통합 | ❌ 불가 | ✅ 가능 |

---

## 2. 기본 사용법

### 2-1. `pd.get_dummies`

```python
import pandas as pd

df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue'],
    'size':  ['S', 'M', 'L', 'S']
})

# 기본 사용
result = pd.get_dummies(df, columns=['color', 'size'])
print(result)
```

```
   color_blue  color_green  color_red  size_L  size_M  size_S
0       False        False       True   False   False    True
1        True        False      False   False    True   False
2       False         True      False    True   False   False
3        True        False      False   False   False    True
```

**주요 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `columns` | 인코딩할 컬럼 지정 | `None` (object 전체) |
| `prefix` | 열 이름 접두어 | 컬럼명 |
| `drop_first` | 첫 번째 더미 변수 제거 (다중공선성 방지) | `False` |
| `dtype` | 출력 데이터 타입 | `bool` |
| `dummy_na` | NaN을 별도 열로 처리 | `False` |

```python
# drop_first=True : 다중공선성 방지 (k-1개 더미 생성)
result = pd.get_dummies(df, columns=['color'], drop_first=True)

# dtype 지정
result = pd.get_dummies(df, columns=['color'], dtype=int)
```

---

### 2-2. `OneHotEncoder`

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue'],
    'size':  ['S', 'M', 'L', 'S']
})

# fit & transform
encoder = OneHotEncoder(sparse_output=False)  # sklearn 1.2+
encoder.fit(df[['color', 'size']])

result = encoder.transform(df[['color', 'size']])
print(encoder.get_feature_names_out())
# ['color_blue' 'color_green' 'color_red' 'size_L' 'size_M' 'size_S']
```

**주요 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `categories` | 카테고리 직접 지정 | `'auto'` |
| `drop` | 더미 변수 제거 방식 (`'first'`, `'if_binary'`) | `None` |
| `sparse_output` | 희소 행렬 출력 여부 | `True` |
| `handle_unknown` | 미학습 카테고리 처리 (`'error'`, `'ignore'`, `'infrequent_if_exist'`) | `'error'` |
| `min_frequency` | 최소 빈도 미만은 infrequent로 묶기 | `None` |
| `max_categories` | 최대 카테고리 수 | `None` |

```python
# 미학습 카테고리 무시 (테스트 데이터 안전 처리)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(df[['color']])

# 테스트 데이터에 새 카테고리 'yellow' 등장
test = pd.DataFrame({'color': ['yellow']})
encoder.transform(test)  # 모두 0으로 처리됨 → 에러 없음
```

---

## 3. 핵심 차이: Train/Test 분리 문제

### `pd.get_dummies` — 위험한 케이스 ⚠️

```python
train = pd.DataFrame({'color': ['red', 'blue']})
test  = pd.DataFrame({'color': ['blue', 'green']})  # green은 train에 없음

train_enc = pd.get_dummies(train)
test_enc  = pd.get_dummies(test)

print(train_enc.columns.tolist())  # ['color_blue', 'color_red']
print(test_enc.columns.tolist())   # ['color_blue', 'color_green']  ← 열 불일치!
```

> ❌ **문제:** 학습/테스트 열 구조가 달라져 모델이 오작동하거나 에러 발생

**임시 해결책 (비권장):**

```python
test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)
```

---

### `OneHotEncoder` — 안전한 처리 ✅

```python
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(train[['color']])  # red, blue 학습

# test에서 green 등장 → 0으로 처리, 열 구조 유지
encoder.transform(test[['color']])
# [[1. 0.]   ← blue
#  [0. 0.]]  ← green → 전부 0 (unknown)
```

---

## 4. Pipeline 통합

### `OneHotEncoder` + `Pipeline` (권장 패턴)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

cat_cols = ['color', 'size']
num_cols = ['price', 'weight']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

pipeline = Pipeline(steps=[
    ('prep', preprocessor),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
pipeline.predict(X_test)  # 전처리 + 예측 일괄 처리
```

> ✅ `get_dummies`는 Pipeline에 통합 불가 — scikit-learn의 `fit/transform` 인터페이스를 따르지 않음

---

## 5. 장단점 정리

### `pd.get_dummies`

| 구분 | 내용 |
|------|------|
| ✅ 장점 | 코드가 간결하고 직관적 |
| ✅ 장점 | 결과가 바로 DataFrame으로 반환 (가독성↑) |
| ✅ 장점 | EDA, 빠른 프로토타이핑에 최적 |
| ✅ 장점 | 별도 import 없이 pandas만으로 사용 가능 |
| ❌ 단점 | fit/transform 분리 없음 → 데이터 누수 위험 |
| ❌ 단점 | Train/Test 열 불일치 문제 발생 가능 |
| ❌ 단점 | sklearn Pipeline에 통합 불가 |
| ❌ 단점 | 새로운 카테고리 자동 처리 없음 |
| ❌ 단점 | 희소 행렬 미지원 → 고차원 데이터에 메모리 낭비 |

---

### `OneHotEncoder`

| 구분 | 내용 |
|------|------|
| ✅ 장점 | fit/transform 분리 → 데이터 누수 방지 |
| ✅ 장점 | Train 카테고리 기억 → Test 열 일관성 보장 |
| ✅ 장점 | `handle_unknown` 옵션으로 유연한 처리 |
| ✅ 장점 | Pipeline/ColumnTransformer 통합 가능 |
| ✅ 장점 | 희소 행렬 지원 → 메모리 효율적 |
| ✅ 장점 | `min_frequency`, `max_categories`로 희귀 카테고리 제어 |
| ❌ 단점 | 코드가 상대적으로 복잡 |
| ❌ 단점 | 결과가 numpy array → DataFrame으로 복원 필요 |
| ❌ 단점 | 간단한 분석에는 과한 설정 |

---

## 6. 사용 권장 / 비권장 시점

### ✅ `pd.get_dummies` 사용 권장

| 상황 | 이유 |
|------|------|
| EDA, 데이터 탐색 단계 | 빠르게 데이터 구조 확인 |
| 주피터 노트북 일회성 분석 | 결과를 DataFrame으로 바로 확인 |
| 단순 선형 모델, 상관 분석 | 전처리 파이프라인 불필요 |
| Train/Test 분리가 없는 경우 | 데이터 누수 걱정 없음 |
| 카테고리 수가 적고 변동이 없는 경우 | 열 불일치 위험 낮음 |
| 빠른 프로토타이핑 | 코드 단순성 우선 |

### ❌ `pd.get_dummies` 사용 비권장

| 상황 | 이유 |
|------|------|
| Train/Test split이 있는 ML 모델 | 열 불일치, 데이터 누수 위험 |
| Production 환경, 배포 모델 | 새 카테고리 대응 불가 |
| Cross-Validation | fold마다 카테고리가 달라질 수 있음 |
| sklearn Pipeline 사용 | 호환 불가 |
| 고차원 카테고리 (카테고리 수 많음) | 희소 행렬 미지원으로 메모리 낭비 |

---

### ✅ `OneHotEncoder` 사용 권장

| 상황 | 이유 |
|------|------|
| ML 모델 학습/예측 파이프라인 | fit/transform 분리로 안전 |
| Train/Test split이 있는 경우 | 열 구조 일관성 보장 |
| Cross-Validation | 각 fold에서 일관된 변환 |
| Production/Serving 환경 | 미학습 카테고리 처리 가능 |
| 카테고리 수가 매우 많은 경우 | Sparse matrix로 메모리 절약 |
| ColumnTransformer와 함께 | 수치형/범주형 혼합 처리 |

### ❌ `OneHotEncoder` 사용 비권장

| 상황 | 이유 |
|------|------|
| 단순 EDA, 빠른 확인 | 코드 오버헤드가 큼 |
| 완성된 분석 결과 공유 (DataFrame 필요) | numpy array 반환으로 가독성↓ |

---

## 7. 다중공선성(Multicollinearity) 처리

범주형 변수의 카테고리 수가 k개일 때, k개의 더미 변수를 모두 생성하면 **완전 다중공선성** 발생

> 예: `color = {red, blue, green}` → `color_red + color_blue + color_green = 1` (항등식)

| 방법 | 코드 |
|------|------|
| `get_dummies` | `drop_first=True` |
| `OneHotEncoder` | `drop='first'` 또는 `drop='if_binary'` |

```python
# get_dummies
pd.get_dummies(df, columns=['color'], drop_first=True)

# OneHotEncoder
OneHotEncoder(drop='first')          # 항상 첫 번째 카테고리 제거
OneHotEncoder(drop='if_binary')      # 이진 카테고리만 제거
```

> **선형 회귀, 로지스틱 회귀**에서는 반드시 drop 필요  
> **트리 기반 모델 (XGBoost, RandomForest)** 에서는 drop 불필요

---

## 8. 희귀 카테고리 처리 (sklearn 1.1+)

```python
# 빈도 2 미만이면 'infrequent_sklearn' 카테고리로 묶기
encoder = OneHotEncoder(min_frequency=2, handle_unknown='infrequent_if_exist')
encoder.fit(df[['color']])
```

> `get_dummies`는 이 기능이 없음 → 희귀 카테고리가 많으면 `OneHotEncoder` 필수

---

## 9. 결론 요약

```
분석/EDA 단계  →  pd.get_dummies  (빠르고 간단)
ML 모델 개발  →  OneHotEncoder   (안전하고 유연)
```

| 기준 | 선택 |
|------|------|
| 빠른 탐색, 일회성 분석 | `get_dummies` |
| 학습/예측 파이프라인 | `OneHotEncoder` |
| Train/Test 분리 있음 | `OneHotEncoder` |
| 배포/서빙 환경 | `OneHotEncoder` |
| 희소 행렬 필요 | `OneHotEncoder` |
| Cross-Validation | `OneHotEncoder` |
| pandas DataFrame 그대로 유지 | `get_dummies` |

---

> 📌 **핵심 원칙:** `get_dummies`는 **데이터를 보기 위한** 도구, `OneHotEncoder`는 **모델을 만들기 위한** 도구
