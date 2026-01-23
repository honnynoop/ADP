# 앙상블 분석 완벽 가이드

## 1. 앙상블의 기본 개념

**앙상블(Ensemble)**은 여러 개의 약한 학습기(weak learner)를 결합하여 강한 학습기(strong learner)를 만드는 기법입니다.

### 핵심 원리
- **다양성(Diversity)**: 서로 다른 모델들이 서로 다른 오류를 만들어야 함
- **집단 지성**: 여러 모델의 예측을 종합하면 개별 모델보다 안정적이고 정확함
- **편향-분산 트레이드오프**: 앙상블을 통해 분산을 줄이거나 편향을 줄일 수 있음

## 2. 주요 앙상블 기법 비교표

| 기법 | 학습 방식 | 샘플링 | 모델 다양성 확보 방법 | 주요 목적 | 대표 알고리즘 |
|------|----------|--------|---------------------|----------|--------------|
| **Bagging** | 병렬 학습 | Bootstrap (복원추출) | 데이터 샘플링 | 분산 감소 | Random Forest |
| **Boosting** | 순차 학습 | 가중치 조정 | 이전 모델의 오류에 집중 | 편향 감소 | AdaBoost, Gradient Boosting, XGBoost |
| **Stacking** | 2단계 학습 | 전체 데이터 | 서로 다른 알고리즘 사용 | 예측 성능 극대화 | Stacked Generalization |
| **Voting** | 병렬 학습 | 전체 데이터 | 서로 다른 알고리즘 사용 | 안정성 향상 | Hard/Soft Voting |

## 3. 세부 기법 설명

### 3.1 Bagging (Bootstrap Aggregating)

**원리**: 원본 데이터에서 복원 추출로 여러 부트스트랩 샘플을 만들고, 각각 독립적으로 모델 학습

```
원본 데이터 → [부트스트랩 샘플1] → 모델1 ┐
           → [부트스트랩 샘플2] → 모델2 ├→ 평균/투표 → 최종 예측
           → [부트스트랩 샘플3] → 모델3 ┘
```

**특징**:
- 분산이 큰 모델(예: 의사결정나무)에 효과적
- 과적합 방지
- 병렬 처리 가능

### 3.2 Boosting

**원리**: 약한 학습기를 순차적으로 학습시키며, 이전 모델이 틀린 샘플에 더 집중

#### AdaBoost
```
데이터(균등 가중치) → 모델1 → 오분류 샘플 가중치↑ 
                          → 모델2 → 오분류 샘플 가중치↑
                                 → 모델3 → ... → 가중 투표
```

#### Gradient Boosting
```
실제값 - 예측값 = 잔차(residual)
모델1 → 잔차1 → 모델2 → 잔차2 → 모델3 → ... → 합산
```

**특징**:
- 편향을 줄이는 데 효과적
- 순차 학습이므로 병렬화 어려움
- 과적합 위험 존재 (조기 종료, 학습률 조정 필요)

### 3.3 Stacking (Stacked Generalization)

**원리**: 1단계 모델들의 예측을 새로운 특징으로 사용하여 2단계 메타 모델 학습

**구조**:
```
원본 데이터
    ↓
┌───┴───┬───────┬───────┐
│       │       │       │
모델1   모델2   모델3   모델4  (Base Models)
│       │       │       │
└───┬───┴───┬───┴───┬───┘
    ↓       ↓       ↓
[예측1, 예측2, 예측3, 예측4] ← 새로운 특징
    ↓
메타 모델 (Meta Learner)
    ↓
최종 예측
```

**교차 검증 방식**:
```
학습 데이터를 K-Fold로 분할

Fold 1: 검증용 | Fold 2-K: 학습용 → 모델1 학습 → Fold 1 예측
Fold 2: 검증용 | Fold 1,3-K: 학습용 → 모델1 학습 → Fold 2 예측
...
→ 전체 학습 데이터에 대한 out-of-fold 예측 생성
→ 이를 메타 모델의 학습 데이터로 사용
```

## 4. 선형 결합의 의미

### 4.1 개념

앙상블에서 **선형 결합**은 여러 모델의 예측을 가중 평균하는 것을 의미합니다.

**수식**:
```
F(x) = w₁f₁(x) + w₂f₂(x) + ... + wₙfₙ(x)

여기서:
- F(x): 최종 예측
- fᵢ(x): i번째 모델의 예측
- wᵢ: i번째 모델의 가중치 (∑wᵢ = 1)
```

### 4.2 적용 예시

| 앙상블 기법 | 선형 결합 방식 | 가중치 |
|------------|---------------|--------|
| **Simple Averaging** | F(x) = (f₁ + f₂ + ... + fₙ)/n | 모두 1/n |
| **Weighted Averaging** | F(x) = w₁f₁ + w₂f₂ + ... + wₙfₙ | 성능 기반 |
| **AdaBoost** | F(x) = sign(∑αᵢfᵢ(x)) | 모델 정확도 기반 α |
| **Gradient Boosting** | F(x) = f₀ + η∑fᵢ | 학습률 η 적용 |

### 4.3 회귀 vs 분류

**회귀 (Regression)**:
```python
# 단순 평균
predictions = [model.predict(X) for model in models]
final_pred = np.mean(predictions, axis=0)

# 가중 평균
final_pred = np.average(predictions, axis=0, weights=weights)
```

**분류 (Classification)**:
```python
# Hard Voting: 다수결
predictions = [model.predict(X) for model in models]
final_pred = mode(predictions, axis=0)

# Soft Voting: 확률 평균
proba = [model.predict_proba(X) for model in models]
final_pred = np.argmax(np.mean(proba, axis=0), axis=1)
```

## 5. Stacking 상세 분석

### 5.1 Stacking의 장점

1. **다양한 알고리즘 활용**: 각 모델의 강점을 결합
2. **비선형 결합 가능**: 메타 모델이 복잡한 패턴 학습
3. **성능 극대화**: 일반적으로 단일 모델보다 우수

### 5.2 Stacking 구현 단계

```python
# 1단계: Base Models 정의
base_models = [
    ('rf', RandomForestClassifier()),
    ('svm', SVC(probability=True)),
    ('gb', GradientBoostingClassifier())
]

# 2단계: Out-of-fold 예측 생성
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros((len(X_train), len(base_models)))

for i, (name, model) in enumerate(base_models):
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        oof_predictions[val_idx, i] = model.predict_proba(X_val)[:, 1]

# 3단계: Meta Model 학습
meta_model = LogisticRegression()
meta_model.fit(oof_predictions, y_train)

# 4단계: 테스트 예측
test_predictions = np.column_stack([
    model.predict_proba(X_test)[:, 1] 
    for name, model in base_models
])
final_pred = meta_model.predict(test_predictions)
```

### 5.3 Stacking vs Blending

| 특징 | Stacking | Blending |
|------|----------|----------|
| 데이터 사용 | K-Fold 교차검증 | Holdout 방식 |
| 메타 학습 데이터 | Out-of-fold 예측 | Validation set 예측 |
| 데이터 활용도 | 높음 (전체 활용) | 낮음 (일부만 메타 학습) |
| 계산 비용 | 높음 | 낮음 |
| 과적합 위험 | 낮음 | 상대적으로 높음 |

## 6. 앙상블 기법별 적용 상황

### 6.1 언제 무엇을 사용할까?

| 상황 | 추천 기법 | 이유 |
|------|----------|------|
| 고분산 모델 안정화 | Bagging | 분산 감소 효과 |
| 약한 모델 성능 향상 | Boosting | 편향 감소, 순차 학습 |
| 최고 성능 필요 | Stacking | 다양한 모델 강점 결합 |
| 빠른 학습 필요 | Bagging | 병렬 처리 가능 |
| 해석 가능성 중요 | Voting | 단순 결합 |
| 불균형 데이터 | Boosting | 어려운 샘플에 집중 |

### 6.2 Python 라이브러리

```python
# Bagging
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# Boosting
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

# Voting
from sklearn.ensemble import VotingClassifier

# Stacking
from sklearn.ensemble import StackingClassifier
```

## 7. 주요 하이퍼파라미터

### Random Forest (Bagging)
- `n_estimators`: 트리 개수
- `max_features`: 분할 시 고려할 특징 수
- `max_depth`: 트리 최대 깊이

### Gradient Boosting
- `n_estimators`: 부스팅 단계 수
- `learning_rate`: 학습률 (0.01~0.1)
- `max_depth`: 트리 깊이 (3~10)
- `subsample`: 샘플링 비율

### Stacking
- Base models 선택
- Meta model 선택
- CV 전략 (K-Fold 수)

## 8. 성능 비교 예시

```
단일 모델 정확도:
- Decision Tree: 0.82
- SVM: 0.85
- Logistic Regression: 0.83

앙상블 정확도:
- Random Forest (Bagging): 0.88
- Gradient Boosting: 0.90
- Voting (3 models): 0.87
- Stacking (3 base + LR meta): 0.91
```

## 9. 주의사항

1. **과적합 위험**: 특히 Boosting과 Stacking
2. **계산 비용**: 모델 수만큼 학습 시간 증가
3. **다양성 확보**: 비슷한 모델 여러 개보다 다른 종류의 모델이 효과적
4. **데이터 누수**: Stacking 시 교차검증 필수
5. **해석 가능성**: 블랙박스 모델이 되기 쉬움

---

## 부록: 앙상블 기법 요약 다이어그램

### Bagging 흐름도
```
원본 데이터 (N개)
    ↓
Bootstrap 샘플링 (복원추출)
    ↓
┌─────────┬─────────┬─────────┐
샘플1(N개) 샘플2(N개) 샘플3(N개)
    ↓         ↓         ↓
  모델1      모델2      모델3
    ↓         ↓         ↓
└─────────┴─────────┴─────────┘
    ↓
평균(회귀) 또는 투표(분류)
    ↓
최종 예측
```

### Boosting 흐름도
```
초기 데이터 (균등 가중치)
    ↓
모델1 학습 → 예측 → 오분류 샘플 식별
    ↓
가중치 업데이트 (오분류↑)
    ↓
모델2 학습 → 예측 → 오분류 샘플 식별
    ↓
가중치 업데이트 (오분류↑)
    ↓
모델3 학습 → ...
    ↓
가중 결합 (α₁·모델1 + α₂·모델2 + ...)
    ↓
최종 예측
```

### Stacking 흐름도
```
학습 데이터
    ↓
K-Fold 분할
    ↓
┌──────────────────────────────┐
│ Base Models (병렬 학습)       │
│  ┌────┬────┬────┬────┐       │
│  │RF  │SVM │GB  │KNN │       │
│  └─┬──┴──┬─┴──┬─┴──┬─┘       │
│    ↓     ↓    ↓    ↓         │
│ Out-of-Fold Predictions      │
└──────────────────────────────┘
    ↓
[pred1, pred2, pred3, pred4] → 새로운 특징
    ↓
Meta Model (LogisticRegression 등)
    ↓
최종 예측
```

---

**빅데이터분석기사 시험 준비 화이팅!** 🎯
