# ADP 필기 서술형 핵심 R 함수 정리

## 1. 군집분석 (Clustering)

### K-means 군집분석

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `kmeans(data, centers=k)` |
| **주요 속성** | `$centers`, `$cluster`, `$withinss`, `$betweenss`, `$tot.withinss`, `$totss` |
| **핵심 지표** | **BSS/TSS 비율** = `between_ss / total_ss` <br> → **1에 가까울수록 좋은 모델** (군집 간 분리 우수) |
| **최적 K 선정** | • Elbow Method: `tot.withinss` 그래프 <br> • Silhouette: `cluster::silhouette()` <br> • `NbClust` 패키지 활용 |
| **인사이트** | • 각 군집의 중심점(`$centers`) 해석 <br> • 군집 크기 불균형 확인 `table(km$cluster)` <br> • WSS(Within Sum of Squares) 감소 추이 분석 |

---

### 계층적 군집분석

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `dist()` → `hclust(method="complete")` → `cutree(h=높이)` |
| **거리 측정** | `dist(method="euclidean")`, `"manhattan"`, `"binary"` |
| **연결 방법** | `complete`(최대), `single`(최소), `average`(평균), `ward.D2`(분산최소) |
| **덴드로그램** | `plot(hc)`, `rect.hclust(hc, k=3)` - 군집 수 시각화 |
| **최적 군집 수** | `cutree(hc, k=3)` 또는 `cutree(hc, h=높이)` |
| **인사이트** | • Ward 방법: 군집 내 분산 최소화 <br> • Dendrogram 높이로 군집 간 거리 파악 <br> • 소속 군집별 특성 비교 분석 |

---

## 2. 분류분석 (Classification)

### 의사결정나무

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `rpart(formula, data, method="class")` <br> `C5.0(x, y)` |
| **가지치기** | `rpart`: `cp` (complexity parameter) <br> `printcp(model)`, `plotcp(model)` |
| **변수 중요도** | `model$variable.importance` (rpart) <br> `C5.0::C5imp(model)` |
| **핵심 지표** | • **정확도(Accuracy)**: `(TP+TN)/전체` <br> • **민감도(Sensitivity)**: `TP/(TP+FN)` <br> • **특이도(Specificity)**: `TN/(TN+FP)` <br> • **정밀도(Precision)**: `TP/(TP+FP)` |
| **평가** | `confusionMatrix()` - caret 패키지 |
| **시각화** | `plot(model)`, `text(model)`, `rpart.plot::rpart.plot()` |
| **인사이트** | • 중요 분할 변수 파악 <br> • 과적합 방지 위해 cp 값 조정 <br> • 규칙 기반 해석 용이 |

---

### 랜덤포레스트

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `randomForest(formula, data, ntree=500, mtry)` |
| **주요 매개변수** | • `ntree`: 트리 개수 (기본 500) <br> • `mtry`: 분할 시 고려할 변수 수 (분류: √p, 회귀: p/3) |
| **변수 중요도** | `importance(model)` <br> • **MeanDecreaseAccuracy**: 정확도 감소량 <br> • **MeanDecreaseGini**: 지니계수 감소량 |
| **OOB 오차** | `model$err.rate` - Out-of-Bag 오차율 <br> **OOB 낮을수록 좋은 모델** |
| **시각화** | `varImpPlot(model)` - 변수 중요도 플롯 |
| **인사이트** | • OOB로 별도 검증 세트 불필요 <br> • Top 변수들로 모델 간소화 가능 <br> • 트리 수 증가 시 성능 수렴 확인 |

---

### 로지스틱 회귀

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `glm(formula, family=binomial(link="logit"), data)` |
| **계수 해석** | `exp(coef(model))` = **Odds Ratio** <br> OR > 1: 양의 영향, OR < 1: 음의 영향 |
| **유의성 검정** | `summary(model)` - **z value, Pr(>|z|)** <br> p-value < 0.05: 유의한 변수 |
| **모델 적합도** | • **AIC**: 낮을수록 좋음 <br> • **Deviance**: 잔차일탈도 (낮을수록 좋음) <br> • **Pseudo R²**: McFadden R² |
| **예측 및 평가** | `predict(model, type="response")` <br> `ROCR` 패키지로 **ROC 커브, AUC** 계산 |
| **임계값 설정** | `ifelse(pred > 0.5, 1, 0)` - 기본 0.5 |
| **인사이트** | • **AUC > 0.8**: 우수한 분류 <br> • 회귀계수로 변수별 기여도 파악 <br> • VIF로 다중공선성 확인 |

---

### SVM (Support Vector Machine)

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `e1071::svm(formula, data, kernel, cost, gamma)` |
| **커널 종류** | `linear`, `polynomial`, `radial`(RBF), `sigmoid` |
| **주요 매개변수** | • **cost(C)**: 오분류 허용도 (작을수록 여유, 클수록 엄격) <br> • **gamma**: RBF 커널 폭 (클수록 복잡한 경계) |
| **튜닝** | `tune.svm()` - 그리드 서치로 최적 파라미터 탐색 |
| **평가** | `confusionMatrix()`, 정확도, 민감도 등 |
| **인사이트** | • 비선형 분류에 강력 (RBF 커널) <br> • 고차원 데이터 효과적 <br> • Overfitting 주의 (gamma, cost 조정) |

---

### 나이브베이즈

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `e1071::naiveBayes(formula, data)` <br> `klaR::NaiveBayes()` |
| **가정** | **조건부 독립 가정**: P(X|Y) = ∏P(Xi|Y) |
| **예측** | `predict(model, newdata, type="class")` <br> `type="raw"`: 확률값 반환 |
| **평가** | Confusion Matrix, 정확도 |
| **장점** | • 빠른 학습 및 예측 <br> • 소규모 데이터에도 효과적 <br> • 텍스트 분류에 강함 |
| **인사이트** | • 사전확률과 조건부확률 해석 <br> • 범주형 데이터에 적합 <br> • 변수 간 독립성 가정 검토 필요 |

---

## 3. 회귀분석 (Regression)

### 선형회귀

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `lm(formula, data)` |
| **모델 요약** | `summary(model)` |
| **핵심 지표** | • **R²**: 설명력 (0~1, 높을수록 좋음) <br> • **Adjusted R²**: 변수 수 고려한 설명력 <br> • **F-statistic**: 모델 전체 유의성 (p < 0.05) <br> • **RMSE**: √(Σ(실제-예측)²/n) - 낮을수록 좋음 |
| **회귀계수** | `coef(model)` - 각 변수의 영향력 <br> **t-value, Pr(>|t|)**: 개별 변수 유의성 |
| **가정 검토** | • 정규성: `shapiro.test(residuals)` <br> • 등분산성: `ncvTest()`, `bptest()` <br> • 독립성: `durbinWatsonTest()` <br> • 다중공선성: `vif()` (VIF > 10 문제) |
| **예측** | `predict(model, newdata, interval="confidence")` |
| **인사이트** | • 회귀계수로 변수별 영향도 해석 <br> • 잔차분석으로 모델 적합성 확인 <br> • VIF로 변수 선택 개선 |

---

## 4. 연관분석 (Association Rule Mining)

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `arules::apriori(data, parameter=list(supp, conf, minlen))` |
| **데이터 형식** | `as(data, "transactions")` - 트랜잭션 객체 변환 |
| **핵심 지표** | • **Support(지지도)**: P(A∩B) - 동시 출현 빈도 <br> • **Confidence(신뢰도)**: P(B|A) = P(A∩B)/P(A) <br> • **Lift(향상도)**: P(B|A)/P(B) <br>   → **Lift > 1**: 양의 연관성 (좋은 규칙) <br>   → Lift = 1: 독립, Lift < 1: 음의 연관성 |
| **규칙 추출** | `inspect(rules)` - 규칙 확인 <br> `sort(rules, by="lift")` - 정렬 |
| **필터링** | `subset(rules, subset=lift>2 & confidence>0.6)` |
| **시각화** | `arulesViz::plot(rules)` |
| **인사이트** | • Lift 높은 규칙으로 교차판매 전략 <br> • 최소 지지도/신뢰도로 의미있는 규칙 필터링 <br> • 장바구니 분석, 추천시스템 활용 |

---

## 5. 차원축소 (Dimensionality Reduction)

### 주성분분석 (PCA)

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `prcomp(data, scale=TRUE)` <br> `princomp(data, cor=TRUE)` |
| **표준화** | `scale=TRUE` 필수 (변수 단위 다를 때) |
| **주요 속성** | • `$rotation`: 주성분 로딩 (고유벡터) <br> • `$sdev`: 표준편차 <br> • `$x`: 주성분 점수 |
| **설명 분산** | `summary(pca)` - **Cumulative Proportion** <br> • 일반적으로 **누적 80~90% 이상** 설명하는 PC까지 사용 |
| **스크리 플롯** | `screeplot(pca)` - Elbow Point 확인 |
| **바이플롯** | `biplot(pca)` - 변수와 관측치 동시 시각화 |
| **주성분 개수** | • 고유값(eigenvalue) ≥ 1 <br> • 누적 설명력 80% 이상 |
| **인사이트** | • 다중공선성 제거 <br> • 차원 축소로 시각화 용이 <br> • PC1, PC2로 주요 패턴 파악 |

---

## 6. 시계열 분석 (Time Series)

### ARIMA 모델

| 구분 | 내용 |
|------|------|
| **핵심 함수** | `arima(x, order=c(p,d,q))` <br> `forecast::auto.arima(x)` |
| **차분 확인** | `diff(data, differences=d)` <br> **정상성 검정**: `adf.test()` (p < 0.05: 정상) |
| **ACF/PACF** | `acf(data)`, `pacf(data)` <br> • ACF로 q 결정 (MA 차수) <br> • PACF로 p 결정 (AR 차수) |
| **모델 선택** | **AIC/BIC 최소화** - `auto.arima()` 자동 선택 |
| **예측** | `forecast(model, h=기간)` |
| **잔차 검정** | `Box.test(residuals, type="Ljung-Box")` <br> p > 0.05: 백색잡음 (good) |
| **인사이트** | • 차분(d)으로 추세 제거 <br> • AR(p): 과거 값의 영향 <br> • MA(q): 과거 오차의 영향 |

---

## 7. 앙상블 기법

### 배깅/부스팅

| 구분 | 내용 |
|------|------|
| **배깅** | `ipred::bagging()` <br> • 부트스트랩 샘플링 → 병렬 학습 <br> • 분산 감소 효과 |
| **부스팅** | `gbm()` - Gradient Boosting <br> `xgboost()` - Extreme Gradient Boosting |
| **GBM 핵심 지표** | • `n.trees`: 트리 개수 <br> • `interaction.depth`: 트리 깊이 <br> • `shrinkage`: 학습률 (0.001~0.1) |
| **변수 중요도** | `summary(gbm_model)` - **상대적 영향력** |
| **평가** | `gbm.perf(model)` - 최적 트리 개수 |
| **인사이트** | • 배깅: 과적합 감소 (RF) <br> • 부스팅: 순차 학습으로 약한 학습기 개선 <br> • 학습률 낮추면 트리 수 증가 필요 |

---

## 8. 모델 평가 핵심 함수

### 분류 모델 평가

```r
# Confusion Matrix
library(caret)
confusionMatrix(pred, actual)

# ROC & AUC
library(ROCR)
pred_obj <- prediction(pred_prob, actual)
perf <- performance(pred_obj, "tpr", "fpr")
plot(perf)
auc <- performance(pred_obj, "auc")@y.values[[1]]
```

| 지표 | 공식 | 의미 |
|------|------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 전체 정확도 |
| **Precision** | TP/(TP+FP) | 양성 예측의 정확도 |
| **Recall (Sensitivity)** | TP/(TP+FN) | 실제 양성 탐지율 |
| **Specificity** | TN/(TN+FP) | 실제 음성 탐지율 |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | 정밀도-재현율 조화평균 |
| **AUC** | ROC 곡선 아래 면적 | 0.5~1.0, **0.8 이상 우수** |

---

### 회귀 모델 평가

```r
# RMSE
sqrt(mean((actual - pred)^2))

# MAE
mean(abs(actual - pred))

# MAPE
mean(abs((actual - pred)/actual)) * 100
```

| 지표 | 의미 | 특징 |
|------|------|------|
| **RMSE** | 오차의 제곱근 평균 | 큰 오차에 민감, 낮을수록 좋음 |
| **MAE** | 절대 오차 평균 | 모든 오차 동일 가중, 낮을수록 좋음 |
| **MAPE** | 평균 절대 백분율 오차 | 스케일 독립적, % 해석 용이 |
| **R²** | 결정계수 | 0~1, 설명력 나타냄 |

---

## 9. 교차검증 (Cross-Validation)

| 구분 | 내용 |
|------|------|
| **K-Fold CV** | `caret::trainControl(method="cv", number=10)` <br> `train(formula, data, method, trControl)` |
| **LOOCV** | `method="LOOCV"` - 관측치 하나씩 검증 |
| **Repeated CV** | `method="repeatedcv", number=10, repeats=3` |
| **인사이트** | • 과적합 방지 <br> • 모델 일반화 성능 평가 <br> • 10-Fold CV가 일반적 |

---

## 10. 데이터 전처리 핵심 함수

| 함수 | 용도 |
|------|------|
| `scale()` | 표준화 (평균 0, 분산 1) |
| `normalize()` | 정규화 (0~1 범위) |
| `na.omit()` | 결측치 제거 |
| `mice::mice()` | 결측치 대치 (다중대치) |
| `caret::preProcess()` | 전처리 통합 (center, scale, pca 등) |
| `dummyVars()` | 범주형 → 더미변수 변환 |
| `createDataPartition()` | 층화추출 데이터 분할 |

---

## 핵심 암기 포인트

### 모델 성능 판단 기준
- **군집**: BSS/TSS → **1에 가까울수록 좋음**
- **분류**: AUC → **0.8 이상 우수**
- **회귀**: R² → **높을수록 좋음**, RMSE → **낮을수록 좋음**
- **연관규칙**: Lift → **1 초과 시 연관성 있음**
- **랜덤포레스트**: OOB Error → **낮을수록 좋음**

### 변수 선택
- **VIF > 10**: 다중공선성 문제 (제거 고려)
- **p-value < 0.05**: 통계적으로 유의한 변수
- **Variable Importance**: 상위 변수 선택

### 주의사항
- PCA: 반드시 **표준화(scale=TRUE)** 필요
- 로지스틱: **exp(coef)** = Odds Ratio 해석
- ARIMA: **정상성 확인** 후 차분(d) 결정
- 교차검증: 과적합 방지를 위해 **필수**

---

**작성일**: 2026-02-04  
**용도**: ADP 필기 서술형 시험 대비
