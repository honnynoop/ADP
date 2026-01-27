# 빅데이터 분석 종합 학습 자료

## 목차
1. [EDA (탐색적 데이터 분석)](#1-eda-탐색적-데이터-분석)
2. [데이터 마이닝 vs 시뮬레이션 평가 지표](#2-데이터-마이닝-vs-시뮬레이션-평가-지표)
3. [R 데이터 처리 핵심 함수](#3-r-데이터-처리-핵심-함수)
4. [택배 차량 배치 최적화 분석](#4-택배-차량-배치-최적화-분석)
5. [PCA (주성분 분석)](#5-pca-주성분-분석)

---

## 1. EDA (탐색적 데이터 분석)

### 핵심 개념

**탐색적 데이터 분석 (Exploratory Data Analysis)** - 데이터를 분석하기 전에 데이터의 특성을 파악하는 과정입니다.

#### 목적
- 데이터의 구조와 특성 이해
- 패턴, 이상치, 관계 발견
- 가설 수립 및 검증
- 적절한 분석 방법 선택

---

### 주요 기법

#### 1. **기술 통계량 (Descriptive Statistics)**

```python
import pandas as pd

# 기본 통계량
df.describe()

# 개별 통계량
df['age'].mean()      # 평균
df['age'].median()    # 중앙값
df['age'].std()       # 표준편차
df['age'].quantile([0.25, 0.75])  # 사분위수
```

**주요 지표:**
- 중심 경향성: 평균, 중앙값, 최빈값
- 산포도: 분산, 표준편차, 범위, IQR
- 분포 형태: 왜도(skewness), 첨도(kurtosis)

#### 2. **데이터 시각화**

##### 분포 확인
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 히스토그램
plt.hist(df['age'], bins=30)

# 박스플롯 (이상치 탐지)
sns.boxplot(x=df['age'])

# 바이올린 플롯
sns.violinplot(x=df['age'])
```

##### 관계 분석
```python
# 산점도 (상관관계)
plt.scatter(df['age'], df['salary'])

# 상관관계 히트맵
sns.heatmap(df.corr(), annot=True)

# 페어플롯
sns.pairplot(df)
```

#### 3. **결측치 분석**

```python
# 결측치 확인
df.isnull().sum()
df.isnull().sum() / len(df) * 100  # 비율

# 시각화
import missingno as msno
msno.matrix(df)
msno.heatmap(df)
```

#### 4. **이상치 탐지**

```python
# IQR 방법
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['age'] < Q1 - 1.5*IQR) | (df['age'] > Q3 + 1.5*IQR)

# Z-score 방법
from scipy import stats
z_scores = np.abs(stats.zscore(df['age']))
outliers = z_scores > 3
```

---

### EDA 프로세스

```
1. 데이터 수집 및 로딩
   ↓
2. 데이터 구조 파악
   - shape, dtypes, info()
   ↓
3. 기술 통계량 확인
   - describe(), value_counts()
   ↓
4. 결측치/이상치 탐지
   ↓
5. 변수 간 관계 분석
   - 상관분석, 시각화
   ↓
6. 인사이트 도출
   ↓
7. 전처리 방향 결정
```

---

### EDA (Event-Driven Architecture)

**이벤트 주도 아키텍처** - 빅데이터 시스템에서도 사용되는 아키텍처 패턴입니다.

#### 핵심 개념

##### 구성 요소
```
Event Producer (생성자)
    ↓ (이벤트 발생)
Event Broker (중개자)
    ↓ (이벤트 전달)
Event Consumer (소비자)
```

##### 특징
- **비동기 처리**: 실시간 데이터 처리
- **느슨한 결합**: 시스템 간 독립성
- **확장성**: 수평적 확장 용이

##### 빅데이터 적용
- **Kafka**: 이벤트 스트리밍 플랫폼
- **Spark Streaming**: 실시간 처리
- **Flink**: 이벤트 기반 처리

---

## 2. 데이터 마이닝 vs 시뮬레이션 평가 지표

데이터 마이닝과 시뮬레이션은 서로 다른 평가 지표를 사용합니다.

---

### 1. 데이터 마이닝 평가 지표

#### 분류 모델 평가

##### **혼동 행렬 (Confusion Matrix) 기반**

```
              예측: Positive  예측: Negative
실제: Positive      TP              FN
실제: Negative      FP              TN
```

##### **주요 지표**

**1. 정확도 (Accuracy)**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- 전체 중 올바르게 예측한 비율
- 불균형 데이터에서는 부적절

**2. 정밀도 (Precision)**
```
Precision = TP / (TP + FP)
```
- Positive로 예측한 것 중 실제 Positive 비율
- 스팸 필터: 정상 메일을 스팸으로 오분류 최소화

**3. 재현율/검출률 (Recall/Detection Rate)**
```
Recall = TP / (TP + FN)
= Sensitivity = TPR (True Positive Rate)
```
- 실제 Positive 중 올바르게 예측한 비율
- 암 진단: 환자를 놓치지 않는 것이 중요

**4. F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Precision과 Recall의 조화평균
- 불균형 데이터 평가에 적합

**5. 리프트 (Lift)**
```
Lift = (TP / (TP + FP)) / ((TP + FN) / (TP + TN + FP + FN))
     = Precision / Support
```
- 모델이 랜덤보다 얼마나 나은지
- 마케팅: 타겟 고객 선정 효과 측정

**예시:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 정확도
accuracy = accuracy_score(y_true, y_pred)

# 정밀도
precision = precision_score(y_true, y_pred)

# 재현율
recall = recall_score(y_true, y_pred)

# F1 Score
f1 = 2 * (precision * recall) / (precision + recall)
```

#### 추가 지표

**ROC-AUC**
- ROC 곡선 아래 면적
- 0.5~1.0 (0.5는 랜덤)

**특이도 (Specificity)**
```
Specificity = TN / (TN + FP)
```
- 실제 Negative를 올바르게 예측

**G-Mean**
```
G-Mean = √(Recall × Specificity)
```
- 불균형 데이터 평가

---

### 2. 시뮬레이션 성능 지표

#### 대기열 이론 (Queueing Theory) 기반

##### **시스템 구조**
```
도착 → [대기열] → [서버] → 출발
```

##### **주요 지표**

**1. Throughput (처리량)**
```
Throughput = 단위 시간당 처리된 작업 수
```
- 시스템의 생산성 측정
- 높을수록 좋음

**2. Average Waiting Time (평균 대기 시간)**
```
Wq = 고객이 큐에서 대기한 평균 시간
```
- 서비스 품질의 핵심 지표
- 낮을수록 좋음

**3. Average Queue Length (평균 큐 길이)**
```
Lq = 큐에서 대기 중인 평균 고객 수
```
- 시스템 혼잡도 측정
- 낮을수록 좋음

**4. Time in System (시스템 체류 시간)**
```
W = Wq + Service Time
  = 대기 시간 + 서비스 시간
```
- 고객 관점의 총 소요 시간

**5. Utilization (이용률)**
```
ρ = λ / μ
λ: 도착률, μ: 서비스율
```
- 서버 사용률
- 0.7~0.8이 적정

**6. Average Number in System (시스템 내 평균 고객 수)**
```
L = Lq + (서비스 중인 고객 수)
```

#### Little's Law

```
L = λ × W
Lq = λ × Wq

L: 시스템 내 평균 고객 수
λ: 도착률
W: 시스템 평균 체류 시간
```

---

### 비교 정리

| 구분 | 데이터 마이닝 | 시뮬레이션 |
|------|--------------|-----------|
| **목적** | 예측 모델 평가 | 시스템 성능 평가 |
| **초점** | 정확성, 신뢰성 | 효율성, 처리 능력 |
| **주요 지표** | Accuracy, Precision, Recall, Lift | Throughput, Waiting Time, Queue Length |
| **데이터** | 과거 데이터 기반 | 확률 분포 기반 |
| **적용 분야** | 분류, 예측, 추천 | 시스템 설계, 용량 계획 |

---

### 빅데이터분석기사 출제 예상

#### 데이터 마이닝 지표 문제

**예제 1:**
```
혼동 행렬:
              예측: 암    예측: 정상
실제: 암         80         20
실제: 정상        10        890

정확도 = ?
정밀도 = ?
재현율 = ?
```

**정답:**
```
Accuracy = (80 + 890) / 1000 = 0.97 (97%)
Precision = 80 / (80 + 10) = 0.889 (88.9%)
Recall = 80 / (80 + 20) = 0.8 (80%)
```

#### 시뮬레이션 지표 문제

**예제 2:**
```
은행 창구 시뮬레이션:
- 시간당 평균 60명 도착 (λ = 1명/분)
- 평균 서비스 시간 = 0.8분 (μ = 1.25명/분)
- 평균 대기 시간 Wq = 3.2분

Little's Law를 이용하여 
평균 큐 길이 Lq를 구하시오.
```

**정답:**
```
Lq = λ × Wq
   = 1 × 3.2
   = 3.2명
```

---

### 실무 활용

#### 데이터 마이닝
```python
# 모델 평가
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
"""
              precision    recall  f1-score   support
           0       0.88      0.91      0.89       100
           1       0.91      0.88      0.89       100
"""
```

#### 시뮬레이션
```python
# SimPy를 이용한 시뮬레이션
import simpy

class Bank:
    def __init__(self, env, num_servers):
        self.env = env
        self.server = simpy.Resource(env, num_servers)
        self.wait_times = []
        
    def serve(self, customer):
        wait_start = self.env.now
        with self.server.request() as req:
            yield req
            wait_time = self.env.now - wait_start
            self.wait_times.append(wait_time)
            
            # 서비스 시간
            yield self.env.timeout(random.expovariate(1/0.8))

# 평균 대기 시간 계산
average_wait = sum(bank.wait_times) / len(bank.wait_times)
```

---

### 핵심 정리

**데이터 마이닝 평가:**
- ✅ 정확도, 정밀도, 재현율, F1
- ✅ 리프트, ROC-AUC
- ✅ 혼동 행렬 기반

**시뮬레이션 평가:**
- ✅ Throughput (처리량)
- ✅ Waiting Time (대기 시간)
- ✅ Queue Length (큐 길이)
- ✅ Little's Law 활용

시험에서는 **각 지표의 계산 방법**과 **적절한 지표 선택**이 출제됩니다!

---

## 3. R 데이터 처리 핵심 함수

빅데이터분석기사 시험에 자주 출제되는 R의 핵심 함수들을 정리합니다.

---

### 1. 데이터 결합 함수

#### **cbind (Column Bind)**
열(column) 방향으로 데이터 결합

```r
# 벡터 결합
x <- c(1, 2, 3)
y <- c(4, 5, 6)
cbind(x, y)
#      x y
# [1,] 1 4
# [2,] 2 5
# [3,] 3 6

# 데이터프레임 결합
df1 <- data.frame(name = c("Alice", "Bob"))
df2 <- data.frame(age = c(25, 30))
cbind(df1, df2)
#    name age
# 1 Alice  25
# 2   Bob  30
```

**특징:**
- 행 개수가 같아야 함
- 열을 옆으로 붙임
- 변수 추가에 사용

#### **rbind (Row Bind)**
행(row) 방향으로 데이터 결합

```r
# 벡터 결합
x <- c(1, 2, 3)
y <- c(4, 5, 6)
rbind(x, y)
#   [,1] [,2] [,3]
# x    1    2    3
# y    4    5    6

# 데이터프레임 결합
df1 <- data.frame(name = "Alice", age = 25)
df2 <- data.frame(name = "Bob", age = 30)
rbind(df1, df2)
#    name age
# 1 Alice  25
# 2   Bob  30
```

**특징:**
- 열 개수와 이름이 같아야 함
- 행을 아래로 붙임
- 관측치 추가에 사용

---

### 2. Apply 계열 함수

데이터에 함수를 반복 적용하는 함수군

#### **apply**
배열(array)이나 행렬(matrix)에 함수 적용

```r
# 행렬 생성
mat <- matrix(1:12, nrow=3, ncol=4)
#      [,1] [,2] [,3] [,4]
# [1,]    1    4    7   10
# [2,]    2    5    8   11
# [3,]    3    6    9   12

# 행 방향 합계 (MARGIN=1)
apply(mat, 1, sum)  # [1] 22 26 30

# 열 방향 합계 (MARGIN=2)
apply(mat, 2, sum)  # [1]  6 15 24 33

# 행 평균
apply(mat, 1, mean)  # [1] 5.5 6.5 7.5
```

**구문:**
```r
apply(X, MARGIN, FUN)
# X: 배열 또는 행렬
# MARGIN: 1(행), 2(열)
# FUN: 적용할 함수
```

#### **lapply (List Apply)**
리스트나 벡터에 함수 적용, **리스트 반환**

```r
# 벡터에 적용
x <- list(a = 1:5, b = 6:10)
lapply(x, mean)
# $a
# [1] 3
# $b
# [1] 8

# 함수 정의하여 적용
lapply(x, function(x) x^2)
# $a
# [1]  1  4  9 16 25
# $b
# [1] 36 49 64 81 100
```

**특징:**
- 항상 **리스트** 반환
- 벡터, 리스트, 데이터프레임에 사용
- 결과 구조 유지

#### **sapply (Simplify Apply)**
lapply의 간소화 버전, **벡터/행렬 반환**

```r
x <- list(a = 1:5, b = 6:10)

# lapply는 리스트 반환
lapply(x, mean)
# $a
# [1] 3
# $b
# [1] 8

# sapply는 벡터 반환
sapply(x, mean)
#  a  b 
#  3  8

# 데이터프레임에 적용
df <- data.frame(x = 1:5, y = 6:10)
sapply(df, sum)
#  x  y 
# 15 40
```

**특징:**
- 가능하면 **벡터나 행렬**로 단순화
- lapply보다 출력이 간단
- 가장 많이 사용됨

#### **tapply (Table Apply)**
그룹별로 함수 적용

```r
# 데이터 생성
age <- c(25, 30, 25, 30, 25)
salary <- c(3000, 4000, 3500, 4500, 3200)

# 나이별 평균 급여
tapply(salary, age, mean)
#   25   30 
# 3233 4250

# 복합 그룹
gender <- c("M", "F", "M", "F", "M")
tapply(salary, list(age, gender), mean)
#      F    M
# 25  NA 3100
# 30 4250   NA
```

**구문:**
```r
tapply(X, INDEX, FUN)
# X: 벡터
# INDEX: 그룹 인덱스
# FUN: 적용할 함수
```

**특징:**
- GROUP BY와 유사
- 범주형 변수로 그룹화
- 집계 함수에 주로 사용

---

### 3. 데이터 병합

#### **merge**
공통 키를 기준으로 데이터프레임 병합 (SQL JOIN과 유사)

```r
# 데이터프레임 생성
df1 <- data.frame(id = c(1, 2, 3), 
                  name = c("Alice", "Bob", "Carol"))
df2 <- data.frame(id = c(1, 2, 4), 
                  age = c(25, 30, 35))

# Inner Join (기본)
merge(df1, df2, by = "id")
#   id  name age
# 1  1 Alice  25
# 2  2   Bob  30

# Left Join
merge(df1, df2, by = "id", all.x = TRUE)
#   id  name age
# 1  1 Alice  25
# 2  2   Bob  30
# 3  3 Carol  NA

# Right Join
merge(df1, df2, by = "id", all.y = TRUE)
#   id  name age
# 1  1 Alice  25
# 2  2   Bob  30
# 3  4  <NA>  35

# Full Outer Join
merge(df1, df2, by = "id", all = TRUE)
#   id  name age
# 1  1 Alice  25
# 2  2   Bob  30
# 3  3 Carol  NA
# 4  4  <NA>  35
```

**주요 옵션:**

| 옵션 | 설명 | SQL 동등 |
|------|------|---------|
| `by = "col"` | 병합 키 지정 | ON |
| `all = FALSE` | Inner Join | INNER JOIN |
| `all.x = TRUE` | Left Join | LEFT JOIN |
| `all.y = TRUE` | Right Join | RIGHT JOIN |
| `all = TRUE` | Full Outer Join | FULL OUTER JOIN |

**다른 열 이름으로 병합:**
```r
df1 <- data.frame(id = c(1, 2, 3), name = c("Alice", "Bob", "Carol"))
df2 <- data.frame(user_id = c(1, 2, 4), age = c(25, 30, 35))

merge(df1, df2, by.x = "id", by.y = "user_id")
#   id  name age
# 1  1 Alice  25
# 2  2   Bob  30
```

---

### 종합 비교표

#### 데이터 결합

| 함수 | 방향 | 조건 | 용도 |
|------|------|------|------|
| **cbind** | 열(옆) | 행 개수 동일 | 변수 추가 |
| **rbind** | 행(아래) | 열 개수/이름 동일 | 관측치 추가 |
| **merge** | 키 기반 | 공통 키 필요 | 테이블 조인 |

#### Apply 계열

| 함수 | 입력 | 출력 | 주요 용도 |
|------|------|------|----------|
| **apply** | 행렬/배열 | 벡터/행렬 | 행/열 연산 |
| **lapply** | 리스트/벡터 | **리스트** | 리스트 처리 |
| **sapply** | 리스트/벡터 | **벡터/행렬** | 간단한 반복 |
| **tapply** | 벡터 + 그룹 | 벡터/배열 | 그룹별 집계 |

---

### 빅데이터분석기사 예상 문제

#### 문제 1: cbind vs rbind

**다음 R 코드의 결과로 옳은 것은?**

```r
df1 <- data.frame(x = 1:3, y = 4:6)
df2 <- data.frame(x = 7:9, y = 10:12)
result <- rbind(df1, df2)
nrow(result)
```

① 2  
② 3  
③ 4  
④ 6

**정답: ④**
- rbind는 행을 아래로 붙임
- df1: 3행, df2: 3행 → 총 6행

---

#### 문제 2: apply vs tapply

**데이터프레임 df에서 성별(gender)에 따른 평균 나이(age)를 구하려고 한다. 적절한 함수는?**

```r
df <- data.frame(
  gender = c("M", "F", "M", "F"),
  age = c(25, 30, 28, 32)
)
```

① `apply(df$age, df$gender, mean)`  
② `tapply(df$age, df$gender, mean)`  
③ `sapply(df$age, mean)`  
④ `lapply(df, mean)`

**정답: ②**
- tapply는 그룹별 집계에 사용
- `tapply(값, 그룹, 함수)` 형식

---

#### 문제 3: sapply vs lapply

**다음 중 실행 결과가 다른 것은?**

```r
x <- list(a = 1:3, b = 4:6)
```

① `lapply(x, sum)`는 리스트를 반환한다.  
② `sapply(x, sum)`는 벡터를 반환한다.  
③ `lapply(x, sum)`와 `sapply(x, sum)`의 계산 결과는 같다.  
④ `sapply(x, sum)`는 항상 리스트를 반환한다.

**정답: ④**
- sapply는 가능하면 벡터/행렬로 단순화
- lapply는 항상 리스트 반환

---

#### 문제 4: merge

**두 데이터프레임을 병합할 때 df1의 모든 행을 유지하려면?**

```r
df1 <- data.frame(id = 1:3, name = c("A", "B", "C"))
df2 <- data.frame(id = c(1, 3), age = c(20, 30))
```

① `merge(df1, df2, by = "id")`  
② `merge(df1, df2, by = "id", all.x = TRUE)`  
③ `merge(df1, df2, by = "id", all.y = TRUE)`  
④ `rbind(df1, df2)`

**정답: ②**
- `all.x = TRUE`: LEFT JOIN (df1 모든 행 유지)
- 결과에 id=2인 행도 포함 (age는 NA)

---

### 실전 예제

#### 예제 1: 데이터 전처리

```r
# 데이터 생성
sales <- data.frame(
  store = c("A", "B", "A", "B", "A"),
  product = c("X", "X", "Y", "Y", "X"),
  amount = c(100, 150, 200, 250, 120)
)

# 1. 매장별 총 매출
tapply(sales$amount, sales$store, sum)
#   A   B 
# 420 400

# 2. 제품별 평균 매출
tapply(sales$amount, sales$product, mean)
#     X     Y 
# 123.3 225.0

# 3. 행렬로 변환 후 행 합계
mat <- matrix(sales$amount, nrow = 5)
apply(mat, 1, sum)
```

#### 예제 2: 데이터 병합

```r
# 고객 정보
customers <- data.frame(
  customer_id = c(1, 2, 3),
  name = c("Alice", "Bob", "Carol")
)

# 주문 정보
orders <- data.frame(
  customer_id = c(1, 1, 2, 4),
  order_amount = c(100, 150, 200, 250)
)

# 고객별 총 주문액
order_sum <- tapply(orders$order_amount, 
                    orders$customer_id, 
                    sum)

# 결과를 데이터프레임으로
order_df <- data.frame(
  customer_id = as.numeric(names(order_sum)),
  total_amount = order_sum
)

# 고객 정보와 병합 (LEFT JOIN)
result <- merge(customers, order_df, 
                by = "customer_id", 
                all.x = TRUE)
#   customer_id  name total_amount
# 1           1 Alice          250
# 2           2   Bob          200
# 3           3 Carol           NA
```

---

### 핵심 정리

**데이터 결합:**
- `cbind`: 열 추가 (옆으로)
- `rbind`: 행 추가 (아래로)
- `merge`: 키 기반 병합 (JOIN)

**반복 처리:**
- `apply`: 행렬의 행/열 연산
- `lapply`: 리스트 반환
- `sapply`: 벡터 반환 (간단)
- `tapply`: 그룹별 집계

**시험 팁:**
- 함수 이름과 반환 타입 구분
- merge의 JOIN 옵션 숙지
- tapply의 그룹 집계 활용

---

## 4. 택배 차량 배치 최적화 분석

택배 차량 배치는 **처방적 분석(Prescriptive Analytics)** 과 **최적화(Optimization)** 기법을 활용합니다.

---

### 1. 분석 유형 분류

#### **처방적 분석 (Prescriptive Analytics)**

```
기술적 분석 (Descriptive)
   ↓ "무슨 일이 일어났는가?"
진단적 분석 (Diagnostic)
   ↓ "왜 일어났는가?"
예측적 분석 (Predictive)
   ↓ "무슨 일이 일어날 것인가?"
처방적 분석 (Prescriptive)
   ↓ "무엇을 해야 하는가?"
   → 택배 차량 배치 최적화
```

**특징:**
- 최적의 의사결정 제시
- 최적화, 시뮬레이션 활용
- 비용 최소화, 효율 극대화

---

### 2. 주요 분석 기법

#### **(1) VRP - Vehicle Routing Problem (차량 경로 문제)**

**정의:**
- 여러 목적지를 최소 비용으로 방문하는 최적 경로 찾기
- 택배 배송의 핵심 문제

**변형 유형:**

**CVRP (Capacitated VRP)**
```
제약조건:
- 차량 적재 용량 제한
- 각 고객 배송량

목표:
- 총 이동 거리 최소화
- 차량 대수 최소화
```

**VRPTW (VRP with Time Windows)**
```
추가 제약:
- 배송 시간대 제한
- "오전 9시~12시 배송"

현실적 택배 문제
```

**MDVRP (Multi-Depot VRP)**
```
여러 물류 센터 고려
- 각 차량의 출발지 선택
- 센터 간 업무 분배
```

#### **(2) 최적화 알고리즘**

**정수계획법 (Integer Programming)**
```python
# 예시 (개념적)
목적함수:
  minimize: ∑ (거리 × 비용) + 차량 고정비용

제약조건:
  - 각 고객은 정확히 1번 방문
  - 차량 용량 초과 금지
  - 시간 제약 충족
```

**휴리스틱 기법:**
- **유전 알고리즘 (Genetic Algorithm)**
- **시뮬레이티드 어닐링 (Simulated Annealing)**
- **타부 서치 (Tabu Search)**
- **개미 군집 최적화 (Ant Colony Optimization)**

**그리디 알고리즘:**
- **최근접 이웃법 (Nearest Neighbor)**
- **절약 알고리즘 (Savings Algorithm)**

#### **(3) 시뮬레이션**

**이산 사건 시뮬레이션 (Discrete Event Simulation)**

```
시나리오 테스트:
1. 차량 5대 vs 7대 vs 10대
2. 배송 시간대별 수요 변동
3. 교통 혼잡 영향

평가 지표:
- Average Delivery Time
- Vehicle Utilization
- Total Cost
- Customer Satisfaction
```

**몬테카를로 시뮬레이션**
```
불확실성 고려:
- 교통 상황 변동
- 배송 시간 변동
- 수요량 변동

여러 시나리오 반복 실험
```

#### **(4) 예측 분석**

**수요 예측:**
```python
# 시계열 예측
- 시간대별 주문량 예측
- 요일별, 계절별 패턴
- ARIMA, Prophet, LSTM

# 공간 분석
- 지역별 배송 밀집도
- 핫스팟 분석
```

**배송 시간 예측:**
```python
# 머신러닝 모델
- Random Forest
- XGBoost
- Neural Networks

특성:
- 거리, 교통량, 시간대
- 날씨, 요일
- 과거 배송 이력
```

---

### 3. 분석 프로세스

```
1단계: 데이터 수집
   - 주문 정보 (위치, 시간, 물량)
   - 차량 정보 (용량, 속도, 비용)
   - 교통 정보 (거리, 시간)
   ↓
2단계: 수요 예측
   - 시간대별 주문량 예측
   - 지역별 배송 밀도 분석
   ↓
3단계: 최적화 모델링
   - 목적함수 정의 (비용 최소화)
   - 제약조건 설정
   ↓
4단계: 최적화 실행
   - VRP 알고리즘 적용
   - 차량 경로 생성
   ↓
5단계: 시뮬레이션
   - 다양한 시나리오 테스트
   - 성능 지표 평가
   ↓
6단계: 실행 및 모니터링
   - 실시간 경로 조정
   - 성과 측정
```

---

### 4. 실제 활용 예시

#### **CJ대한통운 사례 (가상)**

```python
# 1. 데이터 분석
주문 데이터:
- 일일 평균 10,000건
- 지역: 서울 25개 구
- 시간대: 오전/오후 배송

# 2. 클러스터링
from sklearn.cluster import KMeans

# 배송지를 지역으로 클러스터링
coords = [[lat1, lon1], [lat2, lon2], ...]
kmeans = KMeans(n_clusters=5)  # 5개 권역
clusters = kmeans.fit_predict(coords)

# 3. 경로 최적화
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Google OR-Tools를 이용한 VRP
manager = pywrapcp.RoutingIndexManager(
    len(locations),  # 배송지 수
    num_vehicles,    # 차량 수
    depot            # 물류센터
)

routing = pywrapcp.RoutingModel(manager)

# 거리 콜백 함수
def distance_callback(from_index, to_index):
    return distance_matrix[from_index][to_index]

transit_callback_index = routing.RegisterTransitCallback(
    distance_callback
)

routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# 차량 용량 제약
def demand_callback(from_index):
    return demands[from_index]

demand_callback_index = routing.RegisterUnaryTransitCallback(
    demand_callback
)

routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,  # null capacity slack
    vehicle_capacities,  # 차량 용량
    True,  # start cumul to zero
    'Capacity'
)

# 최적화 실행
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

solution = routing.SolveWithParameters(search_parameters)
```

#### **성과 지표**

```
최적화 전:
- 차량 대수: 15대
- 평균 주행거리: 120km/대
- 총 비용: 3,000,000원/일

최적화 후:
- 차량 대수: 12대 (-20%)
- 평균 주행거리: 95km/대 (-21%)
- 총 비용: 2,280,000원/일 (-24%)

추가 효과:
- CO2 배출 감소
- 배송 시간 단축
- 고객 만족도 향상
```

---

### 5. 빅데이터분석기사 출제 예상

#### 문제 1: 분석 유형

**택배 차량을 효율적으로 배치하여 비용을 최소화하는 문제는 어떤 분석 유형인가?**

① 기술적 분석 (Descriptive Analytics)  
② 진단적 분석 (Diagnostic Analytics)  
③ 예측적 분석 (Predictive Analytics)  
④ 처방적 분석 (Prescriptive Analytics)

**정답: ④**
- 최적의 의사결정을 제시하는 처방적 분석
- 최적화 기법 활용

---

#### 문제 2: 최적화 기법

**차량 경로 문제(VRP)에서 고려해야 할 제약조건이 아닌 것은?**

① 차량의 적재 용량  
② 배송 시간대 제약  
③ 각 고객의 1회 방문  
④ 과거 배송 실적 데이터

**정답: ④**
- ①②③는 VRP의 제약조건
- ④는 예측 모델의 입력 데이터

---

#### 문제 3: 평가 지표

**차량 배치 시뮬레이션에서 사용되는 성능 지표는?**

① Accuracy, Precision, Recall  
② Throughput, Utilization, Waiting Time  
③ R², RMSE, MAE  
④ Lift, ROC-AUC, F1-Score

**정답: ②**
- 시뮬레이션 성능 지표
- ①④는 분류 모델, ③은 회귀 모델 지표

---

### 6. 관련 기술

#### **실시간 최적화**

```
IoT + 실시간 데이터:
- GPS 위치 추적
- 교통 상황 실시간 반영
- 동적 경로 재조정

기술:
- 스트림 데이터 처리 (Kafka, Spark Streaming)
- 실시간 최적화 알고리즘
- 모바일 앱 연동
```

#### **머신러닝 통합**

```
예측 + 최적화:
1. 수요 예측 (ML)
   ↓
2. 차량 대수 결정
   ↓
3. 경로 최적화 (OR)
   ↓
4. 실시간 조정 (Reinforcement Learning)
```

---

### 핵심 정리

**분석 유형:**
- ✅ **처방적 분석 (Prescriptive Analytics)**
- 최적의 의사결정 제시

**주요 기법:**
- ✅ **VRP (차량 경로 문제)** - 핵심
- ✅ **최적화 알고리즘** (정수계획법, 휴리스틱)
- ✅ **시뮬레이션** (성능 평가)
- ✅ **예측 분석** (수요 예측)

**평가 지표:**
- Total Cost (총 비용)
- Vehicle Utilization (차량 가동률)
- Average Delivery Time (평균 배송 시간)
- Customer Satisfaction (고객 만족도)

**시험 포인트:**
- 처방적 분석의 개념
- VRP와 제약조건
- 최적화 vs 예측 vs 시뮬레이션 구분

택배 차량 배치는 **처방적 분석 + VRP + 최적화 알고리즘**의 조합으로 해결합니다!

---

## 5. PCA (주성분 분석)

`princomp()` 함수는 **주성분 분석 (Principal Component Analysis, PCA)** 을 수행하는 R 함수입니다.

---

### 1. 주성분 분석 (PCA) 개념

#### 목적
- **차원 축소 (Dimensionality Reduction)**
- 다수의 변수를 소수의 주성분으로 요약
- 정보 손실 최소화하며 변수 개수 감소

#### 원리
```
원본 데이터:
10개 변수 → 상관관계 존재

PCA 적용:
2~3개 주성분으로 전체 정보의 80~90% 설명
```

**장점:**
- 다중공선성 제거
- 시각화 용이 (2D, 3D)
- 계산 효율성 향상
- 노이즈 제거

---

### 2. princomp() 함수

#### 기본 문법

```r
fit <- princomp(college_s)

# 주요 인자
princomp(x,                # 데이터 (수치형)
         cor = FALSE,      # TRUE: 상관행렬, FALSE: 공분산행렬
         scores = TRUE,    # 주성분 점수 계산
         covmat = NULL)    # 공분산행렬 직접 입력
```

#### 데이터 전처리 필요

```r
# PCA는 스케일에 민감함
# 표준화 필수!

# 방법 1: scale() 함수
college_s <- scale(college)

# 방법 2: cor = TRUE 옵션
fit <- princomp(college, cor = TRUE)  # 상관행렬 사용
```

---

### 3. 결과 해석

#### **(1) summary() - 분산 설명력**

```r
summary(fit)

# 출력 예시:
# Importance of components:
#                           Comp.1    Comp.2    Comp.3
# Standard deviation     2.1234567 1.5678901 0.9876543
# Proportion of Variance 0.4500000 0.2500000 0.1000000
# Cumulative Proportion  0.4500000 0.7000000 0.8000000
```

**해석:**
- **Standard deviation**: 각 주성분의 표준편차
- **Proportion of Variance**: 설명하는 분산 비율
  - Comp.1: 전체 분산의 45% 설명
  - Comp.2: 25% 추가 설명
- **Cumulative Proportion**: 누적 설명력
  - Comp.1~2: 70% 설명
  - Comp.1~3: 80% 설명

#### **(2) loadings() - 주성분 적재량**

```r
fit$loadings
# 또는
loadings(fit)

# 출력 예시:
#          Comp.1  Comp.2  Comp.3
# Var1      0.45    0.23   -0.12
# Var2      0.43    0.15    0.08
# Var3      0.41   -0.31    0.25
# Var4      0.38    0.42   -0.18
# ...
```

**해석:**
- 각 원본 변수가 주성분에 기여하는 정도
- 절댓값이 클수록 해당 주성분에 큰 영향
- Comp.1: Var1, Var2가 주요 영향

#### **(3) scores - 주성분 점수**

```r
fit$scores

# 각 관측치의 주성분 값
#      Comp.1  Comp.2  Comp.3
# [1,]   1.23   -0.45    0.12
# [2,]  -0.87    1.34   -0.23
# [3,]   2.15    0.67    0.45
# ...
```

**활용:**
- 회귀분석 입력변수로 사용
- 클러스터링 입력 데이터
- 시각화

#### **(4) screeplot() - 스크리 플롯**

```r
screeplot(fit, type = "lines")
# 또는
plot(fit, type = "lines")
```

**목적:**
- 주성분 개수 결정
- Elbow Point 찾기

```
분산 │     ●
    │      ╲
    │       ●
    │        ╲___● ● ● ● ●
    └──────────────────────
         1  2  3  4  5  6
           주성분 번호
           
Elbow: 2~3번째
→ 2~3개 주성분 선택
```

---

### 4. 실전 예제

#### 전체 코드

```r
# 1. 데이터 로드
data(USArrests)
head(USArrests)
#            Murder Assault UrbanPop Rape
# Alabama      13.2     236       58 21.2
# Alaska       10.0     263       48 44.5
# ...

# 2. 표준화
usa_s <- scale(USArrests)

# 3. PCA 수행
fit <- princomp(usa_s)

# 4. 결과 확인
summary(fit)
# Importance of components:
#                           Comp.1    Comp.2    Comp.3
# Standard deviation     1.5748783 0.9948694 0.5971291
# Proportion of Variance 0.6200604 0.2474413 0.0891408
# Cumulative Proportion  0.6200604 0.8675017 0.9566425

# PC1~2로 전체 분산의 86.8% 설명!

# 5. 적재량 확인
fit$loadings
# Loadings:
#          Comp.1 Comp.2 Comp.3 Comp.4
# Murder    0.536  0.418  0.341  0.649
# Assault   0.583  0.188  0.268 -0.743
# UrbanPop  0.278 -0.873  0.378  0.134
# Rape      0.543 -0.167 -0.818  0.089

# 6. 시각화
# Scree plot
screeplot(fit, type = "lines", main = "Scree Plot")

# Biplot (관측치 + 변수)
biplot(fit, cex = 0.7)
```

#### 주성분 점수로 회귀분석

```r
# PCA 수행
college_s <- scale(college[, -1])  # 종속변수 제외
fit <- princomp(college_s)

# 처음 3개 주성분만 사용
pc_scores <- fit$scores[, 1:3]

# 회귀분석
y <- college$Grad.Rate  # 종속변수
model <- lm(y ~ pc_scores)

summary(model)
```

---

### 5. princomp vs prcomp

#### 두 함수 비교

| 항목 | princomp | prcomp |
|------|----------|--------|
| **방법** | 고유값 분해 | 특이값 분해 (SVD) |
| **정확도** | 보통 | 높음 (수치적 안정) |
| **속도** | 느림 | 빠름 |
| **권장** | 작은 데이터 | **큰 데이터** |

#### prcomp 사용

```r
# 권장 방법
fit <- prcomp(college, scale. = TRUE)  # 자동 표준화

summary(fit)
fit$rotation  # loadings와 동일
fit$x         # scores와 동일
```

---

### 6. cor 옵션의 의미

#### **cor = TRUE**
```r
princomp(data, cor = TRUE)
```
- **상관행렬(Correlation Matrix)** 기반 PCA
- 데이터를 자동으로 표준화하여 계산
- 변수들의 스케일이 다를 때 사용

#### **cor = FALSE** (기본값)
```r
princomp(data, cor = FALSE)
```
- **공분산행렬(Covariance Matrix)** 기반 PCA
- 원본 데이터 그대로 사용
- 이미 표준화된 데이터에 사용

---

### 7. ⚠️ 주의사항 - 중복 표준화

#### 문제 상황

```r
# 1단계: 표준화
college_s <- scale(college)  # 이미 표준화!

# 2단계: PCA with cor=T
fit <- princomp(college_s, cor=T)  # ❌ 중복 표준화!
```

**무슨 일이 일어나는가?**
```
원본 데이터
    ↓ scale() 적용
표준화된 데이터 (college_s)
    ↓ cor=T로 다시 표준화 시도
중복 표준화 → 결과는 동일하지만 불필요한 연산
```

#### 올바른 사용법

**방법 1: scale() 사용 후 cor=FALSE**
```r
# 권장 방법
college_s <- scale(college)
fit <- princomp(college_s, cor = FALSE)  # ✅ 올바름
```

**방법 2: 원본 데이터에 cor=TRUE**
```r
# 이것도 올바름
fit <- princomp(college, cor = TRUE)  # ✅ 올바름
```

---

### 8. 상관행렬 vs 공분산행렬

#### 예제로 이해하기

```r
# 데이터 생성
df <- data.frame(
  age = c(25, 30, 35, 40, 45),        # 스케일: 20~50
  salary = c(3000, 4000, 5000, 6000, 7000)  # 스케일: 3000~7000
)

# ===== 방법 1: 공분산 행렬 (cor=FALSE) =====
fit1 <- princomp(df, cor = FALSE)
summary(fit1)
# salary가 스케일이 크므로 PC1을 지배함
# 문제: 스케일에 의한 왜곡

# ===== 방법 2: 상관 행렬 (cor=TRUE) =====
fit2 <- princomp(df, cor = TRUE)
summary(fit2)
# 모든 변수를 동등하게 취급
# 권장: 스케일 차이 제거
```

#### 결과 비교

```r
# 공분산 기반 (cor=FALSE)
fit1$loadings
#         Comp.1  Comp.2
# age      0.05    0.99   # salary에 밀려 기여도 낮음
# salary   0.99    0.05   # 스케일이 커서 지배적

# 상관 기반 (cor=TRUE)
fit2$loadings
#         Comp.1  Comp.2
# age      0.71    0.71   # 균형잡힌 기여도
# salary   0.71   -0.71
```

---

### 9. 빅데이터분석기사 예상 문제

#### 문제 1: PCA 개념

**주성분 분석(PCA)에 대한 설명으로 옳지 않은 것은?**

① 차원 축소를 위한 비지도 학습 기법이다.  
② 상관관계가 높은 변수들을 독립적인 주성분으로 변환한다.  
③ 첫 번째 주성분이 가장 많은 분산을 설명한다.  
④ 범주형 변수에 주로 사용된다.

**정답: ④**
- PCA는 **수치형 변수**에만 적용
- 범주형은 MCA(다중대응분석) 사용

---

#### 문제 2: 결과 해석

**PCA 결과가 다음과 같을 때, PC1~PC2로 설명되는 분산 비율은?**

```
Proportion of Variance: 
  PC1: 0.45
  PC2: 0.30
  PC3: 0.15
```

① 45%  
② 75%  
③ 90%  
④ 100%

**정답: ②**
- PC1 + PC2 = 0.45 + 0.30 = 0.75 (75%)

---

#### 문제 3: 전처리

**PCA 수행 전 필수적인 전처리는?**

① 결측치 대체  
② **표준화 (Standardization)**  
③ 로그 변환  
④ 원-핫 인코딩

**정답: ②**
- 변수 스케일이 다르면 결과 왜곡
- scale() 또는 cor=TRUE 필수

---

#### 문제 4: 주성분 선택

**Scree Plot에서 주성분 개수를 결정하는 기준은?**

① 첫 번째 주성분만 선택  
② **Elbow Point 이전까지 선택**  
③ 모든 주성분 선택  
④ 분산이 가장 작은 주성분 선택

**정답: ②**
- Elbow(팔꿈치) 지점에서 기울기 급변
- 그 이후는 설명력 낮음

---

#### 문제 5: cor 옵션

**다음 중 올바른 PCA 수행 방법은?**

```r
# 데이터가 이미 표준화되어 있음
data_scaled <- scale(data)
```

① `princomp(data_scaled, cor = TRUE)`  
② `princomp(data_scaled, cor = FALSE)`  
③ `princomp(data, cor = FALSE)`  
④ `prcomp(data, scale. = FALSE)`

**정답: ②**
- 이미 표준화된 데이터는 cor=FALSE
- ①은 중복 표준화
- ③④는 표준화 안 함

---

### 10. 종합 예제 - 단계별 분석

```r
# ===== 1단계: 데이터 준비 =====
library(MASS)  # Boston 데이터셋
data(Boston)
head(Boston)

# 종속변수(medv) 제외
boston_x <- Boston[, -14]

# ===== 2단계: 표준화 =====
boston_s <- scale(boston_x)

# ===== 3단계: PCA =====
fit <- princomp(boston_s)

# ===== 4단계: 설명력 확인 =====
summary(fit)
# 목표: 누적 80% 이상

# ===== 5단계: 주성분 개수 결정 =====
# Scree plot
screeplot(fit, type = "lines")

# 분산 비율 시각화
var_ratio <- fit$sdev^2 / sum(fit$sdev^2)
plot(var_ratio, type = "b", 
     xlab = "PC", ylab = "Variance Ratio")
abline(h = 0.1, col = "red", lty = 2)  # 10% 기준선

# 누적 분산
cumsum(var_ratio)
# [1] 0.475 0.650 0.750 0.825 ...
# → PC1~4로 82.5% 설명

# ===== 6단계: 적재량 확인 =====
loadings(fit)
# PC1에 크게 기여하는 변수 확인

# ===== 7단계: 주성분 추출 =====
pc_data <- fit$scores[, 1:4]  # 처음 4개만

# ===== 8단계: 회귀분석 =====
df_pc <- data.frame(
  medv = Boston$medv,
  pc_data
)

model <- lm(medv ~ ., data = df_pc)
summary(model)

# ===== 9단계: 시각화 =====
# Biplot
biplot(fit, choices = c(1, 2))

# PC1 vs PC2 산점도
plot(fit$scores[, 1], fit$scores[, 2],
     xlab = "PC1", ylab = "PC2",
     main = "PCA Score Plot")
```

---

### 실전 가이드

#### 상황별 선택

| 상황 | 코드 | 이유 |
|------|------|------|
| **원본 데이터** | `princomp(data, cor=T)` | 자동 표준화 |
| **이미 표준화** | `princomp(data_s, cor=F)` | 중복 방지 |
| **스케일 같음** | `princomp(data, cor=F)` | 공분산 사용 |
| **스케일 다름** | `princomp(data, cor=T)` | 표준화 필수 |

---

### 핵심 정리

**princomp() 함수:**
```r
fit <- princomp(data_scaled)  # 표준화된 데이터
# 또는
fit <- princomp(data, cor = TRUE)  # 상관행렬 사용
```

**주요 결과:**
- `summary(fit)`: 설명력
- `fit$loadings`: 적재량
- `fit$scores`: 주성분 점수
- `screeplot(fit)`: 주성분 개수 결정

**전처리:**
- ✅ 표준화 필수 (`scale()`)
- ✅ 수치형 변수만 사용
- ✅ 결측치 제거

**주성분 선택:**
- 누적 설명력 80~90%
- Scree Plot의 Elbow Point
- 고유값 > 1 (Kaiser 기준)

**cor 옵션 선택 규칙:**
```r
# 원본 데이터 + 스케일 다름
princomp(data, cor = TRUE)  ✅

# 원본 데이터 + 스케일 같음
princomp(data, cor = FALSE)  ✅

# 표준화 데이터
college_s <- scale(college)
princomp(college_s, cor = FALSE)  ✅
princomp(college_s)  ✅ (cor=FALSE가 기본)

# 중복 표준화 (결과는 맞지만 비효율)
princomp(college_s, cor = TRUE)  ⚠️
```

**권장 방법:**
```r
# 가장 간단하고 명확한 방법
fit <- prcomp(college, scale. = TRUE)
```

**시험 팁:**
1. **cor=TRUE**: 자동 표준화, 스케일 다를 때
2. **cor=FALSE**: 이미 표준화됨, 스케일 같을 때
3. **prcomp()**: 더 안정적, scale. 옵션 사용
4. **표준화 중요성**: 거의 모든 PCA에서 필수

PCA는 빅데이터분석기사 실기에서 자주 출제되므로 `princomp()` 사용법과 결과 해석을 확실히 익혀두세요!

---

## 학습 마무리

이 자료는 빅데이터분석기사 시험 대비를 위한 핵심 개념들을 정리한 것입니다.

### 중요 개념 요약

1. **EDA (탐색적 데이터 분석)** - 데이터 이해와 전처리의 출발점
2. **평가 지표** - 데이터 마이닝(Accuracy, Precision, Recall) vs 시뮬레이션(Throughput, Waiting Time)
3. **R 함수** - cbind, rbind, apply 계열, merge
4. **최적화** - 처방적 분석, VRP, 차량 경로 문제
5. **PCA** - 차원 축소, princomp(), 주성분 분석

### 시험 대비 팁

- 각 기법의 목적과 활용 사례 이해
- 함수의 입출력 형태 숙지
- 평가 지표 계산 방법 암기
- 실제 코드 작성 연습
- 결과 해석 능력 향상
