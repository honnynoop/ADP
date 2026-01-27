# ANOVA F-검정

ANOVA(Analysis of Variance)의 F-검정은 **3개 이상의 그룹 간 평균 차이**를 검정하는 방법입니다.

## 📊 기본 개념

### 가설 설정
- **H₀ (귀무가설)**: μ₁ = μ₂ = μ₃ = ... = μₖ (모든 그룹의 평균이 같다)
- **H₁ (대립가설)**: 적어도 하나의 그룹 평균이 다르다

### F-통계량

F = **집단 간 분산(MSB)** / **집단 내 분산(MSW)**

- **MSB (Mean Square Between)**: 그룹 간 변동
- **MSW (Mean Square Within)**: 그룹 내 변동

## 🧮 계산 과정

### 1. 총 제곱합 분해 (SST)

**SST** = **SSB** + **SSW**

- **SST** (Total Sum of Squares): 전체 변동
- **SSB** (Between Sum of Squares): 집단 간 변동
- **SSW** (Within Sum of Squares): 집단 내 변동

### 2. 제곱합 계산

```
SST = Σ(xᵢⱼ - x̄)²
SSB = Σnⱼ(x̄ⱼ - x̄)²
SSW = ΣΣ(xᵢⱼ - x̄ⱼ)²
```

### 3. 평균 제곱 계산

```
MSB = SSB / (k-1)
MSW = SSW / (n-k)
```

- k: 그룹 수
- n: 전체 샘플 수

### 4. F-통계량

```
F = MSB / MSW
```

자유도: df₁ = k-1, df₂ = n-k

## 💻 Python 예제

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 예제 데이터: 3개 학습 방법의 시험 점수
group_A = [85, 88, 90, 87, 86]  # 방법 A
group_B = [78, 82, 80, 81, 79]  # 방법 B
group_C = [92, 95, 93, 94, 91]  # 방법 C

# 방법 1: scipy.stats.f_oneway
f_stat, p_value = stats.f_oneway(group_A, group_B, group_C)
print(f"F-통계량: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# 방법 2: 수동 계산으로 이해하기
data = pd.DataFrame({
    'score': group_A + group_B + group_C,
    'group': ['A']*5 + ['B']*5 + ['C']*5
})

# 전체 평균
grand_mean = data['score'].mean()

# 각 그룹 평균
group_means = data.groupby('group')['score'].mean()
print("\n그룹별 평균:")
print(group_means)

# SSB (집단 간 제곱합)
ssb = sum(5 * (group_means - grand_mean)**2)  # 각 그룹 n=5

# SSW (집단 내 제곱합)
ssw = sum((data[data['group']=='A']['score'] - group_means['A'])**2) + \
      sum((data[data['group']=='B']['score'] - group_means['B'])**2) + \
      sum((data[data['group']=='C']['score'] - group_means['C'])**2)

# SST (총 제곱합)
sst = sum((data['score'] - grand_mean)**2)

print(f"\nSST: {sst:.4f}")
print(f"SSB: {ssb:.4f}")
print(f"SSW: {ssw:.4f}")
print(f"SST = SSB + SSW: {ssb + ssw:.4f}")

# 자유도
k = 3  # 그룹 수
n = 15  # 전체 샘플 수
df_between = k - 1
df_within = n - k

# 평균 제곱
msb = ssb / df_between
msw = ssw / df_within

# F-통계량
f_manual = msb / msw

print(f"\nMSB: {msb:.4f}")
print(f"MSW: {msw:.4f}")
print(f"F-통계량: {f_manual:.4f}")

# 임계값과 비교
alpha = 0.05
critical_value = stats.f.ppf(1-alpha, df_between, df_within)
print(f"\n유의수준 {alpha}에서 임계값: {critical_value:.4f}")
print(f"결론: {'귀무가설 기각' if f_manual > critical_value else '귀무가설 채택'}")
```

## 📈 ANOVA 테이블

| 변동 원인 | 제곱합(SS) | 자유도(df) | 평균제곱(MS) | F-통계량 | p-value |
|---------|-----------|-----------|-------------|---------|---------|
| 집단 간 | SSB | k-1 | MSB | F | p |
| 집단 내 | SSW | n-k | MSW | - | - |
| 전체 | SST | n-1 | - | - | - |

## 🔍 사후 검정 (Post-hoc Test)

ANOVA에서 귀무가설이 기각되면, **어느 그룹 간에 차이가 있는지** 확인:

```python
from scipy.stats import tukey_hsd

# Tukey HSD 검정
res = tukey_hsd(group_A, group_B, group_C)
print("\nTukey HSD 결과:")
print(res)

# 또는 statsmodels 사용
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(
    endog=data['score'],
    groups=data['group'],
    alpha=0.05
)
print(tukey)
```

## 🎯 실무 적용 시 주의사항

### 가정 사항
1. **정규성**: 각 그룹이 정규분포를 따름
2. **등분산성**: 각 그룹의 분산이 동일
3. **독립성**: 각 관측치는 독립적

### 가정 검정

```python
# 1. 정규성 검정 (Shapiro-Wilk)
for group_name, group_data in [('A', group_A), ('B', group_B), ('C', group_C)]:
    stat, p = stats.shapiro(group_data)
    print(f"{group_name} 정규성 p-value: {p:.4f}")

# 2. 등분산성 검정 (Levene's test)
stat, p = stats.levene(group_A, group_B, group_C)
print(f"\nLevene 검정 p-value: {p:.4f}")
```

### 대안 방법

- **등분산성 위배**: Welch's ANOVA
- **정규성 위배**: Kruskal-Wallis 검정 (비모수 검정)

```python
# Welch's ANOVA (등분산 가정 불필요)
from scipy.stats import alexandergovern
stat, p = alexandergovern(group_A, group_B, group_C)

# Kruskal-Wallis (비모수)
stat, p = stats.kruskal(group_A, group_B, group_C)
```

## 📝 해석 가이드

1. **p-value < 0.05**: 적어도 한 그룹의 평균이 유의하게 다름
2. **p-value ≥ 0.05**: 그룹 간 평균 차이가 유의하지 않음
3. **F-통계량이 클수록**: 그룹 간 차이가 그룹 내 변동에 비해 큼

## 💡 시험 대비 핵심 포인트

### 1. 제곱합 분해 이해
- SST = SSB + SSW 관계 기억
- 각 제곱합의 의미 정확히 이해

### 2. F-통계량 계산
- F = MSB / MSW
- 자유도 계산: df₁ = k-1, df₂ = n-k

### 3. ANOVA 테이블 작성
- 변동 원인별로 SS, df, MS 계산
- F값과 p-value 해석

### 4. 사후 검정
- ANOVA는 "차이가 있다"만 알려줌
- "어디에 차이가 있는지"는 사후 검정 필요

### 5. 가정 확인
- 정규성, 등분산성, 독립성
- 가정 위배 시 대안 방법 선택

시험에서 자주 출제되는 포인트이니 제곱합 분해와 F-통계량 계산을 확실히 이해하세요! 📚
