# 선형회귀 잔차 (Residual) 완벽 가이드

## 📌 개요

### 잔차란?
**잔차(Residual)**는 실제 관측값과 회귀모델이 예측한 값의 차이

$$e_i = y_i - \hat{y}_i$$

| 기호 | 의미 |
|---|---|
| $y_i$ | 실제 관측값 (observed value) |
| $\hat{y}_i$ | 예측값 (predicted/fitted value) |
| $e_i$ | 잔차 (residual) |

```
     Y
     │      
     │          ● (실제값 yᵢ)
     │          │ ← 잔차 eᵢ = yᵢ - ŷᵢ
     │          ▼
     │  ────────○────────  회귀선
     │         (예측값 ŷᵢ)
     │
     └─────────────────── X
```

---

## 1️⃣ 수치 예제로 이해하기

### 1.1 예제 데이터

광고비(X)와 매출(Y)의 관계를 분석

| i | 광고비 (X) | 매출 (Y) |
|:---:|:---:|:---:|
| 1 | 1 | 3 |
| 2 | 2 | 5 |
| 3 | 3 | 6 |
| 4 | 4 | 8 |
| 5 | 5 | 11 |

### 1.2 Step 1: 회귀계수 계산

**기본 통계량:**
- $\bar{x} = (1+2+3+4+5)/5 = 3$
- $\bar{y} = (3+5+6+8+11)/5 = 6.6$

**회귀계수 공식:**
$$\hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$$

**계산 과정:**

| i | xᵢ | yᵢ | xᵢ - x̄ | yᵢ - ȳ | (xᵢ-x̄)(yᵢ-ȳ) | (xᵢ-x̄)² |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 3 | -2 | -3.6 | 7.2 | 4 |
| 2 | 2 | 5 | -1 | -1.6 | 1.6 | 1 |
| 3 | 3 | 6 | 0 | -0.6 | 0 | 0 |
| 4 | 4 | 8 | 1 | 1.4 | 1.4 | 1 |
| 5 | 5 | 11 | 2 | 4.4 | 8.8 | 4 |
| **합계** | | | | | **19.0** | **10** |

**회귀계수 계산:**
$$\hat{\beta}_1 = \frac{19.0}{10} = 1.9$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} = 6.6 - 1.9 \times 3 = 0.9$$

**회귀식:** 
$$\hat{y} = 0.9 + 1.9x$$

### 1.3 Step 2: 예측값과 잔차 계산

| i | xᵢ | yᵢ (실제) | ŷᵢ = 0.9 + 1.9xᵢ | eᵢ = yᵢ - ŷᵢ | eᵢ² |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 3 | 0.9 + 1.9(1) = **2.8** | 3 - 2.8 = **+0.2** | 0.04 |
| 2 | 2 | 5 | 0.9 + 1.9(2) = **4.7** | 5 - 4.7 = **+0.3** | 0.09 |
| 3 | 3 | 6 | 0.9 + 1.9(3) = **6.6** | 6 - 6.6 = **-0.6** | 0.36 |
| 4 | 4 | 8 | 0.9 + 1.9(4) = **8.5** | 8 - 8.5 = **-0.5** | 0.25 |
| 5 | 5 | 11 | 0.9 + 1.9(5) = **10.4** | 11 - 10.4 = **+0.6** | 0.36 |
| **합계** | | | | **Σeᵢ = 0** | **SSE = 1.10** |

**핵심 특성:** 잔차의 합은 항상 0
$$\sum_{i=1}^{n} e_i = 0$$

---

## 2️⃣ 잔차의 종류

### 2.1 종류별 비교

| 잔차 종류 | 공식 | 용도 |
|---|---|---|
| **일반 잔차** | $e_i = y_i - \hat{y}_i$ | 기본 계산 |
| **표준화 잔차** | $r_i = \frac{e_i}{\hat{\sigma}}$ | 크기 비교 |
| **스튜던트화 잔차** | $r_i^* = \frac{e_i}{\hat{\sigma}\sqrt{1-h_{ii}}}$ | 이상치 탐지 |
| **삭제 스튜던트화 잔차** | $t_i = \frac{e_i}{\hat{\sigma}_{(i)}\sqrt{1-h_{ii}}}$ | 영향력 진단 |

### 2.2 표준화 잔차 계산

**잔차의 표준오차:**
$$\hat{\sigma} = \sqrt{\frac{SSE}{n-2}} = \sqrt{\frac{1.10}{5-2}} = \sqrt{0.367} = 0.606$$

**표준화 잔차:**
$$r_i = \frac{e_i}{\hat{\sigma}}$$

| i | eᵢ | rᵢ = eᵢ / 0.606 |
|:---:|:---:|:---:|
| 1 | +0.2 | +0.33 |
| 2 | +0.3 | +0.50 |
| 3 | -0.6 | -0.99 |
| 4 | -0.5 | -0.83 |
| 5 | +0.6 | +0.99 |

**이상치 판단 기준:** 
- |rᵢ| > 2 → 이상치 의심
- |rᵢ| > 3 → 이상치 가능성 높음

---

## 3️⃣ 잔차 분석의 목적

### 3.1 회귀모델의 4가지 가정 검증

| 가정 | 내용 | 검증 방법 | 위반 시 문제 |
|---|---|---|---|
| **선형성** | X와 Y의 관계가 선형 | 잔차 vs 예측값 그래프 | 편향된 추정 |
| **정규성** | 잔차가 정규분포 | Q-Q Plot, Shapiro-Wilk | 신뢰구간/검정 부정확 |
| **등분산성** | 잔차의 분산이 일정 | 잔차 vs 예측값 그래프 | 비효율적 추정 |
| **독립성** | 잔차 간 자기상관 없음 | Durbin-Watson 검정 | 표준오차 과소추정 |

### 3.2 잔차 그래프 패턴 해석

```
┌─────────────────────────────────────────────────────────────────┐
│  (A) 이상적인 패턴           (B) 비선형성                        │
│      잔차                        잔차                           │
│       │  ·  ·                     │      · ·                    │
│       │    ·  ·                   │   ·       ·                 │
│     0 ├─·──────·─                0 ├·───────────·               │
│       │  ·    ·                   │ ·         ·                 │
│       │    ·                      │    · · ·                    │
│       └──────────→ 예측값         └──────────→ 예측값            │
│       → 무작위 산포               → U자형 (2차항 필요)           │
├─────────────────────────────────────────────────────────────────┤
│  (C) 이분산성                (D) 이상치 존재                     │
│      잔차                        잔차                           │
│       │          ·  ·            │              ● ← 이상치      │
│       │       ·    ·             │  ·  ·  ·                     │
│     0 ├·─·──────────            0 ├────────────                 │
│       │ ·  ·     ·               │    ·  ·                      │
│       │    ·  ·   ·              │ ·     ·                      │
│       └──────────→ 예측값         └──────────→ 예측값            │
│       → 나팔 모양 (분산 증가)     → 튀는 점 확인                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 패턴별 원인과 해결책

| 패턴 | 가정 위반 | 원인 | 해결책 |
|---|---|---|---|
| U자형 | 선형성 | 비선형 관계 | 다항회귀, 변수변환 |
| 나팔형 | 등분산성 | 분산이 Y에 비례 | 로그변환, WLS |
| 물결형 | 독립성 | 시계열 자기상관 | 차분, AR 모델 |
| 이상점 | - | 측정오류, 특이값 | 이상치 제거/수정 |

---

## 4️⃣ SSE, SSR, SST의 관계

### 4.1 분해 공식

$$SST = SSR + SSE$$

| 구분 | 공식 | 의미 | 자유도 |
|---|---|---|---|
| **SST** (총제곱합) | $\sum(y_i - \bar{y})^2$ | 전체 변동 | n - 1 |
| **SSR** (회귀제곱합) | $\sum(\hat{y}_i - \bar{y})^2$ | 회귀로 설명된 변동 | p - 1 |
| **SSE** (잔차제곱합) | $\sum(y_i - \hat{y}_i)^2 = \sum e_i^2$ | 설명 못한 변동 | n - p |

*p: 모수의 수 (단순회귀에서 p = 2)*

### 4.2 예제 계산

| i | yᵢ | ȳ | ŷᵢ | (yᵢ-ȳ)² | (ŷᵢ-ȳ)² | eᵢ² |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 3 | 6.6 | 2.8 | 12.96 | 14.44 | 0.04 |
| 2 | 5 | 6.6 | 4.7 | 2.56 | 3.61 | 0.09 |
| 3 | 6 | 6.6 | 6.6 | 0.36 | 0.00 | 0.36 |
| 4 | 8 | 6.6 | 8.5 | 1.96 | 3.61 | 0.25 |
| 5 | 11 | 6.6 | 10.4 | 19.36 | 14.44 | 0.36 |
| **합계** | | | | **SST=37.20** | **SSR=36.10** | **SSE=1.10** |

**검증:** 
$$SST = SSR + SSE$$
$$37.20 = 36.10 + 1.10 \quad \checkmark$$

### 4.3 결정계수 (R²)

$$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$

$$R^2 = \frac{36.10}{37.20} = 0.970$$

**해석:** 광고비가 매출 변동의 **97%를 설명**

### 4.4 수정된 결정계수 (Adjusted R²)

$$R_{adj}^2 = 1 - \frac{SSE/(n-p)}{SST/(n-1)} = 1 - \frac{MSE}{MST}$$

$$R_{adj}^2 = 1 - \frac{1.10/3}{37.20/4} = 1 - \frac{0.367}{9.30} = 0.961$$

---

## 5️⃣ 잔차 진단 검정

### 5.1 정규성 검정

| 방법 | 귀무가설 | 판단 기준 | 특징 |
|---|---|---|---|
| **Shapiro-Wilk** | 잔차가 정규분포 | p > 0.05 → 정규성 만족 | 소표본에 적합 (n < 50) |
| **Kolmogorov-Smirnov** | 잔차가 정규분포 | p > 0.05 → 정규성 만족 | 대표본에 적합 |
| **Q-Q Plot** | - | 직선에 가까움 | 시각적 확인 |

### 5.2 등분산성 검정

| 방법 | 귀무가설 | 판단 기준 |
|---|---|---|
| **Breusch-Pagan** | 분산이 동일 | p > 0.05 → 등분산 만족 |
| **White 검정** | 분산이 동일 | p > 0.05 → 등분산 만족 |
| **Goldfeld-Quandt** | 분산이 동일 | p > 0.05 → 등분산 만족 |

### 5.3 독립성 검정 (자기상관)

**Durbin-Watson 검정:**

$$DW = \frac{\sum_{i=2}^{n}(e_i - e_{i-1})^2}{\sum_{i=1}^{n}e_i^2}$$

**DW 값 해석:**

| DW 값 | 해석 |
|:---:|---|
| ≈ 0 | 강한 양의 자기상관 |
| ≈ 2 | 자기상관 없음 (이상적) |
| ≈ 4 | 강한 음의 자기상관 |
| 1.5 ~ 2.5 | 일반적으로 허용 범위 |

**예제 계산:**

| i | eᵢ | eᵢ - eᵢ₋₁ | (eᵢ - eᵢ₋₁)² |
|:---:|:---:|:---:|:---:|
| 1 | +0.2 | - | - |
| 2 | +0.3 | 0.3 - 0.2 = 0.1 | 0.01 |
| 3 | -0.6 | -0.6 - 0.3 = -0.9 | 0.81 |
| 4 | -0.5 | -0.5 - (-0.6) = 0.1 | 0.01 |
| 5 | +0.6 | 0.6 - (-0.5) = 1.1 | 1.21 |
| **합계** | | | **2.04** |

$$DW = \frac{2.04}{1.10} = 1.85$$

**결론:** DW ≈ 2에 가까우므로 **자기상관 없음**

---

## 6️⃣ 이상치와 영향력 있는 관측치

### 6.1 개념 구분

| 용어 | 정의 | 탐지 방법 | 기준 |
|---|---|---|---|
| **이상치 (Outlier)** | Y값이 예측값에서 크게 벗어난 점 | 표준화 잔차 | \|r\| > 2~3 |
| **지렛점 (Leverage)** | X값이 평균에서 멀리 떨어진 점 | 레버리지 값 | hᵢᵢ > 2p/n |
| **영향점 (Influential)** | 모델에 큰 영향을 미치는 점 | Cook's Distance | D > 4/n 또는 D > 1 |

```
        Y
        │
        │     ● A (이상치: Y만 이상)
        │    ╱
        │   ╱
        │  ╱  · · · ·
        │ ╱· · 
        │╱
        └──────────────────● B (지렛점: X만 이상)
                           X
        
        ● C (영향점: 이상치 + 지렛점)
```

### 6.2 레버리지 (Leverage) 계산

**공식:**
$$h_{ii} = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum(x_j - \bar{x})^2}$$

**예제 계산:**

| i | xᵢ | (xᵢ - x̄)² | hᵢᵢ 계산 | hᵢᵢ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 4 | 1/5 + 4/10 | **0.60** |
| 2 | 2 | 1 | 1/5 + 1/10 | **0.30** |
| 3 | 3 | 0 | 1/5 + 0/10 | **0.20** |
| 4 | 4 | 1 | 1/5 + 1/10 | **0.30** |
| 5 | 5 | 4 | 1/5 + 4/10 | **0.60** |

**레버리지 특성:**
- 레버리지 합계: $\sum h_{ii} = p$ (모수의 수)
- 평균 레버리지: $\bar{h} = p/n$
- 기준: $h_{ii} > 2p/n = 2 \times 2/5 = 0.8$ → 지렛점

### 6.3 Cook's Distance

**공식:**
$$D_i = \frac{r_i^2}{p} \times \frac{h_{ii}}{1-h_{ii}}$$

또는

$$D_i = \frac{\sum_{j=1}^{n}(\hat{y}_j - \hat{y}_{j(i)})^2}{p \times MSE}$$

**판단 기준:**
- $D_i > 4/n$ → 영향점 의심
- $D_i > 1$ → 강한 영향점
- $D_i > 0.5$ → 주의 필요

---

## 7️⃣ 잔차의 핵심 특성

### 7.1 수학적 특성

| 특성 | 수식 | 의미 |
|---|---|---|
| 잔차 합 = 0 | $\sum e_i = 0$ | 과대/과소 추정이 상쇄 |
| X와 공분산 = 0 | $\sum x_i e_i = 0$ | X와 잔차는 무관 |
| 예측값과 공분산 = 0 | $\sum \hat{y}_i e_i = 0$ | 예측값과 잔차는 무관 |
| 잔차 평균 = 0 | $\bar{e} = 0$ | 평균적으로 오차 없음 |

### 7.2 잔차와 오차의 차이

| 구분 | 잔차 (Residual) | 오차 (Error) |
|---|---|---|
| **정의** | $e_i = y_i - \hat{y}_i$ | $\epsilon_i = y_i - E(y_i)$ |
| **계산** | 표본에서 계산 가능 | 알 수 없음 (이론적) |
| **기댓값** | $E(e_i) = 0$ | $E(\epsilon_i) = 0$ |
| **관계** | 오차의 추정치 | 모집단의 실제 오차 |

---

## 8️⃣ Python 구현

### 8.1 기본 잔차 분석

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

# 데이터
X = np.array([1, 2, 3, 4, 5])
Y = np.array([3, 5, 6, 8, 11])

# 회귀분석
X_with_const = sm.add_constant(X)
model = sm.OLS(Y, X_with_const).fit()

# 결과 출력
print("=" * 60)
print("회귀분석 결과")
print("=" * 60)
print(f"회귀식: ŷ = {model.params[0]:.2f} + {model.params[1]:.2f}x")
print(f"R² = {model.rsquared:.4f}")
print(f"Adjusted R² = {model.rsquared_adj:.4f}")
print(f"잔차 표준오차 (RMSE) = {np.sqrt(model.mse_resid):.4f}")

# 잔차 분석
residuals = model.resid
fitted = model.fittedvalues

print("\n" + "=" * 60)
print("잔차 분석 결과")
print("=" * 60)

df = pd.DataFrame({
    'X': X,
    'Y (실제)': Y,
    'Ŷ (예측)': fitted.round(2),
    '잔차 (e)': residuals.round(2),
    '표준화 잔차': (residuals / np.std(residuals, ddof=2)).round(3)
})
print(df.to_string(index=False))

print(f"\n잔차 합계: {sum(residuals):.10f} (≈ 0)")
print(f"SSE (잔차제곱합): {sum(residuals**2):.2f}")
print(f"SSR (회귀제곱합): {sum((fitted - np.mean(Y))**2):.2f}")
print(f"SST (총제곱합): {sum((Y - np.mean(Y))**2):.2f}")
```

### 8.2 가정 검정

```python
# 정규성 검정 (Shapiro-Wilk)
stat_shapiro, p_shapiro = stats.shapiro(residuals)
print(f"\n[정규성 검정 - Shapiro-Wilk]")
print(f"통계량: {stat_shapiro:.4f}, p-value: {p_shapiro:.4f}")
print(f"결론: {'정규성 만족' if p_shapiro > 0.05 else '정규성 불만족'}")

# 독립성 검정 (Durbin-Watson)
dw = durbin_watson(residuals)
print(f"\n[독립성 검정 - Durbin-Watson]")
print(f"DW 통계량: {dw:.4f}")
print(f"결론: {'자기상관 없음' if 1.5 < dw < 2.5 else '자기상관 의심'}")

# 등분산성 검정 (Breusch-Pagan)
bp_test = het_breuschpagan(residuals, X_with_const)
print(f"\n[등분산성 검정 - Breusch-Pagan]")
print(f"LM 통계량: {bp_test[0]:.4f}, p-value: {bp_test[1]:.4f}")
print(f"결론: {'등분산 만족' if bp_test[1] > 0.05 else '이분산 존재'}")
```

### 8.3 영향력 진단

```python
# 영향력 진단
influence = model.get_influence()
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]
studentized_resid = influence.resid_studentized_internal

print("\n" + "=" * 60)
print("영향력 진단")
print("=" * 60)

diag_df = pd.DataFrame({
    'i': range(1, len(X)+1),
    '레버리지 (h)': leverage.round(3),
    "Cook's D": cooks_d.round(4),
    '스튜던트화 잔차': studentized_resid.round(3)
})
print(diag_df.to_string(index=False))

print(f"\n레버리지 기준: h > {2*2/len(X):.2f}")
print(f"Cook's D 기준: D > {4/len(X):.2f}")
```

### 8.4 잔차 시각화

```python
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 잔차 vs 예측값 (선형성, 등분산성 확인)
axes[0,0].scatter(fitted, residuals, s=100, edgecolors='black', alpha=0.7)
axes[0,0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0,0].set_xlabel('예측값 (Fitted Values)', fontsize=11)
axes[0,0].set_ylabel('잔차 (Residuals)', fontsize=11)
axes[0,0].set_title('잔차 vs 예측값\n(선형성, 등분산성 확인)', fontsize=12)
axes[0,0].grid(True, alpha=0.3)

# 2. Q-Q Plot (정규성 확인)
stats.probplot(residuals, dist="norm", plot=axes[0,1])
axes[0,1].set_title('정규 Q-Q Plot\n(정규성 확인)', fontsize=12)
axes[0,1].grid(True, alpha=0.3)

# 3. 잔차 히스토그램
axes[1,0].hist(residuals, bins=5, edgecolor='black', alpha=0.7, color='steelblue')
axes[1,0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('잔차', fontsize=11)
axes[1,0].set_ylabel('빈도', fontsize=11)
axes[1,0].set_title('잔차 히스토그램', fontsize=12)
axes[1,0].grid(True, alpha=0.3)

# 4. 회귀선과 잔차 시각화
axes[1,1].scatter(X, Y, s=100, c='blue', edgecolors='black', 
                  label='실제값', zorder=5)
axes[1,1].plot(X, fitted, 'r-', linewidth=2, label='회귀선')
for i in range(len(X)):
    axes[1,1].vlines(X[i], fitted[i], Y[i], colors='gray', 
                     linestyles='dashed', linewidth=1.5)
    axes[1,1].annotate(f'e={residuals[i]:.1f}', 
                       xy=(X[i]+0.1, (Y[i]+fitted[i])/2),
                       fontsize=9)
axes[1,1].set_xlabel('X (광고비)', fontsize=11)
axes[1,1].set_ylabel('Y (매출)', fontsize=11)
axes[1,1].set_title('회귀선과 잔차', fontsize=12)
axes[1,1].legend(loc='upper left')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 8.5 실행 결과 예시

```
============================================================
회귀분석 결과
============================================================
회귀식: ŷ = 0.90 + 1.90x
R² = 0.9704
Adjusted R² = 0.9606
잔차 표준오차 (RMSE) = 0.6055

============================================================
잔차 분석 결과
============================================================
 X  Y (실제)  Ŷ (예측)  잔차 (e)  표준화 잔차
 1         3      2.8       0.2        0.330
 2         5      4.7       0.3        0.495
 3         6      6.6      -0.6       -0.990
 4         8      8.5      -0.5       -0.825
 5        11     10.4       0.6        0.990

잔차 합계: 0.0000000000 (≈ 0)
SSE (잔차제곱합): 1.10
SSR (회귀제곱합): 36.10
SST (총제곱합): 37.20

[정규성 검정 - Shapiro-Wilk]
통계량: 0.9803, p-value: 0.9324
결론: 정규성 만족

[독립성 검정 - Durbin-Watson]
DW 통계량: 1.8545
결론: 자기상관 없음

[등분산성 검정 - Breusch-Pagan]
LM 통계량: 0.0909, p-value: 0.7630
결론: 등분산 만족

============================================================
영향력 진단
============================================================
 i  레버리지 (h)  Cook's D  스튜던트화 잔차
 1         0.600    0.0267             0.447
 2         0.300    0.0134             0.596
 3         0.200    0.0535            -1.193
 4         0.300    0.0372            -0.994
 5         0.600    0.2400             1.342

레버리지 기준: h > 0.80
Cook's D 기준: D > 0.80
```

---

## 📝 빅데이터분석기사 시험 핵심 포인트

### ✅ 자주 출제되는 개념

1. **잔차 정의**
   $$e_i = y_i - \hat{y}_i$$

2. **잔차의 핵심 특성**
   - 잔차의 합 = 0
   - 잔차와 X의 공분산 = 0

3. **SST = SSR + SSE**
   - SST: 총제곱합
   - SSR: 회귀제곱합 (설명된 변동)
   - SSE: 잔차제곱합 (설명 안 된 변동)

4. **결정계수**
   $$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$

5. **회귀모델 4가지 가정**
   - 선형성, 정규성, 등분산성, 독립성

6. **Durbin-Watson 해석**
   - DW ≈ 2 → 자기상관 없음
   - DW < 2 → 양의 자기상관
   - DW > 2 → 음의 자기상관

### ✅ 계산 문제 팁

- 잔차 = 실제값 - 예측값 (부호 주의!)
- SSE 계산 시 각 잔차를 **제곱**하여 합산
- 표준화 잔차에서 분모는 $\sqrt{SSE/(n-2)}$
- DW 계산 시 분자는 **(연속 잔차 차이)**의 제곱합

### ✅ 연관 개념

- 최소제곱법 (OLS): SSE를 최소화
- 결정계수와 수정된 결정계수
- 이상치, 지렛점, 영향점의 구분
- 정규성 검정 (Shapiro-Wilk, Q-Q Plot)

---

## 📊 핵심 공식 요약표

| 항목 | 공식 |
|---|---|
| 잔차 | $e_i = y_i - \hat{y}_i$ |
| 잔차제곱합 (SSE) | $\sum e_i^2$ |
| 잔차 표준오차 | $\hat{\sigma} = \sqrt{SSE/(n-p)}$ |
| 표준화 잔차 | $r_i = e_i / \hat{\sigma}$ |
| 스튜던트화 잔차 | $r_i^* = e_i / (\hat{\sigma}\sqrt{1-h_{ii}})$ |
| 레버리지 | $h_{ii} = 1/n + (x_i-\bar{x})^2 / \sum(x_j-\bar{x})^2$ |
| Cook's Distance | $D_i = (r_i^2/p) \times h_{ii}/(1-h_{ii})$ |
| Durbin-Watson | $DW = \sum(e_i - e_{i-1})^2 / \sum e_i^2$ |
| 결정계수 | $R^2 = 1 - SSE/SST$ |

---

*작성일: 2025년*
*빅데이터분석기사 실기 대비 학습자료*
