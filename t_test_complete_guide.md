# T 검정 (T-Test) 완전 가이드
## 수식 유도 · 원리 · 사용법 · 수치 예시 · 해석

---

## 목차
1. [T 검정이란?](#1-t-검정이란)
2. [수식 유도 (핵심 원리)](#2-수식-유도-핵심-원리)
3. [T 분포의 특성](#3-t-분포의-특성)
4. [T 검정의 종류](#4-t-검정의-종류)
5. [단일 표본 T 검정 (수치 예시)](#5-단일-표본-t-검정-수치-예시)
6. [독립 표본 T 검정 (수치 예시)](#6-독립-표본-t-검정-수치-예시)
7. [대응 표본 T 검정 (수치 예시)](#7-대응-표본-t-검정-수치-예시)
8. [검정력(Power)과 표본 크기 결정](#8-검정력power과-표본-크기-결정)
9. [Python 전체 구현](#9-python-전체-구현)
10. [가정 위반 시 대안](#10-가정-위반-시-대안)
11. [효과 크기 (Effect Size)](#11-효과-크기-effect-size)
12. [완전 해석 가이드](#12-완전-해석-가이드)

---

## 1. T 검정이란?

T 검정은 **모집단의 분산(σ²)을 모를 때**, 표본 데이터를 이용해 **평균에 관한 가설을 검정**하는 통계 방법이다.

### 배경: 왜 T 검정이 필요한가?

```
이상적인 세계 (σ 알 때)          현실 (σ 모를 때)
────────────────────────          ──────────────────
Z 통계량 사용 가능                T 통계량 사용
         X̄ - μ                           X̄ - μ
Z = ─────────────                T = ─────────────
       σ / √n                           S / √n

Z ~ N(0,1) 표준정규분포           T ~ t(n-1) T분포
```

> 핵심: **σ를 S(표본표준편차)로 대체**하면서 발생하는 불확실성을
> T 분포가 흡수한다.

### 발견 역사

```
1908년 William Sealy Gosset
  ↓ (기네스 맥주 회사 근무 중 소표본 품질관리 문제 직면)
  ↓ (회사 기밀 보호를 위해 "Student" 필명 사용)
  ↓
"Student's t-distribution" 발표
  ↓
현재: 소표본(n<30), 대표본 모두 사용
```

---

## 2. 수식 유도 (핵심 원리)

### 2-1. 표본 평균의 분포

모집단: $X \sim N(\mu, \sigma^2)$에서 크기 $n$인 표본을 추출하면,

$$\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$$

표본 평균의 분포:

$$\bar{X} \sim N\left(\mu,\ \frac{\sigma^2}{n}\right)$$

표준화하면:

$$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$$

### 2-2. 표본분산과 카이제곱 분포

$\sigma^2$을 모를 때 표본분산 $S^2$으로 추정:

$$S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2 \quad \text{(불편 추정량)}$$

**핵심 정리:**

$$\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$$

증명 스케치:
$$\sum_{i=1}^{n}\left(\frac{X_i - \mu}{\sigma}\right)^2 = \sum_{i=1}^{n}\left(\frac{X_i - \bar{X}}{\sigma}\right)^2 + n\left(\frac{\bar{X}-\mu}{\sigma}\right)^2$$

$$\underbrace{\chi^2(n)}_{\text{좌변}} = \underbrace{\frac{(n-1)S^2}{\sigma^2}}_{\chi^2(n-1)} + \underbrace{\left(\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\right)^2}_{\chi^2(1)}$$

따라서 $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$이고, $\bar{X}$와 독립.

### 2-3. T 통계량 유도 ★

**T 분포의 정의:** 독립인 $Z \sim N(0,1)$, $V \sim \chi^2(\nu)$일 때

$$T = \frac{Z}{\sqrt{V/\nu}} \sim t(\nu)$$

이를 단일 표본 T 검정에 적용:

$$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}}, \quad V = \frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$$

$$T = \frac{Z}{\sqrt{V/(n-1)}} = \frac{\dfrac{\bar{X}-\mu}{\sigma/\sqrt{n}}}{\sqrt{\dfrac{(n-1)S^2}{\sigma^2(n-1)}}} = \frac{\dfrac{\bar{X}-\mu}{\sigma/\sqrt{n}}}{\dfrac{S}{\sigma}} = \frac{\bar{X}-\mu}{S/\sqrt{n}}$$

$$\boxed{T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t(n-1)}$$

> **σ가 완전히 소거됨!** S/√n이 표준오차(SE)의 추정값 역할을 한다.

### 2-4. 독립 표본 T 통계량 유도

두 집단 $X \sim N(\mu_1, \sigma^2)$, $Y \sim N(\mu_2, \sigma^2)$ (등분산 가정):

합동 분산(Pooled Variance):

$$S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$$

유도 과정:
$$\frac{(n_1-1)S_1^2}{\sigma^2} \sim \chi^2(n_1-1), \quad \frac{(n_2-1)S_2^2}{\sigma^2} \sim \chi^2(n_2-1)$$

독립이면 합산: $\frac{(n_1+n_2-2)S_p^2}{\sigma^2} \sim \chi^2(n_1+n_2-2)$

$$(\bar{X} - \bar{Y}) \sim N\left(\mu_1-\mu_2,\ \sigma^2\left(\frac{1}{n_1}+\frac{1}{n_2}\right)\right)$$

$$\boxed{T = \frac{(\bar{X}-\bar{Y}) - (\mu_1-\mu_2)}{S_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}} \sim t(n_1+n_2-2)}$$

### 2-5. 대응 표본 T 통계량 유도

쌍(pair)의 차이 $D_i = X_{1i} - X_{2i}$로 변환:

$$D_i \sim N(\mu_D, \sigma_D^2)$$

단일 표본 T 검정과 동일하게:

$$\bar{D} = \frac{1}{n}\sum_{i=1}^n D_i, \quad S_D = \sqrt{\frac{\sum(D_i - \bar{D})^2}{n-1}}$$

$$\boxed{T = \frac{\bar{D} - 0}{S_D/\sqrt{n}} \sim t(n-1)}$$

---

## 3. T 분포의 특성

### 자유도(df)에 따른 형태 변화

```
    t분포(df=1)  ← 꼬리 가장 두꺼움 (불확실성 최대)
    t분포(df=5)
    t분포(df=30)
    t분포(df=∞) = 표준정규분포 N(0,1)  ← 꼬리 가장 얇음
```

**왜 꼬리가 두꺼운가?**

S로 σ를 추정하기 때문에 추가 불확실성 발생.
→ 표본이 작을수록(자유도 낮을수록) 극단값이 나올 확률↑

### 주요 임계값 비교

| 자유도(df) | α=0.05 (양측) | α=0.01 (양측) | α=0.05 (단측) |
|-----------|-------------|-------------|-------------|
| 1         | ±12.706     | ±63.657     | 6.314       |
| 5         | ±2.571      | ±4.032      | 2.015       |
| 10        | ±2.228      | ±3.169      | 1.812       |
| 20        | ±2.086      | ±2.845      | 1.725       |
| 30        | ±2.042      | ±2.750      | 1.697       |
| ∞ (Z)     | ±1.960      | ±2.576      | 1.645       |

> df가 클수록 표준정규분포에 수렴 → 소표본일수록 더 엄격한 기준

---

## 4. T 검정의 종류

```
T 검정
├── 단일 표본 T 검정 (One-Sample t-test)
│   └── 표본 평균 vs 알려진 모집단 평균(μ₀)
│
├── 독립 표본 T 검정 (Independent Two-Sample t-test)
│   ├── 등분산 (Student's t-test): S_p 사용
│   └── 이분산 (Welch's t-test): 자유도 보정
│
└── 대응 표본 T 검정 (Paired t-test)
    └── 동일 개체의 전·후 비교 (D = X_after - X_before)
```

### 적용 조건 (가정)

| 가정 | 확인 방법 |
|------|---------|
| **정규성** | Shapiro-Wilk 검정, Q-Q 플롯 |
| **독립성** | 실험 설계로 확인 |
| **등분산** (독립 표본) | Levene 검정, F-검정 |

---

## 5. 단일 표본 T 검정 (수치 예시)

### 문제 설정

> 어느 커피 전문점은 아메리카노 1잔의 카페인 함량이 **150mg**이라고 표기한다.
> 소비자단체가 무작위로 **10잔**을 구매해 분석한 결과는 다음과 같다.
> 표기 함량(150mg)이 정확한지 유의수준 **α = 0.05**로 검정하라.

**데이터:**

| 잔 번호 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|--------|---|---|---|---|---|---|---|---|---|---|
| 카페인(mg) | 162 | 148 | 155 | 171 | 142 | 158 | 149 | 163 | 145 | 157 |

---

### Step 1: 가설 설정

$$H_0: \mu = 150 \quad \text{(표기 함량과 동일)}$$
$$H_1: \mu \neq 150 \quad \text{(표기 함량과 다름, 양측 검정)}$$

---

### Step 2: 기술 통계량 계산

**표본 평균:**

$$\bar{X} = \frac{162+148+155+171+142+158+149+163+145+157}{10} = \frac{1550}{10} = 155.0$$

**편차 계산:**

| $X_i$ | $X_i - \bar{X}$ | $(X_i - \bar{X})^2$ |
|--------|----------------|-------------------|
| 162    | +7.0           | 49.00             |
| 148    | −7.0           | 49.00             |
| 155    | 0.0            | 0.00              |
| 171    | +16.0          | 256.00            |
| 142    | −13.0          | 169.00            |
| 158    | +3.0           | 9.00              |
| 149    | −6.0           | 36.00             |
| 163    | +8.0           | 64.00             |
| 145    | −10.0          | 100.00            |
| 157    | +2.0           | 4.00              |
| **합** | **0.0** ✓      | **736.00**        |

**표본 분산 (불편 추정량):**

$$S^2 = \frac{\sum(X_i - \bar{X})^2}{n-1} = \frac{736.00}{9} = 81.78$$

**표본 표준편차:**

$$S = \sqrt{81.78} = 9.043$$

**표준 오차 (SE):**

$$SE = \frac{S}{\sqrt{n}} = \frac{9.043}{\sqrt{10}} = \frac{9.043}{3.162} = 2.860$$

---

### Step 3: T 통계량 계산

$$T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} = \frac{155.0 - 150.0}{2.860} = \frac{5.0}{2.860} = \mathbf{1.748}$$

---

### Step 4: 기각역 및 p값 결정

- 자유도: $df = n - 1 = 9$
- 유의수준: $\alpha = 0.05$ (양측)
- 임계값: $t_{0.025, 9} = \pm 2.262$

```
기각역 도식:
         기각역          채택역          기각역
──────────┤────────────────────────────────┤──────────
        -2.262           0            +2.262

                        T = 1.748
                           ↑
                      (채택역 안에 있음)
```

**p값 계산:**

$$p\text{-value} = 2 \times P(T_{9} > 1.748) = 2 \times 0.0572 \approx 0.1143$$

---

### Step 5: 판정 및 해석

| 항목 | 값 |
|------|-----|
| T 통계량 | **1.748** |
| 임계값 | ±2.262 |
| p-value | **0.1143** |
| 판정 기준 | p > α (0.1143 > 0.05) |
| **결론** | **H₀ 채택 (기각 실패)** |

> **해석:**
> 표본에서 측정된 평균 카페인 함량(155.0mg)이 표기값(150mg)보다 5mg 높지만,
> 이 차이는 통계적으로 유의하지 않다 (p=0.114 > 0.05).
> 즉, 이 정도의 차이는 표본 추출의 우연한 변동으로 설명될 수 있으며,
> 커피 전문점의 표기가 거짓이라는 충분한 증거가 없다.

**95% 신뢰구간:**

$$\bar{X} \pm t_{0.025,9} \cdot SE = 155.0 \pm 2.262 \times 2.860 = 155.0 \pm 6.469$$

$$\boxed{95\% \text{ CI}: [148.53,\ 161.47]}$$

> μ₀=150이 신뢰구간 안에 포함 → H₀ 채택과 일치

---

## 6. 독립 표본 T 검정 (수치 예시)

### 문제 설정

> 두 학원(A, B)의 수능 수학 성적을 비교하려 한다.
> 각 학원에서 무작위로 학생을 선발해 점수를 측정했다.
> 두 학원의 평균 성적에 차이가 있는지 **α = 0.05**로 검정하라.

**데이터:**

| A 학원 (n₁=8) | 85 | 92 | 78 | 95 | 88 | 82 | 90 | 87 |
|--------------|----|----|----|----|----|----|----|----|

| B 학원 (n₂=7) | 79 | 85 | 91 | 74 | 83 | 77 | 80 |
|--------------|----|----|----|----|----|----|---|

---

### Step 1: 가설 설정

$$H_0: \mu_A = \mu_B \quad (\mu_A - \mu_B = 0)$$
$$H_1: \mu_A \neq \mu_B \quad \text{(양측)}$$

---

### Step 2: 기술 통계량

**A 학원:**

$$\bar{X}_A = \frac{85+92+78+95+88+82+90+87}{8} = \frac{697}{8} = 87.125$$

| $X_i$ | $X_i - \bar{X}_A$ | $(X_i - \bar{X}_A)^2$ |
|--------|------------------|---------------------|
| 85     | −2.125           | 4.516               |
| 92     | +4.875           | 23.766              |
| 78     | −9.125           | 83.266              |
| 95     | +7.875           | 62.016              |
| 88     | +0.875           | 0.766               |
| 82     | −5.125           | 26.266              |
| 90     | +2.875           | 8.266               |
| 87     | −0.125           | 0.016               |
| **합** | **0** ✓          | **208.875**         |

$$S_A^2 = \frac{208.875}{7} = 29.839, \quad S_A = 5.463$$

**B 학원:**

$$\bar{X}_B = \frac{79+85+91+74+83+77+80}{7} = \frac{569}{7} = 81.286$$

| $X_i$ | $X_i - \bar{X}_B$ | $(X_i - \bar{X}_B)^2$ |
|--------|------------------|---------------------|
| 79     | −2.286           | 5.224               |
| 85     | +3.714           | 13.796              |
| 91     | +9.714           | 94.368              |
| 74     | −7.286           | 53.082              |
| 83     | +1.714           | 2.938               |
| 77     | −4.286           | 18.368              |
| 80     | −1.286           | 1.653               |
| **합** | **0** ✓          | **189.429**         |

$$S_B^2 = \frac{189.429}{6} = 31.571, \quad S_B = 5.619$$

---

### Step 3: 등분산 검정 (Levene's Test)

$H_0: \sigma_A^2 = \sigma_B^2$ → p=0.762 > 0.05 → **등분산 가정 성립** → Student's t-test 적용

---

### Step 4: 합동 분산 및 T 통계량

**합동 분산:**

$$S_p^2 = \frac{(n_1-1)S_A^2 + (n_2-1)S_B^2}{n_1+n_2-2} = \frac{7 \times 29.839 + 6 \times 31.571}{8+7-2}$$

$$S_p^2 = \frac{208.875 + 189.429}{13} = \frac{398.304}{13} = 30.639$$

$$S_p = \sqrt{30.639} = 5.535$$

**표준 오차:**

$$SE = S_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}} = 5.535 \times \sqrt{\frac{1}{8}+\frac{1}{7}} = 5.535 \times \sqrt{0.2679} = 5.535 \times 0.5176 = 2.865$$

**T 통계량:**

$$T = \frac{\bar{X}_A - \bar{X}_B}{SE} = \frac{87.125 - 81.286}{2.865} = \frac{5.839}{2.865} = \mathbf{2.038}$$

---

### Step 5: 판정

- 자유도: $df = n_1 + n_2 - 2 = 13$
- 임계값: $t_{0.025, 13} = \pm 2.160$
- p-value: $2 \times P(T_{13} > 2.038) \approx 0.0623$

```
기각역 도식:
        기각역          채택역          기각역
──────────┤──────────────────────────────┤──────────
        -2.160           0           +2.160

                             T = 2.038
                                ↑
                          (채택역 안, 근접)
```

| 항목 | 값 |
|------|-----|
| T 통계량 | **2.038** |
| 임계값 | ±2.160 |
| p-value | **0.0623** |
| 판정 | p > 0.05 → **H₀ 채택** |

> **해석:**
> A 학원 평균(87.1점)이 B 학원(81.3점)보다 5.8점 높지만,
> p=0.062로 유의수준 0.05를 **간신히 넘기지 못해** 통계적으로 유의하지 않다.
> **단, p=0.062는 '경계선상'이므로**, α=0.10으로 검정하면 유의하다.
> 표본 크기를 늘려 재검정하는 것이 권장된다.

**95% 신뢰구간 (차이):**

$$(\bar{X}_A - \bar{X}_B) \pm t_{0.025,13} \times SE = 5.839 \pm 2.160 \times 2.865$$

$$= 5.839 \pm 6.188 \Rightarrow \boxed{[-0.349,\ 12.027]}$$

> 0이 신뢰구간에 포함 → "차이가 0일 수도 있다" → H₀ 채택과 일치

---

### Welch's T-test (이분산 시)

$$T_W = \frac{\bar{X}_A - \bar{X}_B}{\sqrt{\frac{S_A^2}{n_1}+\frac{S_B^2}{n_2}}}$$

**Welch-Satterthwaite 자유도 보정:**

$$df_W = \frac{\left(\frac{S_A^2}{n_1}+\frac{S_B^2}{n_2}\right)^2}{\frac{(S_A^2/n_1)^2}{n_1-1}+\frac{(S_B^2/n_2)^2}{n_2-1}}$$

이분산일 경우 위 공식 사용 → 자유도가 소수점이 될 수 있음.

---

## 7. 대응 표본 T 검정 (수치 예시)

### 문제 설정

> 다이어트 프로그램의 효과를 검증하기 위해 **12명**의 참가자 체중(kg)을
> 프로그램 **전·후**로 측정했다.
> 체중 감소 효과가 있는지 **α = 0.05** (단측)로 검정하라.

**데이터:**

| 참가자 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|--------|---|---|---|---|---|---|---|---|---|----|----|-----|
| **전** (kg) | 78 | 85 | 92 | 71 | 88 | 95 | 76 | 83 | 90 | 69 | 86 | 91 |
| **후** (kg) | 74 | 81 | 88 | 70 | 84 | 89 | 75 | 79 | 85 | 68 | 82 | 87 |

---

### Step 1: 가설 설정

$$H_0: \mu_D \leq 0 \quad \text{(효과 없거나 체중 증가)}$$
$$H_1: \mu_D > 0 \quad \text{(체중 감소 효과 있음, 단측 검정)}$$

여기서 $D_i = \text{전}_i - \text{후}_i$ (양수면 감소)

---

### Step 2: 차이 계산

| 참가자 | 전 | 후 | $D_i=전-후$ | $D_i - \bar{D}$ | $(D_i - \bar{D})^2$ |
|--------|----|----|-----------|----------------|-------------------|
| 1      | 78 | 74 | **4**     | 0.333          | 0.111             |
| 2      | 85 | 81 | **4**     | 0.333          | 0.111             |
| 3      | 92 | 88 | **4**     | 0.333          | 0.111             |
| 4      | 71 | 70 | **1**     | −2.667         | 7.111             |
| 5      | 88 | 84 | **4**     | 0.333          | 0.111             |
| 6      | 95 | 89 | **6**     | 2.333          | 5.444             |
| 7      | 76 | 75 | **1**     | −2.667         | 7.111             |
| 8      | 83 | 79 | **4**     | 0.333          | 0.111             |
| 9      | 90 | 85 | **5**     | 1.333          | 1.778             |
| 10     | 69 | 68 | **1**     | −2.667         | 7.111             |
| 11     | 86 | 82 | **4**     | 0.333          | 0.111             |
| 12     | 91 | 87 | **4**     | 0.333          | 0.111             |
| **합** |    |    | **42**    | **0** ✓        | **29.333**        |

---

### Step 3: 기술 통계량

$$\bar{D} = \frac{42}{12} = 3.500 \text{ kg}$$

$$S_D^2 = \frac{29.333}{11} = 2.667, \quad S_D = \sqrt{2.667} = 1.633 \text{ kg}$$

$$SE_D = \frac{S_D}{\sqrt{n}} = \frac{1.633}{\sqrt{12}} = \frac{1.633}{3.464} = 0.4714 \text{ kg}$$

---

### Step 4: T 통계량

$$T = \frac{\bar{D} - 0}{SE_D} = \frac{3.500}{0.4714} = \mathbf{7.423}$$

---

### Step 5: 판정

- 자유도: $df = n - 1 = 11$
- 임계값 (단측): $t_{0.05, 11} = 1.796$
- p-value: $P(T_{11} > 7.423) \approx 0.0000076$

```
단측 검정 기각역:
         채택역                기각역
──────────────────────────────┤──────────
              0              1.796

                                  T = 7.423
                                        ↑
                            (기각역에 매우 깊이 위치)
```

| 항목 | 값 |
|------|-----|
| $\bar{D}$ | **3.500 kg** |
| $S_D$ | 1.633 kg |
| T 통계량 | **7.423** |
| 임계값 (단측) | 1.796 |
| p-value | **< 0.0001** |
| **결론** | **H₀ 기각 → 다이어트 효과 있음** |

> **해석:**
> 12명의 참가자는 프로그램 후 평균 **3.5kg** 감소했다.
> T=7.423으로 임계값 1.796을 크게 초과하며, p<0.0001로
> 이 감소는 우연이 아닌 **통계적으로 매우 유의한 효과**이다.
> 이 다이어트 프로그램은 체중 감소에 효과적이라 할 수 있다.

**95% 단측 신뢰구간 (하한):**

$$\bar{D} - t_{0.05,11} \times SE_D = 3.500 - 1.796 \times 0.4714 = 3.500 - 0.847 = 2.653$$

$$\text{단측 95% CI: } [2.653,\ +\infty)$$

> 신뢰구간 하한이 2.653 > 0 → 최소 2.653kg 이상 감소 기대

---

## 8. 검정력(Power)과 표본 크기 결정

### 검정력 개념

```
실제 H₀ 참          실제 H₀ 거짓
──────────────────────────────────
H₀ 채택: 올바른 결정    2종 오류(β)
H₀ 기각: 1종 오류(α)   올바른 결정 → 검정력(1-β)
```

$$\text{검정력} = P(\text{H}_0 \text{ 기각} \mid \text{H}_1 \text{ 참}) = 1 - \beta$$

### 표본 크기 공식 (단일 표본)

$$n = \left(\frac{(z_{\alpha/2} + z_\beta) \cdot \sigma}{\delta}\right)^2$$

- $\delta$: 탐지하고 싶은 최소 차이 (effect size × σ)
- $z_{\alpha/2}$: 유의수준에 따른 Z값
- $z_\beta$: 검정력에 따른 Z값

**수치 예시:**

> α=0.05, 검정력=80%, 탐지 차이=5mg, σ=9.043

$$n = \left(\frac{(1.96 + 0.842) \times 9.043}{5}\right)^2 = \left(\frac{2.802 \times 9.043}{5}\right)^2 = \left(\frac{25.34}{5}\right)^2 = (5.068)^2 \approx 26$$

→ **26개 이상의 표본이 필요**하다 (앞 예시의 10개는 부족!)

---

## 9. Python 전체 구현

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════
# 1. 단일 표본 T 검정
# ══════════════════════════════════════════════════════════
print("=" * 55)
print("1. 단일 표본 T 검정 (커피 카페인)")
print("=" * 55)

caffeine = np.array([162, 148, 155, 171, 142, 158, 149, 163, 145, 157])
mu0 = 150

n    = len(caffeine)
xbar = caffeine.mean()
s    = caffeine.std(ddof=1)
se   = s / np.sqrt(n)
t_stat, p_val = stats.ttest_1samp(caffeine, mu0)

print(f"n          = {n}")
print(f"X̄          = {xbar:.4f}")
print(f"S          = {s:.4f}")
print(f"SE         = {se:.4f}")
print(f"T 통계량   = {t_stat:.4f}")
print(f"p-value    = {p_val:.4f}")
print(f"결론: {'H₀ 기각' if p_val < 0.05 else 'H₀ 채택'}")

# 신뢰구간
ci = stats.t.interval(0.95, df=n-1, loc=xbar, scale=se)
print(f"95% CI     = [{ci[0]:.4f}, {ci[1]:.4f}]")

# ══════════════════════════════════════════════════════════
# 2. 독립 표본 T 검정
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("2. 독립 표본 T 검정 (학원 성적 비교)")
print("=" * 55)

A = np.array([85, 92, 78, 95, 88, 82, 90, 87])
B = np.array([79, 85, 91, 74, 83, 77, 80])

# 등분산 검정 (Levene)
levene_stat, levene_p = stats.levene(A, B)
print(f"Levene 검정: p={levene_p:.4f} → {'등분산' if levene_p>0.05 else '이분산'}")

# 등분산: Student's t / 이분산: Welch's t
equal_var = levene_p > 0.05
t2, p2 = stats.ttest_ind(A, B, equal_var=equal_var)

n1, n2    = len(A), len(B)
s1, s2    = A.std(ddof=1), B.std(ddof=1)
sp2       = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
sp        = np.sqrt(sp2)
se_diff   = sp * np.sqrt(1/n1 + 1/n2)

print(f"A 평균={A.mean():.3f}, S={s1:.3f}")
print(f"B 평균={B.mean():.3f}, S={s2:.3f}")
print(f"합동 표준편차Sp = {sp:.4f}")
print(f"T 통계량       = {t2:.4f}")
print(f"p-value        = {p2:.4f}")
print(f"결론: {'H₀ 기각' if p2 < 0.05 else 'H₀ 채택'}")

ci2 = stats.t.interval(0.95, df=n1+n2-2,
                        loc=A.mean()-B.mean(), scale=se_diff)
print(f"95% CI(차이)   = [{ci2[0]:.4f}, {ci2[1]:.4f}]")

# ══════════════════════════════════════════════════════════
# 3. 대응 표본 T 검정
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("3. 대응 표본 T 검정 (다이어트 효과)")
print("=" * 55)

before = np.array([78, 85, 92, 71, 88, 95, 76, 83, 90, 69, 86, 91])
after  = np.array([74, 81, 88, 70, 84, 89, 75, 79, 85, 68, 82, 87])
diff   = before - after

t3, p3_two = stats.ttest_rel(before, after)
p3_one = p3_two / 2  # 단측 p값

print(f"평균 차이(D̄)  = {diff.mean():.4f} kg")
print(f"S_D           = {diff.std(ddof=1):.4f} kg")
print(f"T 통계량      = {t3:.4f}")
print(f"단측 p-value  = {p3_one:.6f}")
print(f"결론: {'H₀ 기각 → 다이어트 효과 있음' if p3_one < 0.05 else 'H₀ 채택'}")

n3 = len(diff)
se3 = diff.std(ddof=1) / np.sqrt(n3)
ci3_lower = diff.mean() - stats.t.ppf(0.95, df=n3-1) * se3
print(f"단측 95% CI   = [{ci3_lower:.4f}, +∞)")

# ══════════════════════════════════════════════════════════
# 4. 효과 크기 (Cohen's d)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("4. 효과 크기 (Cohen's d)")
print("=" * 55)

def cohens_d_one(x, mu0):
    return (x.mean() - mu0) / x.std(ddof=1)

def cohens_d_two(x1, x2):
    sp = np.sqrt(((len(x1)-1)*x1.std(ddof=1)**2 +
                  (len(x2)-1)*x2.std(ddof=1)**2) / (len(x1)+len(x2)-2))
    return (x1.mean() - x2.mean()) / sp

def cohens_d_paired(d):
    return d.mean() / d.std(ddof=1)

d1 = cohens_d_one(caffeine, mu0)
d2 = cohens_d_two(A, B)
d3 = cohens_d_paired(diff)

def interpret_d(d):
    d = abs(d)
    if d < 0.2: return "무시할 수 있음"
    elif d < 0.5: return "소 (small)"
    elif d < 0.8: return "중 (medium)"
    else: return "대 (large)"

print(f"단일 표본 d = {d1:.4f} → {interpret_d(d1)}")
print(f"독립 표본 d = {d2:.4f} → {interpret_d(d2)}")
print(f"대응 표본 d = {d3:.4f} → {interpret_d(d3)}")

# ══════════════════════════════════════════════════════════
# 5. 시각화 (T 분포 + 기각역)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('T 검정 시각화 (기각역 및 T 통계량)', fontsize=14, fontweight='bold')

test_cases = [
    ("단일 표본\n(카페인)", t_stat, 9,  p_val,  '양측', '#2196F3'),
    ("독립 표본\n(학원 성적)", t2, 13, p2, '양측', '#4CAF50'),
    ("대응 표본\n(다이어트)", t3, 11, p3_two, '단측', '#FF5722'),
]

for ax, (title, t_val, df, pv, sides, color) in zip(axes, test_cases):
    x = np.linspace(-5, 5, 500)
    y = stats.t.pdf(x, df=df)

    ax.plot(x, y, 'k-', linewidth=2, label=f't분포(df={df})')

    if sides == '양측':
        crit = stats.t.ppf(0.975, df=df)
        # 기각역 색칠
        x_left  = x[x <= -crit]
        x_right = x[x >=  crit]
        ax.fill_between(x_left,  stats.t.pdf(x_left,  df), alpha=0.4, color='red', label='기각역(α=0.05)')
        ax.fill_between(x_right, stats.t.pdf(x_right, df), alpha=0.4, color='red')
        ax.axvline(-crit, color='red', linestyle='--', linewidth=1.5)
        ax.axvline( crit, color='red', linestyle='--', linewidth=1.5)
    else:
        crit = stats.t.ppf(0.95, df=df)
        x_right = x[x >= crit]
        ax.fill_between(x_right, stats.t.pdf(x_right, df), alpha=0.4, color='red', label='기각역(α=0.05)')
        ax.axvline(crit, color='red', linestyle='--', linewidth=1.5)

    # T 통계량 표시 (범위 내로 클리핑)
    t_plot = max(min(t_val, 4.9), -4.9)
    ax.axvline(t_plot, color=color, linestyle='-', linewidth=2.5, label=f'T={t_val:.3f}')
    ax.set_title(f'{title}\np={pv:.4f}', fontsize=11)
    ax.set_xlabel('T 값')
    ax.set_ylabel('확률 밀도')
    ax.legend(fontsize=8)
    ax.set_xlim(-5, 5)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('t_test_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n시각화 저장 완료: t_test_visualization.png")
```

---

## 10. 가정 위반 시 대안

### 정규성 위반

```
정규성 검정 (Shapiro-Wilk)
├── p > 0.05: 정규성 가정 만족 → T 검정 사용
└── p < 0.05: 정규성 위반
    ├── n ≥ 30: 중심극한정리에 의해 T 검정 여전히 사용 가능
    └── n < 30: 비모수 검정으로 대체
        ├── 단일 표본 → Wilcoxon Signed-Rank Test
        ├── 독립 표본 → Mann-Whitney U Test
        └── 대응 표본 → Wilcoxon Signed-Rank Test
```

### Python 비모수 대안

```python
from scipy import stats

# 단일 표본 비모수: Wilcoxon
stat, p = stats.wilcoxon(caffeine - 150)
print(f"Wilcoxon (단일): stat={stat}, p={p:.4f}")

# 독립 표본 비모수: Mann-Whitney U
stat, p = stats.mannwhitneyu(A, B, alternative='two-sided')
print(f"Mann-Whitney U: stat={stat}, p={p:.4f}")

# 대응 표본 비모수: Wilcoxon
stat, p = stats.wilcoxon(before, after, alternative='greater')
print(f"Wilcoxon (대응): stat={stat}, p={p:.4f}")
```

---

## 11. 효과 크기 (Effect Size)

효과 크기는 통계적 유의성과 별개로 **실질적 중요성**을 나타낸다.

### Cohen's d 해석 기준

| Cohen's d | 해석 | 예시 |
|-----------|------|------|
| 0.0 ~ 0.2 | 무시할 수 있음 | 거의 차이 없음 |
| 0.2 ~ 0.5 | 소 (Small) | 코칭 효과 |
| 0.5 ~ 0.8 | 중 (Medium) | 새 교수법 효과 |
| 0.8 이상  | 대 (Large)  | 획기적인 신약 효과 |

### 중요성

```
⚠️ 통계적 유의 ≠ 실질적 의미

예: n=100,000일 때 평균 차이 0.001mg이 p<0.05이지만
    Cohen's d ≈ 0.001 → 실질적으로 무의미

⚠️ 표본이 작으면 큰 효과도 유의하지 않을 수 있음

예: 앞 학원 비교에서 d=0.763 (중~대 효과)이지만
    p=0.062로 유의하지 않음 → 표본 크기 부족
```

---

## 12. 완전 해석 가이드

### 결과 보고 표준 형식

```
t(df) = T값, p = p값, d = 코헨d, 95% CI [하한, 상한]

예: t(9) = 1.748, p = .114, d = 0.553, 95% CI [148.53, 161.47]
```

### 의사결정 트리

```
데이터 수집
     ↓
정규성 검정 (Shapiro-Wilk)
     ├── 만족 (p>0.05) 또는 n≥30
     │        ↓
     │   비교 대상 결정
     │   ├── 모집단 vs 표본 평균  → 단일 표본 T
     │   ├── 두 집단 비교
     │   │   ├── 등분산 (Levene p>0.05) → 독립 표본 T (Student)
     │   │   └── 이분산 (Levene p<0.05) → 독립 표본 T (Welch)
     │   └── 동일 개체 전·후 비교    → 대응 표본 T
     │
     └── 위반 (p<0.05) 및 n<30    → 비모수 검정
```

### p값 해석 시 주의사항

```
✅ 올바른 해석:
"p=0.03은, H₀가 참이라 가정할 때 이 정도의 차이가
 우연히 발생할 확률이 3%임을 의미한다."

❌ 틀린 해석:
"p=0.03은 H₀가 참일 확률이 3%다." (베이즈 확률과 혼동)
"p=0.06은 거의 유의하다." (임의의 절사점에 집착)
"유의하지 않으면 H₀가 참이다." (부재 증거는 증거의 부재가 아님)
```

---

## 핵심 공식 요약

$$\text{단일 표본: } T = \frac{\bar{X}-\mu_0}{S/\sqrt{n}} \sim t(n-1)$$

$$\text{독립 표본: } T = \frac{(\bar{X}_1-\bar{X}_2)}{S_p\sqrt{1/n_1+1/n_2}} \sim t(n_1+n_2-2), \quad S_p^2=\frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}$$

$$\text{대응 표본: } T = \frac{\bar{D}}{S_D/\sqrt{n}} \sim t(n-1), \quad D_i = X_{1i}-X_{2i}$$

$$\text{Cohen's d (단일): } d = \frac{\bar{X}-\mu_0}{S}, \quad \text{Cohen's d (독립): } d = \frac{\bar{X}_1-\bar{X}_2}{S_p}$$

---

*참고: Student (W.S. Gosset, 1908), Cohen (1988) Statistical Power Analysis for the Behavioral Sciences*
