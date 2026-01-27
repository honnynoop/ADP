# Lift & Captured Response 완벽 가이드
## 모델 성능 평가와 타겟 마케팅 분석

## 목차
1. [Lift 개요](#1-lift-개요)
2. [Captured Response](#2-captured-response)
3. [Gain Chart](#3-gain-chart)
4. [계산 예제](#4-계산-예제)
5. [실전 응용](#5-실전-응용)
6. [Python 구현](#6-python-구현)
7. [시험 대비](#7-시험-대비)

---

## 1. Lift 개요

### 1.1 Lift란?

**Lift (향상도)**는 예측 모델을 사용했을 때 무작위 선택 대비 얼마나 더 효과적인지를 나타내는 지표입니다.

**정의:**
```
Lift = 모델 사용 시 반응률 / 전체 반응률
     = P(Y=1|예측=1) / P(Y=1)
```

**의미:**
- Lift = 2.0: 모델을 사용하면 무작위 대비 **2배** 효과적
- Lift = 1.0: 모델과 무작위가 동일
- Lift < 1.0: 모델이 무작위보다 못함

### 1.2 왜 Lift가 중요한가?

**비즈니스 시나리오:**

```
상황: 100,000명 고객 중 5,000명이 구매 (전체 반응률 5%)
목표: 10,000명에게 DM 발송

방법 1 - 무작위 선택:
  10,000명 × 5% = 500명 구매 예상

방법 2 - 모델 사용 (Lift=3.0):
  10,000명 × (5% × 3.0) = 1,500명 구매 예상
  
차이: 1,000명 추가 구매! (200% 개선)
```

### 1.3 Lift의 종류

| 종류 | 설명 | 용도 |
|------|------|------|
| **Decile Lift** | 10분위별 향상도 | 구간별 성능 비교 |
| **Cumulative Lift** | 누적 향상도 | 상위 N% 효과 측정 |
| **Overall Lift** | 전체 향상도 | 모델 전체 성능 |

### 1.4 Lift 계산 공식

**Decile Lift:**
```
Lift(i) = (해당 구간 반응률) / (전체 반응률)
        = [구간 i의 반응자 수 / 구간 i의 총 인원] / [전체 반응자 수 / 전체 인원]
```

**Cumulative Lift:**
```
Cum_Lift(i) = (상위 i%까지 반응률) / (전체 반응률)
            = [상위 i%까지 반응자 수 / 상위 i%까지 총 인원] / [전체 반응자 수 / 전체 인원]
```

---

## 2. Captured Response

### 2.1 Captured Response란?

**Captured Response (포착 반응률)** 또는 **%Captured Response**는 상위 N%의 고객을 타겟팅했을 때 전체 반응자 중 몇 %를 포착할 수 있는지를 나타냅니다.

**정의:**
```
Captured Response(i) = (상위 i%에 포함된 반응자 수) / (전체 반응자 수) × 100%
```

**의미:**
- 상위 10%를 타겟팅 → 전체 반응자의 30% 포착
- 상위 20%를 타겟팅 → 전체 반응자의 50% 포착

### 2.2 왜 중요한가?

**비용-효과 분석:**

```
전체 고객: 100,000명
전체 반응자: 5,000명
DM 비용: 1,000원/명
구매당 이익: 10,000원

시나리오 1 - 전체 발송:
  비용: 100,000 × 1,000 = 1억원
  수익: 5,000 × 10,000 = 5억원
  순이익: 4억원

시나리오 2 - 상위 20% 발송 (Captured Response=60%):
  비용: 20,000 × 1,000 = 2천만원
  수익: 3,000 × 10,000 = 3억원
  순이익: 2.8억원
  
→ 80% 비용 절감으로 70% 수익 달성!
```

### 2.3 Gain과의 관계

**Gain = Captured Response**
- 같은 개념의 다른 표현
- Gain Chart = Lift Chart의 누적 버전

```
예시:
상위 20% 타겟팅으로 전체 반응자의 50% 포착
→ Gain = 50%
→ Captured Response = 50%
```

---

## 3. Gain Chart

### 3.1 Gain Chart란?

**Gain Chart (이득 차트)**는 예측 점수 순으로 정렬했을 때 누적 반응자 비율을 보여주는 그래프입니다.

**구성 요소:**

```
X축: 타겟팅한 고객 비율 (0% ~ 100%)
Y축: 포착한 반응자 비율 (0% ~ 100%)

두 개의 선:
1. 모델 곡선 (Gain Curve): 실제 모델 성능
2. 기준선 (Baseline): 무작위 선택 (대각선)
```

### 3.2 Gain Chart 해석

```
        100% ┤                    ●
             │                 ●
             │              ●
 Captured    │           ●
 Response    │        ●              ← Model Curve
   (%)       │     ●
             │   ●
             │ ●
           0 ┼─────────────────────
             0%                  100%
               Population (%)
               
             ╱ ← Baseline (Random)
```

**해석 포인트:**
1. **곡선이 대각선보다 위** → 모델이 무작위보다 좋음
2. **곡선의 기울기가 가파름** → 상위권에 반응자 집중
3. **곡선과 대각선 사이 면적** → 모델의 전체 효과

### 3.3 Lift Chart vs Gain Chart

| 특징 | Lift Chart | Gain Chart |
|------|-----------|-----------|
| **Y축** | Lift (향상도) | Cumulative % Response |
| **기준선** | Y = 1.0 (수평선) | Y = X (대각선) |
| **용도** | 구간별 효과 비교 | 누적 효과 확인 |
| **해석** | 비율 (몇 배) | 백분율 (몇 %) |

---

## 4. 계산 예제

### 4.1 기본 데이터

**시나리오:** 1,000명 고객에게 마케팅 캠페인 실시

| 고객 ID | 예측 점수 | 실제 반응 |
|---------|----------|----------|
| 001 | 0.95 | 1 (Yes) |
| 002 | 0.92 | 1 (Yes) |
| 003 | 0.88 | 1 (Yes) |
| ... | ... | ... |
| 1000 | 0.05 | 0 (No) |

**전체 통계:**
- 총 고객: 1,000명
- 실제 반응자: 100명
- 전체 반응률: 100/1,000 = 10%

### 4.2 Decile 분석 테이블

**Step 1: 예측 점수로 정렬하여 10분위 생성**

| Decile | 고객 수 | 실제 반응자 | 반응률 | Lift | Cumulative 반응자 | Cumulative Lift | %Captured |
|--------|---------|------------|--------|------|------------------|----------------|-----------|
| 1 (상위 10%) | 100 | 35 | 35% | 3.5 | 35 | 3.5 | 35% |
| 2 | 100 | 20 | 20% | 2.0 | 55 | 2.75 | 55% |
| 3 | 100 | 15 | 15% | 1.5 | 70 | 2.33 | 70% |
| 4 | 100 | 10 | 10% | 1.0 | 80 | 2.0 | 80% |
| 5 | 100 | 8 | 8% | 0.8 | 88 | 1.76 | 88% |
| 6 | 100 | 5 | 5% | 0.5 | 93 | 1.55 | 93% |
| 7 | 100 | 3 | 3% | 0.3 | 96 | 1.37 | 96% |
| 8 | 100 | 2 | 2% | 0.2 | 98 | 1.23 | 98% |
| 9 | 100 | 1 | 1% | 0.1 | 99 | 1.10 | 99% |
| 10 (하위 10%) | 100 | 1 | 1% | 0.1 | 100 | 1.0 | 100% |
| **전체** | **1,000** | **100** | **10%** | **1.0** | **100** | **1.0** | **100%** |

### 4.3 계산 과정 상세

#### Decile 1 계산:

**1) 반응률:**
```
반응률 = 35명 / 100명 = 0.35 = 35%
```

**2) Lift:**
```
Lift = 35% / 10% = 3.5
```
→ 상위 10%를 타겟팅하면 무작위 대비 **3.5배** 효과적

**3) %Captured Response:**
```
%Captured = 35명 / 100명(전체 반응자) = 35%
```
→ 상위 10%만 타겟팅해도 전체 반응자의 **35%** 포착

#### Decile 1~3 (상위 30%) 누적 계산:

**1) 누적 반응자:**
```
35 + 20 + 15 = 70명
```

**2) 누적 반응률:**
```
70명 / 300명 = 0.233 = 23.3%
```

**3) Cumulative Lift:**
```
Cum_Lift = 23.3% / 10% = 2.33
```

**4) Cumulative %Captured:**
```
%Captured = 70명 / 100명 = 70%
```
→ 상위 30%를 타겟팅하면 전체 반응자의 **70%** 포착

### 4.4 실전 문제

**문제:**
다음 데이터에서 상위 20%를 타겟팅할 때의 Lift와 %Captured Response를 구하시오.

| Decile | 고객 수 | 반응자 |
|--------|---------|--------|
| 1 | 500 | 150 |
| 2 | 500 | 100 |
| 3 | 500 | 75 |
| 4 | 500 | 50 |
| 5 | 500 | 25 |
| **전체** | **2,500** | **400** |

**풀이:**

**Step 1: 전체 반응률**
```
전체 반응률 = 400 / 2,500 = 0.16 = 16%
```

**Step 2: 상위 20% = Decile 1 (500명)**

**Lift:**
```
Decile 1 반응률 = 150 / 500 = 30%
Lift = 30% / 16% = 1.875
```

**%Captured Response:**
```
%Captured = 150 / 400 = 37.5%
```

**답:**
- **Lift = 1.875** (무작위 대비 1.875배 효과)
- **%Captured Response = 37.5%** (전체 반응자의 37.5% 포착)

### 4.5 Gain 계산 예제

**전체 데이터:** 1,000명, 반응자 100명

| 누적 % | 누적 고객 | 누적 반응자 | Gain (%) |
|--------|----------|------------|----------|
| 10% | 100 | 35 | 35% |
| 20% | 200 | 55 | 55% |
| 30% | 300 | 70 | 70% |
| 40% | 400 | 80 | 80% |
| 50% | 500 | 88 | 88% |
| 60% | 600 | 93 | 93% |
| 70% | 700 | 96 | 96% |
| 80% | 800 | 98 | 98% |
| 90% | 900 | 99 | 99% |
| 100% | 1,000 | 100 | 100% |

**해석:**
- 상위 10% 타겟팅 → 전체 반응자의 35% 포착
- 상위 30% 타겟팅 → 전체 반응자의 70% 포착
- 상위 50% 타겟팅 → 전체 반응자의 88% 포착

---

## 5. 실전 응용

### 5.1 타겟 마케팅 의사결정

**시나리오:**
- 총 고객: 100,000명
- 예상 반응자: 5,000명 (5%)
- DM 비용: 500원/통
- 구매당 이익: 20,000원

**Decile 분석 결과:**

| Decile | Lift | %Captured | 타겟 고객 | 예상 반응 | 비용 | 수익 | 순이익 |
|--------|------|-----------|---------|----------|------|------|--------|
| 1 | 4.0 | 40% | 10,000 | 2,000 | 500만 | 4,000만 | 3,500만 |
| 1~2 | 3.0 | 60% | 20,000 | 3,000 | 1,000만 | 6,000만 | 5,000만 |
| 1~3 | 2.5 | 75% | 30,000 | 3,750 | 1,500만 | 7,500만 | 6,000만 |
| 1~4 | 2.0 | 85% | 40,000 | 4,250 | 2,000만 | 8,500만 | 6,500만 |
| 1~5 | 1.6 | 90% | 50,000 | 4,500 | 2,500만 | 9,000만 | 6,500만 |

**최적 의사결정:**
- **상위 40% 타겟팅** → 순이익 최대 (6,500만원)
- Decile 1~4만 발송
- 60% 비용으로 85% 효과

### 5.2 ROI 분석

**ROI (Return on Investment) 계산:**
```
ROI = (수익 - 비용) / 비용 × 100%
```

| 타겟 범위 | 비용 | 수익 | ROI |
|---------|------|------|-----|
| 상위 10% | 500만 | 4,000만 | 700% |
| 상위 20% | 1,000만 | 6,000만 | 500% |
| 상위 30% | 1,500만 | 7,500만 | 400% |
| 상위 40% | 2,000만 | 8,500만 | 325% |
| 상위 50% | 2,500만 | 9,000만 | 260% |
| 전체 100% | 5,000만 | 10,000만 | 100% |

**인사이트:**
- 상위 10%의 ROI가 가장 높음 (700%)
- 하지만 절대 순이익은 상위 40%가 최대

### 5.3 모델 비교

**두 모델의 성능 비교:**

| Decile | 모델 A Lift | 모델 B Lift | 우세 모델 |
|--------|------------|------------|----------|
| 1 | 4.5 | 4.0 | A |
| 2 | 3.0 | 3.5 | B |
| 3 | 2.0 | 2.5 | B |
| 4 | 1.5 | 1.8 | B |
| 5 | 1.0 | 1.2 | B |

**분석:**
- **모델 A**: 상위 10%에 집중 (정밀 타겟팅)
- **모델 B**: 상위 50%까지 골고루 (광범위 타겟팅)

**선택 기준:**
- 소규모 캠페인 → 모델 A
- 대규모 캠페인 → 모델 B

### 5.4 비즈니스 케이스

**케이스 1: 신용카드 발급**
```
목표: 연회비 카드 발급 극대화
전략: 상위 20% 타겟 (Lift=3.0, Captured=50%)
결과: 발급 비용 80% 절감, 발급률 50% 달성
```

**케이스 2: 해지 방지 캠페인**
```
목표: 고위험 고객 이탈 방지
전략: 상위 30% 타겟 (Lift=2.5, Captured=65%)
결과: 전체 이탈자의 65% 사전 포착
```

**케이스 3: 상품 추천**
```
목표: 교차 판매 증대
전략: 상위 40% 타겟 (Lift=2.0, Captured=75%)
결과: 추천 정확도 2배 향상
```

---

## 6. Python 구현

### 6.1 기본 Lift & Gain 계산

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def calculate_lift_table(y_true, y_pred_proba, n_bins=10):
    """
    Lift 테이블 계산
    
    Parameters:
    - y_true: 실제 타겟 (0 or 1)
    - y_pred_proba: 예측 확률
    - n_bins: 분위 수 (기본 10분위)
    
    Returns:
    - DataFrame: Lift 분석 테이블
    """
    # 데이터프레임 생성
    df = pd.DataFrame({
        'actual': y_true,
        'predicted_prob': y_pred_proba
    })
    
    # 예측 확률 기준 내림차순 정렬
    df = df.sort_values('predicted_prob', ascending=False).reset_index(drop=True)
    
    # 분위 할당
    df['decile'] = pd.qcut(df.index, q=n_bins, labels=False, duplicates='drop') + 1
    
    # 전체 통계
    total_count = len(df)
    total_responders = df['actual'].sum()
    overall_response_rate = total_responders / total_count
    
    # 분위별 집계
    lift_table = df.groupby('decile').agg({
        'actual': ['count', 'sum']
    }).reset_index()
    
    lift_table.columns = ['Decile', 'Count', 'Responders']
    
    # 반응률 계산
    lift_table['Response_Rate'] = lift_table['Responders'] / lift_table['Count']
    
    # Lift 계산
    lift_table['Lift'] = lift_table['Response_Rate'] / overall_response_rate
    
    # 누적 계산
    lift_table['Cum_Responders'] = lift_table['Responders'].cumsum()
    lift_table['Cum_Count'] = lift_table['Count'].cumsum()
    lift_table['Cum_Response_Rate'] = lift_table['Cum_Responders'] / lift_table['Cum_Count']
    lift_table['Cum_Lift'] = lift_table['Cum_Response_Rate'] / overall_response_rate
    
    # %Captured Response
    lift_table['Pct_Captured'] = (lift_table['Cum_Responders'] / total_responders) * 100
    
    # %Population
    lift_table['Pct_Population'] = (lift_table['Cum_Count'] / total_count) * 100
    
    return lift_table

# 예제 데이터 생성
np.random.seed(42)
n_samples = 1000

# 예측 확률 생성 (실제와 어느 정도 상관)
y_true = np.random.binomial(1, 0.1, n_samples)
y_pred_proba = np.where(y_true == 1, 
                        np.random.beta(5, 2, n_samples),  # 반응자는 높은 확률
                        np.random.beta(2, 5, n_samples))  # 비반응자는 낮은 확률

# Lift 테이블 계산
lift_table = calculate_lift_table(y_true, y_pred_proba, n_bins=10)

print("=== Lift Analysis Table ===")
print(lift_table.to_string(index=False))

# 요약 통계
print("\n=== Summary ===")
print(f"Total Samples: {len(y_true)}")
print(f"Total Responders: {y_true.sum()}")
print(f"Overall Response Rate: {y_true.mean():.2%}")
print(f"Top 10% Lift: {lift_table.iloc[0]['Lift']:.2f}")
print(f"Top 20% Captured Response: {lift_table.iloc[1]['Pct_Captured']:.2f}%")
```

### 6.2 Lift Chart 시각화

```python
def plot_lift_chart(lift_table):
    """Lift Chart 그리기"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Decile Lift Chart
    ax1.bar(lift_table['Decile'], lift_table['Lift'], 
            color='steelblue', edgecolor='black')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax1.set_xlabel('Decile', fontsize=12)
    ax1.set_ylabel('Lift', fontsize=12)
    ax1.set_title('Decile Lift Chart', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for i, (decile, lift) in enumerate(zip(lift_table['Decile'], lift_table['Lift'])):
        ax1.text(decile, lift + 0.1, f'{lift:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 2. Cumulative Lift Chart
    ax2.plot(lift_table['Pct_Population'], lift_table['Cum_Lift'], 
             marker='o', linewidth=2, markersize=6, label='Model')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_xlabel('% of Population', fontsize=12)
    ax2.set_ylabel('Cumulative Lift', fontsize=12)
    ax2.set_title('Cumulative Lift Chart', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lift_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_lift_chart(lift_table)
```

### 6.3 Gain Chart 시각화

```python
def plot_gain_chart(lift_table):
    """Gain Chart (Cumulative Response) 그리기"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 모델 곡선
    ax.plot(lift_table['Pct_Population'], lift_table['Pct_Captured'],
            marker='o', linewidth=2.5, markersize=8, 
            color='blue', label='Model')
    
    # 기준선 (Random)
    ax.plot([0, 100], [0, 100], 
            linestyle='--', linewidth=2, color='red', label='Random')
    
    # 이상적 모델 (Perfect Model)
    first_100_pct = lift_table.iloc[0]['Pct_Population']
    ax.plot([0, first_100_pct, 100], [0, 100, 100],
            linestyle=':', linewidth=2, color='green', label='Perfect Model')
    
    # 채우기 (모델과 랜덤 사이)
    ax.fill_between(lift_table['Pct_Population'], 
                    lift_table['Pct_Captured'],
                    lift_table['Pct_Population'],
                    alpha=0.2, color='blue')
    
    ax.set_xlabel('% of Population Targeted', fontsize=12)
    ax.set_ylabel('% of Total Responders Captured', fontsize=12)
    ax.set_title('Gain Chart (Cumulative Response)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # 주요 지점 표시
    for i in [0, 1, 2, 4, 9]:  # 10%, 20%, 30%, 50%, 100%
        pct_pop = lift_table.iloc[i]['Pct_Population']
        pct_cap = lift_table.iloc[i]['Pct_Captured']
        ax.plot(pct_pop, pct_cap, 'ro', markersize=8)
        ax.annotate(f'{pct_cap:.1f}%', 
                   xy=(pct_pop, pct_cap),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gain_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_gain_chart(lift_table)
```

### 6.4 종합 대시보드

```python
def create_lift_dashboard(y_true, y_pred_proba):
    """종합 Lift 분석 대시보드"""
    lift_table = calculate_lift_table(y_true, y_pred_proba)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Decile Lift
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(lift_table['Decile'], lift_table['Lift'], color='steelblue')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
    ax1.set_title('Decile Lift', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Decile')
    ax1.set_ylabel('Lift')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Cumulative Lift
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(lift_table['Pct_Population'], lift_table['Cum_Lift'],
            marker='o', linewidth=2, color='green')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Cumulative Lift', fontsize=13, fontweight='bold')
    ax2.set_xlabel('% Population')
    ax2.set_ylabel('Cumulative Lift')
    ax2.grid(True, alpha=0.3)
    
    # 3. Gain Chart
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(lift_table['Pct_Population'], lift_table['Pct_Captured'],
            marker='o', linewidth=2.5, color='blue', label='Model')
    ax3.plot([0, 100], [0, 100], '--', linewidth=2, color='red', label='Random')
    ax3.fill_between(lift_table['Pct_Population'],
                    lift_table['Pct_Captured'],
                    lift_table['Pct_Population'],
                    alpha=0.2, color='blue')
    ax3.set_title('Gain Chart', fontsize=13, fontweight='bold')
    ax3.set_xlabel('% Population')
    ax3.set_ylabel('% Captured Response')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Response Rate by Decile
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(lift_table['Decile'], lift_table['Response_Rate'] * 100,
           color='coral')
    ax4.axhline(y=y_true.mean() * 100, color='red', linestyle='--',
               linewidth=2, label='Overall Rate')
    ax4.set_title('Response Rate by Decile', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Decile')
    ax4.set_ylabel('Response Rate (%)')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Lift Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('lift_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

create_lift_dashboard(y_true, y_pred_proba)
```

### 6.5 scikit-learn 모델과 통합

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# 데이터 생성
X, y = make_classification(n_samples=10000, n_features=20, 
                          n_informative=15, n_redundant=5,
                          n_classes=2, weights=[0.9, 0.1],
                          random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 확률
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Lift 분석
lift_table = calculate_lift_table(y_test, y_pred_proba)

print("\n=== Model Performance ===")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Top 10% Lift: {lift_table.iloc[0]['Lift']:.2f}")
print(f"Top 20% Captured: {lift_table.iloc[1]['Pct_Captured']:.2f}%")
print(f"Top 30% Captured: {lift_table.iloc[2]['Pct_Captured']:.2f}%")

# 시각화
create_lift_dashboard(y_test, y_pred_proba)
```

---

## 7. 시험 대비

### 7.1 핵심 개념 정리

| 개념 | 공식 | 의미 |
|------|------|------|
| **Lift** | (구간 반응률) / (전체 반응률) | 무작위 대비 개선도 |
| **Cumulative Lift** | (누적 반응률) / (전체 반응률) | 상위 N% 효과 |
| **%Captured Response** | (포착 반응자) / (전체 반응자) × 100 | 반응자 포착률 |
| **Gain** | Captured Response와 동일 | 누적 포착률 |

### 7.2 계산 공식

```
┌─────────────────────────────────────┐
│ Lift 계산                            │
│ Lift = (구간 반응자 / 구간 인원)      │
│        ─────────────────────        │
│        (전체 반응자 / 전체 인원)      │
│                                     │
│ Cumulative Lift                     │
│ Cum_Lift = (누적 반응자 / 누적 인원) │
│            ───────────────────      │
│            (전체 반응자 / 전체 인원)  │
│                                     │
│ %Captured Response                  │
│ %Cap = 누적 반응자 / 전체 반응자 ×100│
└─────────────────────────────────────┘
```

### 7.3 시험 문제 유형

#### 유형 1: Lift 계산

**문제:**
전체 1,000명 중 100명이 반응. 상위 100명 중 40명이 반응했다면 Lift는?

**풀이:**
```
전체 반응률 = 100/1,000 = 10%
상위 10% 반응률 = 40/100 = 40%
Lift = 40% / 10% = 4.0
```

**답: 4.0**

#### 유형 2: %Captured Response

**문제:**
전체 반응자 200명. 상위 30%를 타겟팅했을 때 150명 포착. %Captured는?

**풀이:**
```
%Captured = 150 / 200 × 100 = 75%
```

**답: 75%**

#### 유형 3: 종합 분석

**문제:**
다음 테이블에서 상위 20%의 Cumulative Lift와 %Captured를 구하시오.

| Decile | 인원 | 반응자 |
|--------|------|--------|
| 1 | 100 | 30 |
| 2 | 100 | 20 |
| 전체 | 1,000 | 100 |

**풀이:**
```
전체 반응률 = 100/1,000 = 10%

상위 20% (200명):
누적 반응자 = 30 + 20 = 50
누적 반응률 = 50/200 = 25%

Cumulative Lift = 25% / 10% = 2.5
%Captured = 50/100 × 100 = 50%
```

**답: Cum_Lift = 2.5, %Captured = 50%**

### 7.4 자주 틀리는 포인트

| 실수 | 올바른 방법 |
|------|-----------|
| Lift와 %Captured 혼동 | Lift는 비율, %Captured는 백분율 |
| 분모 착각 | Lift 분모는 전체 반응률 |
| 누적 계산 실수 | 상위부터 차례로 더하기 |
| 단위 착각 | %를 소수로 또는 그 반대 |

### 7.5 실전 체크리스트

**Lift 계산 시:**
- [ ] 전체 반응률 먼저 계산
- [ ] 구간 반응률 계산
- [ ] 나눗셈으로 Lift 도출
- [ ] Lift ≥ 1.0인지 확인

**%Captured 계산 시:**
- [ ] 전체 반응자 수 확인
- [ ] 누적 반응자 수 계산
- [ ] 백분율로 환산
- [ ] 0~100% 범위 확인

### 7.6 해석 가이드

**Lift 해석:**
- Lift = 3.0: 무작위보다 3배 좋음
- Lift = 1.0: 무작위와 같음
- Lift = 0.5: 무작위보다 나쁨

**%Captured 해석:**
- 30%: 상위 N%로 전체의 30% 포착
- 높을수록 효율적
- 100%: 모든 반응자 포착

**비즈니스 의사결정:**
1. Lift 높은 구간 우선 타겟
2. ROI 고려하여 최적점 찾기
3. %Captured로 목표 달성도 평가

---

## 요약

### 핵심 포인트

1. **Lift는 모델 효과성의 핵심 지표**
   - 무작위 대비 몇 배 효과적인지
   - Decile별, Cumulative 두 가지

2. **%Captured Response는 포괄성 지표**
   - 전체 반응자 중 몇 % 포착
   - Gain과 같은 개념

3. **Gain Chart로 시각적 평가**
   - 모델 곡선 vs 기준선
   - 곡선 아래 면적이 클수록 좋음

4. **비즈니스 의사결정에 직접 활용**
   - 타겟 범위 결정
   - ROI 최적화
   - 캠페인 효율성 극대화

### 시험 전 최종 점검

- [ ] Lift 공식 완벽히 암기
- [ ] %Captured 계산법 숙지
- [ ] 누적 계산 연습
- [ ] Gain Chart 해석 능력
- [ ] 비즈니스 응용 이해
- [ ] 손으로 계산 연습

---

**작성일:** 2026년 1월  
**용도:** 빅데이터분석기사 Lift & Gain 분석 완벽 대비  
**참고:** 타겟 마케팅, 모델 평가, ROI 분석
