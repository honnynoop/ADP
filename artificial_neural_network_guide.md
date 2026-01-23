# 인공신경망 분석 완벽 가이드

## 목차
1. [인공신경망 기본 개념](#1-인공신경망-기본-개념)
2. [퍼셉트론](#2-퍼셉트론)
3. [다층 퍼셉트론 (MLP)](#3-다층-퍼셉트론-mlp)
4. [활성화 함수](#4-활성화-함수)
5. [순전파와 역전파](#5-순전파와-역전파)
6. [손실 함수](#6-손실-함수)
7. [최적화 알고리즘](#7-최적화-알고리즘)
8. [정규화 기법](#8-정규화-기법)
9. [주요 용어 정리](#9-주요-용어-정리)
10. [실전 구현](#10-실전-구현)

---

## 1. 인공신경망 기본 개념

### 1.1 정의
**인공신경망(Artificial Neural Network, ANN)**은 인간 뇌의 뉴런 구조를 모방한 기계학습 모델로, 입력층, 은닉층, 출력층으로 구성된 네트워크입니다.

### 1.2 생물학적 뉴런 vs 인공 뉴런

```
[생물학적 뉴런]
수상돌기 → 세포체 → 축삭돌기 → 시냅스 → 다음 뉴런

[인공 뉴런]
입력(x) → 가중합(Σw·x) → 활성화함수(f) → 출력(y)
```

| 구성요소 | 생물학적 뉴런 | 인공 뉴런 |
|---------|-------------|----------|
| 입력 | 수상돌기 | 입력값 (x₁, x₂, ...) |
| 처리 | 세포체 | 가중합 + 편향 (Σwᵢxᵢ + b) |
| 활성화 | 역치 | 활성화 함수 (sigmoid, ReLU 등) |
| 출력 | 축삭돌기 | 출력값 (y) |
| 연결 | 시냅스 | 가중치 (w) |

### 1.3 인공신경망의 구조

```
입력층        은닉층1       은닉층2       출력층
(Input)     (Hidden 1)   (Hidden 2)   (Output)

  x₁  ●─────●─────●─────●─────● y₁
            │╲   ╱│╲   ╱│
  x₂  ●─────●─●─●─●─●─●─●
            │╱   ╲│╱   ╲│
  x₃  ●─────●─────●─────●─────● y₂
            │     │     │
  x₄  ●─────●─────●─────●

      ↓       ↓       ↓       ↓
    입력    특징추출  특징추출   예측
```

---

## 2. 퍼셉트론

### 2.1 단일 퍼셉트론 (Single Perceptron)

**구조**:
```
입력: x₁, x₂, ..., xₙ
가중치: w₁, w₂, ..., wₙ
편향: b

y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
  = f(Σwᵢxᵢ + b)
  = f(w·x + b)
```

**다이어그램**:
```
     x₁ ──w₁──┐
               │
     x₂──w₂───┤
               ├─→ Σ ─→ f(z) ─→ y
     x₃──w₃───┤
               │
     b ────────┘
```

### 2.2 퍼셉트론의 한계

| 문제 | 설명 | 해결 방법 |
|------|------|----------|
| **선형 분리 가능성** | XOR 문제 해결 불가 | 다층 퍼셉트론 (MLP) |
| **단순한 결정 경계** | 직선으로만 분류 | 비선형 활성화 함수 |
| **표현력 제한** | 복잡한 패턴 학습 불가 | 은닉층 추가 |

### 2.3 XOR 문제

```
단일 퍼셉트론으로 불가능:
  x₁ | x₂ | XOR
  ---|----|----
   0 |  0 |  0
   0 |  1 |  1
   1 |  0 |  1
   1 |  1 |  0

시각화:
  x₂ ^
   1 | 1 (출력=1)    0 (출력=0)
     |
   0 | 0 (출력=0)    1 (출력=1)
     +-------------------> x₁
         0              1

→ 하나의 직선으로 분리 불가능!
```

---

## 3. 다층 퍼셉트론 (MLP)

### 3.1 MLP 구조

**정의**: 입력층과 출력층 사이에 하나 이상의 은닉층을 가진 신경망

```
상세 구조:

Layer 0      Layer 1         Layer 2         Layer 3
(입력층)     (은닉층 1)      (은닉층 2)      (출력층)

x₁ ●────┐   
         ├──● h₁₁ ────┐
x₂ ●────┤             ├──● h₂₁ ────┐
         ├──● h₁₂ ────┤             ├──● ŷ₁
x₃ ●────┤             ├──● h₂₂ ────┤
         ├──● h₁₃ ────┤             ├──● ŷ₂
x₄ ●────┘             ├──● h₂₃ ────┘
                      ┘

각 연결선은 가중치(weight)를 나타냄
각 노드는 활성화 함수를 적용
```

### 3.2 MLP 특징 비교

| 특성 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **은닉층 개수** | 깊이(depth) | 복잡한 패턴 학습 | 과적합 위험, 학습 어려움 |
| **노드 개수** | 너비(width) | 표현력 증가 | 계산 비용 증가 |
| **활성화 함수** | 비선형성 부여 | XOR 등 비선형 문제 해결 | 기울기 소실 문제 |
| **가중치 초기화** | 학습 시작점 | 수렴 속도 향상 | 잘못된 초기화 시 학습 실패 |

### 3.3 층별 역할

```
입력층 (Input Layer)
  ↓ 원시 데이터를 받음
  
은닉층 1 (Hidden Layer 1)
  ↓ 저수준 특징 추출 (edges, corners)
  
은닉층 2 (Hidden Layer 2)
  ↓ 중수준 특징 추출 (textures, patterns)
  
은닉층 3 (Hidden Layer 3)
  ↓ 고수준 특징 추출 (objects, concepts)
  
출력층 (Output Layer)
  ↓ 최종 예측/분류
```

---

## 4. 활성화 함수

### 4.1 주요 활성화 함수 비교표

| 함수 | 수식 | 범위 | 미분 | 장점 | 단점 | 사용처 |
|------|------|------|------|------|------|--------|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | (0, 1) | σ(x)(1-σ(x)) | 확률 해석 가능 | 기울기 소실 | 출력층 (이진분류) |
| **Tanh** | tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | 1-tanh²(x) | 0 중심, Sigmoid보다 나음 | 기울기 소실 | 은닉층 |
| **ReLU** | f(x) = max(0, x) | [0, ∞) | 1 (x>0), 0 (x≤0) | 계산 빠름, 기울기 소실 완화 | Dead ReLU | 은닉층 (기본) |
| **Leaky ReLU** | f(x) = max(αx, x) | (-∞, ∞) | α (x<0), 1 (x≥0) | Dead ReLU 해결 | α 선택 필요 | 은닉층 |
| **ELU** | x (x>0), α(eˣ-1) (x≤0) | (-α, ∞) | 1 (x>0), f(x)+α (x≤0) | 평균 0에 가까움 | 계산 비용 | 은닉층 |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1), Σ=1 | - | 다중 클래스 확률 | - | 출력층 (다중분류) |

### 4.2 활성화 함수 시각화

```
Sigmoid:           Tanh:              ReLU:
  1 |    ╭──         1 |    ╭──         |    ╱
    |   ╱              |   ╱              |   ╱
0.5 |  ╱             0 |  ╱             0 |  ╱
    | ╱                | ╱                | ╱────
  0 |╱──             -1|╱──               |╱
    ───────x           ───────x           ───────x
   -5  0  5          -3  0  3           -3  0  3

Leaky ReLU:        ELU:               Softmax:
    |    ╱             |    ╱           (다중 출력)
    |   ╱              |   ╱            P₁ ─┐
  0 |  ╱             0 |  ╱             P₂ ──┼─ Σ = 1
    | ╱                | ╱╮              P₃ ─┘
    |╱─────            |╱──╮            모든 Pᵢ ∈ (0,1)
    ───────x           ───────x
```

### 4.3 활성화 함수 선택 가이드

```
선택 흐름도:

시작
 ↓
출력층인가?
 ├─ YES → 이진분류? ─ YES → Sigmoid
 │         └─ NO → 다중분류? ─ YES → Softmax
 │                  └─ NO → 회귀 → Linear (활성화 없음)
 │
 └─ NO (은닉층)
     ↓
     기본: ReLU
     ↓
     문제 있음?
     ├─ Dead ReLU → Leaky ReLU / ELU
     ├─ 기울기 소실 → ReLU / Leaky ReLU
     └─ 음수 출력 필요 → Tanh / ELU
```

---

## 5. 순전파와 역전파

### 5.1 순전파 (Forward Propagation)

**과정**: 입력 데이터가 네트워크를 통과하여 예측값을 생성

```
수식:
Layer l의 출력:
  z⁽ˡ⁾ = W⁽ˡ⁾·a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
  a⁽ˡ⁾ = f(z⁽ˡ⁾)

여기서:
- a⁽ˡ⁾: l층의 활성화 출력
- W⁽ˡ⁾: l층의 가중치 행렬
- b⁽ˡ⁾: l층의 편향 벡터
- f: 활성화 함수
```

**예시**:
```
입력층 → 은닉층1 → 은닉층2 → 출력층

x = [1, 2, 3]ᵀ

Step 1: 은닉층1
  z₁ = W₁·x + b₁
  a₁ = ReLU(z₁)

Step 2: 은닉층2
  z₂ = W₂·a₁ + b₂
  a₂ = ReLU(z₂)

Step 3: 출력층
  z₃ = W₃·a₂ + b₃
  ŷ = Sigmoid(z₃)
```

### 5.2 역전파 (Backpropagation)

**과정**: 출력층부터 입력층 방향으로 오차를 전파하며 가중치 업데이트

```
핵심 아이디어: 연쇄 법칙 (Chain Rule)

∂L/∂W⁽ˡ⁾ = ∂L/∂a⁽ˡ⁾ · ∂a⁽ˡ⁾/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾

역전파 흐름:
출력층 손실
    ↓ δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾
은닉층 L-1
    ↓ δ⁽ᴸ⁻¹⁾ = (W⁽ᴸ⁾)ᵀ·δ⁽ᴸ⁾ ⊙ f'(z⁽ᴸ⁻¹⁾)
은닉층 L-2
    ↓ δ⁽ᴸ⁻²⁾ = (W⁽ᴸ⁻¹⁾)ᵀ·δ⁽ᴸ⁻¹⁾ ⊙ f'(z⁽ᴸ⁻²⁾)
...
입력층
```

### 5.3 순전파 vs 역전파 비교

| 구분 | 순전파 | 역전파 |
|------|--------|--------|
| **방향** | 입력 → 출력 | 출력 → 입력 |
| **목적** | 예측값 계산 | 가중치 업데이트 |
| **계산** | 가중합 + 활성화 | 기울기 계산 |
| **사용 데이터** | 입력 데이터 x | 손실 함수 기울기 |
| **수식** | z = W·a + b, a = f(z) | δ = ∂L/∂z |

### 5.4 역전파 알고리즘 단계

```python
# 의사코드
def backpropagation(X, y, W, b):
    # 1. 순전파: 예측값 계산
    a = [X]  # 활성화 값 저장
    z = []   # 가중합 저장
    
    for l in range(L):
        z_l = W[l] @ a[l] + b[l]
        a_l = activation(z_l)
        z.append(z_l)
        a.append(a_l)
    
    # 2. 손실 계산
    loss = compute_loss(a[-1], y)
    
    # 3. 역전파: 기울기 계산
    delta = [None] * L
    dW = [None] * L
    db = [None] * L
    
    # 출력층
    delta[-1] = a[-1] - y  # 교차 엔트로피 + Softmax
    
    # 은닉층 (역방향)
    for l in range(L-2, -1, -1):
        delta[l] = (W[l+1].T @ delta[l+1]) * activation_derivative(z[l])
    
    # 4. 가중치 기울기
    for l in range(L):
        dW[l] = delta[l] @ a[l].T / m  # m: 배치 크기
        db[l] = np.sum(delta[l], axis=1) / m
    
    # 5. 가중치 업데이트
    for l in range(L):
        W[l] -= learning_rate * dW[l]
        b[l] -= learning_rate * db[l]
    
    return W, b, loss
```

---

## 6. 손실 함수

### 6.1 주요 손실 함수 비교

| 손실 함수 | 수식 | 미분 | 문제 유형 | 특징 |
|----------|------|------|----------|------|
| **MSE** (평균제곱오차) | L = (1/n)Σ(yᵢ-ŷᵢ)² | 2(ŷ-y) | 회귀 | 이상치에 민감 |
| **MAE** (평균절대오차) | L = (1/n)Σ\|yᵢ-ŷᵢ\| | sign(ŷ-y) | 회귀 | 이상치에 강건 |
| **Binary Cross-Entropy** | L = -Σ(y·log(ŷ)+(1-y)·log(1-ŷ)) | ŷ-y | 이진분류 | Sigmoid 출력 |
| **Categorical Cross-Entropy** | L = -Σyᵢ·log(ŷᵢ) | ŷ-y | 다중분류 | Softmax 출력 |
| **Sparse Categorical CE** | L = -log(ŷy) | ŷ-y | 다중분류 | 정수 레이블 |
| **Huber Loss** | L = {½(y-ŷ)² if \|y-ŷ\|≤δ, δ(\|y-ŷ\|-½δ) otherwise} | - | 회귀 | MSE+MAE 결합 |

### 6.2 손실 함수 선택 가이드

```
문제 유형별 선택:

회귀 문제
├─ 이상치 많음? ─ YES → MAE / Huber Loss
└─ NO → MSE

분류 문제
├─ 이진 분류 → Binary Cross-Entropy
│   └─ 출력층: Sigmoid
│
└─ 다중 분류 → Categorical Cross-Entropy
    ├─ One-hot 레이블 → Categorical CE
    └─ 정수 레이블 → Sparse Categorical CE
    └─ 출력층: Softmax
```

### 6.3 손실 함수 시각화

```
MSE vs MAE:

Loss ^                  MSE: 제곱
     |                    /
     |                  /
     |                /
     |              /
     |           _/
     |        _/
     |   ___/_____ MAE: 절대값
     |_______________> Error
     0

MSE는 이상치에 큰 페널티 부여
MAE는 모든 오차에 동일한 비율로 페널티
```

---

## 7. 최적화 알고리즘

### 7.1 주요 최적화 알고리즘 비교

| 알고리즘 | 수식 | 학습률 | 메모리 | 특징 | 사용 추천 |
|---------|------|--------|--------|------|----------|
| **SGD** | θ = θ - η∇L | 고정 | 낮음 | 간단, 느림 | 작은 데이터셋 |
| **Momentum** | v = βv + ∇L<br>θ = θ - ηv | 고정 | 낮음 | 관성 추가, SGD보다 빠름 | 일반적 |
| **AdaGrad** | θ = θ - η/√(G+ε)·∇L | 적응적 감소 | 중간 | 희소 데이터에 좋음 | NLP |
| **RMSProp** | θ = θ - η/√(E[g²]+ε)·∇L | 적응적 | 중간 | AdaGrad 개선 | RNN |
| **Adam** | θ = θ - η·m̂/(√v̂+ε) | 적응적 | 높음 | Momentum + RMSProp | **기본 추천** |
| **AdamW** | Adam + Weight Decay | 적응적 | 높음 | 정규화 개선 | Transformer |

### 7.2 최적화 알고리즘 시각화

```
경사하강법 비교:

손실 곡면                SGD (느림, 진동)
    ^                       /\/\/\
    |         ●목표        /      \
    |          ○          /        \
    |        ╱  ╲        ╱          ╲
    |      ╱      ╲    ●시작점        ●
    |    ╱          ╲
    |__________________>

Momentum (빠름, 관성)      Adam (빠름, 적응적)
    ^                       ^
    |         ●목표            ●목표
    |          ○               ○
    |        ╱  ╲           ╱  ╲
    |      ╱      ╲       ╱      ╲
    |   ●→→→→      ╲   ●─────→   ╲
    |__________╲      ├─────────→
                ╲    /             
```

### 7.3 최적화 하이퍼파라미터

| 파라미터 | SGD | Momentum | Adam |
|---------|-----|----------|------|
| **학습률 (η)** | 0.01~0.1 | 0.01~0.1 | 0.001~0.01 |
| **β (모멘텀)** | - | 0.9 | β₁=0.9, β₂=0.999 |
| **ε** | - | - | 1e-8 |
| **감쇠율** | 선택적 | 선택적 | 선택적 |

### 7.4 학습률 스케줄링

```python
# 주요 학습률 스케줄링 기법

# 1. Step Decay
lr = initial_lr * (decay_rate ** (epoch // step_size))

# 2. Exponential Decay
lr = initial_lr * (decay_rate ** epoch)

# 3. Cosine Annealing
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * epoch / max_epoch))

# 4. Warm-up then Decay
if epoch < warmup_epochs:
    lr = initial_lr * (epoch / warmup_epochs)
else:
    lr = initial_lr * decay_function(epoch - warmup_epochs)
```

---

## 8. 정규화 기법

### 8.1 주요 정규화 기법 비교

| 기법 | 목적 | 적용 위치 | 수식/방법 | 효과 |
|------|------|----------|----------|------|
| **L1 정규화** | 희소성 | 손실 함수 | L = L₀ + λΣ\|wᵢ\| | 일부 가중치를 0으로 |
| **L2 정규화** | 가중치 크기 제한 | 손실 함수 | L = L₀ + λΣwᵢ² | 가중치 감소 |
| **Dropout** | 과적합 방지 | 은닉층 | 노드 p 확률로 제거 | 앙상블 효과 |
| **Batch Normalization** | 학습 안정화 | 각 층 출력 | (x-μ)/σ | 내부 공변량 이동 감소 |
| **Layer Normalization** | 학습 안정화 | 각 층 출력 | 배치 독립적 정규화 | RNN에 적합 |
| **Early Stopping** | 과적합 방지 | 학습 과정 | 검증 손실 모니터링 | 최적 에포크 자동 선택 |
| **Data Augmentation** | 데이터 다양성 | 입력 데이터 | 변환, 회전, 노이즈 | 일반화 성능 향상 |

### 8.2 Dropout 시각화

```
학습 시 (Dropout=0.5):

입력층     은닉층        출력층
  ●────────●────────●
  ●────────×(제거)   ●
  ●────────●────────●
  ●────────×(제거)
  ●────────●────────●

테스트 시:
모든 노드 사용, 가중치에 (1-p) 곱함

효과:
- 노드 간 co-adaptation 방지
- 앙상블 효과 (여러 서브네트워크 학습)
```

### 8.3 Batch Normalization

**수식**:
```
입력: x = [x₁, x₂, ..., xₘ]  (미니배치)

1. 평균 계산: μ = (1/m)Σxᵢ
2. 분산 계산: σ² = (1/m)Σ(xᵢ-μ)²
3. 정규화: x̂ᵢ = (xᵢ-μ)/√(σ²+ε)
4. 스케일/시프트: yᵢ = γx̂ᵢ + β

여기서 γ, β는 학습 가능한 파라미터
```

**효과**:
```
Before BN:              After BN:
입력 분포 변화          입력 분포 안정
(Internal Covariate     
Shift)                  

Layer 1 출력 분포       정규화된 분포
    ┌─────┐                ┌──┐
    │     │                │  │
────┴─────┴────  →  ───────┴──┴───────
(불안정)               (평균 0, 분산 1)

→ 더 큰 학습률 사용 가능
→ 초기화에 덜 민감
→ 정규화 효과 (Dropout 대체 가능)
```

### 8.4 정규화 기법 적용 가이드

```
네트워크 구조에 따른 선택:

일반 MLP
├─ 과적합? → Dropout (0.2~0.5)
├─ 학습 불안정? → Batch Normalization
└─ 가중치 크기 문제? → L2 정규화

CNN
├─ Batch Normalization (거의 필수)
└─ Dropout (출력층 근처)

RNN/Transformer
├─ Layer Normalization
└─ Dropout (시간축 주의)
```

---

## 9. 주요 용어 정리

### 9.1 기본 용어

| 용어 | 영문 | 정의 | 예시/설명 |
|------|------|------|----------|
| **뉴런** | Neuron | 신경망의 기본 단위 | 입력 받아 가중합 계산 후 활성화 |
| **가중치** | Weight | 입력의 중요도 | 학습을 통해 업데이트되는 파라미터 |
| **편향** | Bias | 활성화 함수의 이동 | 결정 경계 조정 |
| **층** | Layer | 뉴런의 집합 | 입력층, 은닉층, 출력층 |
| **에포크** | Epoch | 전체 데이터 1회 학습 | 100 에포크 = 전체 데이터 100번 학습 |
| **배치** | Batch | 한 번에 처리하는 데이터 수 | 배치 크기 32 = 한 번에 32개 샘플 |
| **반복** | Iteration | 1 배치 학습 | 1000개 데이터, 배치 100 → 10 반복 |

### 9.2 학습 관련 용어

| 용어 | 영문 | 정의 | 수식/설명 |
|------|------|------|----------|
| **순전파** | Forward Propagation | 입력→출력 계산 | ŷ = f(W·x + b) |
| **역전파** | Backpropagation | 기울기 계산 | ∂L/∂W, ∂L/∂b |
| **경사하강법** | Gradient Descent | 최적화 알고리즘 | θ = θ - η∇L |
| **학습률** | Learning Rate | 업데이트 크기 | η (0.001~0.1) |
| **기울기 소실** | Vanishing Gradient | 기울기 0에 수렴 | 깊은 네트워크에서 발생 |
| **기울기 폭발** | Exploding Gradient | 기울기 발산 | RNN에서 자주 발생 |
| **과적합** | Overfitting | 훈련 데이터에 과도 적합 | 검증 손실 증가 |
| **과소적합** | Underfitting | 학습 부족 | 훈련/검증 손실 모두 높음 |

### 9.3 네트워크 구조 용어

| 용어 | 영문 | 정의 | 특징 |
|------|------|------|------|
| **완전연결층** | Fully Connected Layer | 모든 노드 연결 | Dense Layer, 파라미터 많음 |
| **합성곱층** | Convolutional Layer | 지역적 패턴 추출 | CNN, 이미지 처리 |
| **풀링층** | Pooling Layer | 다운샘플링 | Max Pooling, Average Pooling |
| **순환층** | Recurrent Layer | 시퀀스 데이터 처리 | RNN, LSTM, GRU |
| **임베딩층** | Embedding Layer | 범주형 → 연속형 | 단어 임베딩 |
| **잔차연결** | Residual Connection | 기울기 소실 완화 | x + F(x), ResNet |
| **스킵연결** | Skip Connection | 층 건너뛰기 | U-Net, DenseNet |

### 9.4 성능 평가 용어

| 용어 | 영문 | 정의 | 수식 |
|------|------|------|------|
| **손실** | Loss | 예측 오차 | L(y, ŷ) |
| **정확도** | Accuracy | 정확히 예측한 비율 | (TP+TN)/(TP+TN+FP+FN) |
| **정밀도** | Precision | 양성 예측 중 실제 양성 | TP/(TP+FP) |
| **재현율** | Recall | 실제 양성 중 예측 양성 | TP/(TP+FN) |
| **F1 점수** | F1 Score | 정밀도와 재현율 조화평균 | 2·P·R/(P+R) |
| **혼동행렬** | Confusion Matrix | 예측 결과 요약표 | [[TN,FP],[FN,TP]] |

### 9.5 고급 기법 용어

| 용어 | 영문 | 정의 | 설명 |
|------|------|------|------|
| **전이학습** | Transfer Learning | 사전학습 모델 활용 | ImageNet → 특정 도메인 |
| **미세조정** | Fine-tuning | 사전학습 모델 재학습 | 전체 또는 일부 층 |
| **증류** | Knowledge Distillation | 큰 모델 → 작은 모델 | 교사-학생 학습 |
| **앙상블** | Ensemble | 여러 모델 결합 | Voting, Stacking |
| **하이퍼파라미터** | Hyperparameter | 학습 전 설정 값 | 학습률, 배치 크기, 층 수 |
| **조기종료** | Early Stopping | 검증 손실 기반 중단 | 과적합 방지 |

---

## 10. 실전 구현

### 10.1 간단한 MLP 구현 (NumPy)

```python
import numpy as np

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # 가중치 초기화 (Xavier)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2./hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        # 순전파
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        # 역전파
        # 출력층
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 은닉층
        dz1 = (dz2 @ self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 가중치 업데이트
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            # 순전파
            y_pred = self.forward(X)
            
            # 손실 계산 (Binary Cross-Entropy)
            loss = -np.mean(y * np.log(y_pred + 1e-8) + 
                           (1-y) * np.log(1-y_pred + 1e-8))
            losses.append(loss)
            
            # 역전파 및 업데이트
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses

# 사용 예시
# XOR 문제
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

model = SimpleMLP(input_size=2, hidden_size=4, output_size=1)
losses = model.train(X, y, epochs=5000, learning_rate=0.1)

# 예측
predictions = model.forward(X)
print("\n예측 결과:")
print(predictions)
print("\n이진 예측:")
print((predictions > 0.5).astype(int))
```

### 10.2 Keras/TensorFlow 구현

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sequential API
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Functional API (더 복잡한 구조에 적합)
inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(32, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 콜백 설정
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    )
]

# 학습
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### 10.3 PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 은닉층
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 출력층
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 모델 생성
model = MLP(input_size=784, hidden_sizes=[128, 64], output_size=10)

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # 순전파
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 평가
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
```

### 10.4 하이퍼파라미터 튜닝

```python
# Grid Search 예시
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

def create_model(hidden_units=64, learning_rate=0.001, dropout=0.3):
    model = keras.Sequential([
        layers.Dense(hidden_units, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(dropout),
        layers.Dense(hidden_units//2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 파라미터 그리드
param_grid = {
    'hidden_units': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout': [0.2, 0.3, 0.5],
    'batch_size': [32, 64],
    'epochs': [50]
}

model = KerasClassifier(model=create_model, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

print(f"Best: {grid_result.best_score_:.4f} using {grid_result.best_params_}")
```

---

## 11. 실전 팁 및 체크리스트

### 11.1 모델 설계 체크리스트

```
□ 문제 유형 결정 (회귀/분류/다중분류)
□ 입력 데이터 전처리
  - 정규화/표준화
  - 결측치 처리
  - 범주형 변수 인코딩
□ 네트워크 구조 설계
  - 은닉층 개수 (2-3개로 시작)
  - 각 층의 노드 수 (점진적 감소)
  - 활성화 함수 선택
□ 출력층 설정
  - 노드 수 (클래스 수)
  - 활성화 함수 (Sigmoid/Softmax/Linear)
□ 손실 함수 선택
□ 옵티마이저 선택 (Adam 추천)
□ 정규화 기법 적용
  - Dropout
  - Batch Normalization
  - L2 정규화
□ 콜백 설정
  - Early Stopping
  - Learning Rate Scheduling
```

### 11.2 문제 해결 가이드

| 문제 | 증상 | 해결 방법 |
|------|------|----------|
| **과적합** | 훈련 정확도↑, 검증 정확도↓ | Dropout, L2 정규화, 데이터 증강 |
| **과소적합** | 훈련/검증 정확도 모두 낮음 | 모델 복잡도 증가, 에포크 증가 |
| **기울기 소실** | 학습 안 됨, 손실 감소 없음 | ReLU, Batch Norm, ResNet 구조 |
| **기울기 폭발** | 손실이 NaN | Gradient Clipping, 학습률 감소 |
| **학습 느림** | 수렴까지 오래 걸림 | Adam, Batch Norm, 학습률 증가 |
| **불안정한 학습** | 손실이 진동 | 학습률 감소, Batch 크기 증가 |

### 11.3 성능 개선 전략

```
단계별 접근:

1단계: 베이스라인 구축
  - 간단한 모델 (2-3 은닉층)
  - Adam 옵티마이저
  - 기본 학습률 (0.001)

2단계: 과적합 확인
  - 작은 데이터로 과적합시킴
  - 과적합 가능하면 → 모델 용량 OK
  - 불가능하면 → 모델 복잡도 증가

3단계: 정규화 적용
  - Dropout 추가
  - Batch Normalization
  - Data Augmentation

4단계: 하이퍼파라미터 튜닝
  - 학습률
  - 배치 크기
  - 네트워크 구조

5단계: 앙상블
  - 여러 모델 결합
  - K-Fold 교차검증
```

---

## 12. 부록: 수학적 기초

### 12.1 행렬 연산

```
벡터 내적:
  a·b = Σaᵢbᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ

행렬 곱셈:
  C = A·B
  cᵢⱼ = Σₖ aᵢₖbₖⱼ

전치 행렬:
  (AB)ᵀ = BᵀAᵀ

미분:
  ∂(Wx+b)/∂W = x
  ∂(Wx+b)/∂x = W
```

### 12.2 연쇄 법칙 (Chain Rule)

```
함수 합성: z = f(g(x))

∂z/∂x = (∂z/∂g)·(∂g/∂x)

신경망 예시:
  L = loss(f(Wx + b))
  
∂L/∂W = (∂L/∂f)·(∂f/∂z)·(∂z/∂W)
       = δ·f'(z)·x

여기서:
  z = Wx + b
  f(z) = 활성화 함수
  δ = ∂L/∂f
```

### 12.3 주요 미분 공식

```
Sigmoid:
  σ(x) = 1/(1+e⁻ˣ)
  σ'(x) = σ(x)(1-σ(x))

Tanh:
  tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)
  tanh'(x) = 1 - tanh²(x)

ReLU:
  f(x) = max(0, x)
  f'(x) = {1 if x>0, 0 if x≤0}

Softmax:
  sᵢ = eˣⁱ/Σeˣʲ
  ∂sᵢ/∂xⱼ = sᵢ(δᵢⱼ - sⱼ)
```

---

## 요약

인공신경망은 다음과 같은 핵심 요소로 구성됩니다:

1. **구조**: 입력층, 은닉층, 출력층으로 구성된 계층적 네트워크
2. **학습**: 순전파로 예측, 역전파로 가중치 업데이트
3. **활성화**: 비선형성 부여로 복잡한 패턴 학습
4. **최적화**: 경사하강법 기반 알고리즘으로 손실 최소화
5. **정규화**: 과적합 방지와 일반화 성능 향상

**빅데이터분석기사 시험 핵심 포인트**:
- 퍼셉트론의 한계와 다층 퍼셉트론의 필요성
- 활성화 함수의 종류와 특징
- 순전파와 역전파의 원리
- 주요 최적화 알고리즘 (SGD, Momentum, Adam)
- 과적합 방지 기법 (Dropout, Batch Normalization)
- 손실 함수와 문제 유형의 매칭

---

**시험 준비 화이팅! 🚀**
