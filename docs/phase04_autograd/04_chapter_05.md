# Chapter 5. 하이퍼파라미터와 파라미터 초기화

지금까지 배운 트랜스포머 블록들을 조립하여 전체 모델을 구성하는 단계입니다. 이 챕터는 `microgpt.py`의 241~257줄에 해당합니다. 

## 5-1. 하이퍼파라미터의 의미

하이퍼파라미터(Hyperparameter)는 모델이 스스로 학습하는 값(가중치)이 아니라, **사용자가 모델의 뼈대를 직접 설정해 주는 값**입니다.

```python
# 242~246줄
n_layer = 1      # 트랜스포머 블록을 몇 층으로 쌓을 것인지
n_embd = 16      # 임베딩 차원수 (글자 하나를 16개의 실수로 표현)
block_size = 16  # 컨텍스트 길이 (가장 긴 이름이 15자이므로 16으로 설정)
n_head = 4       # 멀티 헤드 어텐션에서 몇 개의 관점으로 나눌 것인지
head_dim = n_embd // n_head  # 각 헤드가 담당할 차원 수 (16 // 4 = 4)
```

이 값들을 키우면 모델의 "뇌 공간(Capacity)"이 커져서 더 똑똑해질 수 있지만, 학습에 필요한 메모리와 시간이 기하급수적으로 늘어납니다. `microgpt`는 아주 단순한 이름 생성기이므로 극히 작은 값(`n_layer=1`, `n_embd=16`)을 사용합니다. 실제 GPT-3 모델은 `n_layer=96`, `n_embd=12288`을 사용합니다!


## 5-2. 가우시안 랜덤 초기화

신경망 학습은 가중치(`Weight`)를 조정하는 과정입니다. 그러면 이 수많은 가중치들은 최초에 어떤 값을 가지고 시작해야 할까요? 전부 0으로 시작하면 안 될까요?

**전부 0으로 시작할 때의 문제점 (Symmetry Breaking 문제)**:
모든 가중치가 똑같이 0이면, 모든 노드가 똑같은 계산을 하고 역전파 때 똑같은 미분값을 받게 됩니다. 수만 개의 뇌세포가 전부 똑같은 생각 하나만 하게 되어 학습이 붕괴됩니다.

### 랜덤하게, 하지만 아주 작게
```python
# 247줄: 가중치 초기화를 위한 익명 함수(lambda)
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)] 
    for _ in range(nout)
]
```

`random.gauss(0, std)`는 평균이 0이고 표준편차가 `std(0.08)`인 정규 분포(가우시안 분포)에서 난수를 뽑습니다. 즉, 0에 아주 가까운 작은 양수나 음수(`0.03`, `-0.01` 등)를 무작위로 채워 넣습니다.
- **랜덤의 이유**: 각 노드가 서로 다른 생각(파라미터)을 가지게 하여 비대칭성 확보
- **작은 수(0.08)의 이유**: 가중치가 처음부터 너무 크면 출력값이 극단으로 치우쳐 폭발할 수 있음


## 5-3. `state_dict` 구조와 파라미터 조립

### 모델의 모든 뇌세포를 한 바구니에
`state_dict`는 모델의 모든 가중치를 체계적으로 보관하는 딕셔너리입니다. 파이토치(PyTorch)에서도 정확히 이 이름을 사용하여 모델의 상태(가중치)를 저장합니다.

```python
# 248~255줄: 필요한 모든 행렬들을 생성하고 이름을 붙임
state_dict = {
    'wte': matrix(vocab_size, n_embd),     # 토큰 임베딩 (27x16)
    'wpe': matrix(block_size, n_embd),     # 위치 임베딩 (16x16)
    'lm_head': matrix(vocab_size, n_embd)  # 최종 출력 변환 (27x16)
}

# 트랜스포머 블록 내부 가중치들 (n_layer 번 반복)
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd) # Q 가중치
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd) # K 가중치
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd) # V 가중치
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd) # 가중합 투영
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd) # 확장
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd) # 압축
```

### 파라미터 평탄화 (Flatten)
`state_dict` 안에는 2차원 리스트(행렬)들이 들어있습니다. 옵티마이저가 가중치를 업데이트하려면(루프를 돌기 쉽게) 이 복잡한 구조를 모두 풀어서 **자잘한 `Value` 객체들의 긴 1차원 리스트**로 만들어야 합니다.

```python
# 256줄
# 딕셔너리의 값을 꺼내고 → 각 행렬의 행렬을 꺼내고 → 그 안의 Value 노드를 꺼냄
params = [p for mat in state_dict.values() for row in mat for p in row]
```

이렇게 평탄화된 `params` 리스트에는 모델의 모든 학습 가능한 뇌세포(`Value` 객체)가 일렬로 담기게 됩니다. 이제 학습 루프만 돌리면 이 `params`들이 정답을 향해 조금씩 스스로의 값을 조정하게 됩니다!
