# Chapter 6. 리스트 내포와 고급 파이썬 구문

`microgpt.py` 코드를 읽다 보면, 한 줄로 된 복잡한 대괄호 표현식이나 `lambda` 같은 생소한 문법이 자꾸 눈을 가립니다. 다른 언어(Java, C#, JavaScript 등)에서 넘어온 개발자에게 특히 낯설 수 있는 파이썬 고급 표현식들을 이 챕터에서 전부 정리합니다.

## 6-1. 리스트 내포 (List Comprehension)
반복문(`for`)을 한 줄로 압축하여 새로운 리스트를 만들어내는 파이썬의 핵심 문법입니다. `microgpt.py` 전체에서 가장 빈번하게 등장합니다.

```python
# 일반 for문
result = []
for i in range(5):
    result.append(i * 2)
# result = [0, 2, 4, 6, 8]

# 리스트 내포(한 줄로 같은 결과)
result = [i * 2 for i in range(5)]
# result = [0, 2, 4, 6, 8]
```

### 중첩 리스트 내포 (2차원 배열 만들기)
`microgpt.py`에서 가중치 행렬(Weight Matrix)을 만들 때 이중 내포가 사용됩니다.

```python
# 3행 4열의 2차원 배열 (행렬) 생성
matrix = [[0 for col in range(4)] for row in range(3)]
# 결과: [[0,0,0,0], [0,0,0,0], [0,0,0,0]]

# microgpt.py 247번째 줄의 핵심 패턴:
# 바깥 리스트 내포: nout개의 행(row)을 생성
# 안쪽 리스트 내포: 각 행에 nin개의 Value 객체를 생성
matrix = [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]
```
**핵심 포인트**: `_` (언더스코어)는 \"이 변수는 실제로 사용하지 않겠다\"는 파이썬 관례입니다. 단순히 반복 횟수만 필요할 때 사용합니다.


## 6-2. Lambda 함수 (익명 함수)
`def` 없이, 한 줄로 함수를 정의해버리는 문법입니다. 간단한 변환 공식이나 \"팩토리 함수(물건 찍어내는 기계)\"를 만들 때 많이 씁니다.

```python
# 일반 함수 정의
def square(x):
    return x ** 2

# lambda로 동일한 함수를 한 줄로 선언
square = lambda x: x ** 2

print(square(5)) # 25
```

### 💻 `microgpt.py` 적용 (247줄)
```python
# 가중치 행렬을 찍어내는 "공장 함수"
# lambda 인자: nout(행 수), nin(열 수), std(표준편차, 기본값 0.08)
# lambda 본문: 2차원 리스트 내포로 Value 객체 행렬 생성
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)] 
    for _ in range(nout)
]

# 사용 예시: 27개 행(vocab_size) × 16개 열(n_embd)의 행렬 생성
wte = matrix(27, 16)  # matrix(nout=27, nin=16, std=0.08) 과 동일
```
**핵심 포인트**: `lambda`에 기본 인자(`std=0.08`)를 줄 수 있습니다. 호출할 때 `std`를 생략하면 0.08이 자동 적용되어, 모든 가중치가 0 근처의 작은 랜덤 숫자로 초기화됩니다.


## 6-3. 제너레이터 표현식과 내장 함수
리스트 내포와 비슷하지만 대괄호(`[]`) 대신 소괄호(`()`)를 쓰면, 값을 미리 전부 만들지 않고 **필요할 때 하나씩 꺼냅니다(Lazy 평가)**. `sum()` 이나 `max()` 같은 내장 함수와 합치면 메모리 절약 + 간결한 코드가 됩니다.

```python
# 리스트 내포: 메모리에 리스트를 전부 만든 뒤 sum() 호출
total = sum([x * x for x in data])

# 제너레이터 표현식: 리스트를 만들지 않고, 하나씩 꺼내서 바로 더함 (효율적!)
total = sum(x * x for x in data)
```

### 💻 `microgpt.py` 적용
```python
# 275줄: linear 함수 내부의 가중합(Weighted Sum)
# zip(wo, x)로 가중치와 입력값을 쌍으로 묶어 하나씩 곱한 뒤 전부 합산
return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

# 291줄: softmax 내부에서 최댓값 찾기
max_val = max(val.data for val in logits)

# 310줄: rmsnorm 내부의 평균 제곱 계산
ms = sum(xi * xi for xi in x) / len(x)
```

### `zip()` 함수
두 개 이상의 리스트를 **같은 위치끼리 짝을 지어** 동시에 순회합니다.
```python
names = ["a", "b", "c"]
scores = [10, 20, 30]

for name, score in zip(names, scores):
    print(f"{name}: {score}")
# 출력: a:10, b:20, c:30
```
**핵심 포인트**: `microgpt.py`에서 `zip(v._children, v._local_grads)` 형태로, 자식 노드와 그에 대응하는 로컬 미분값을 쌍으로 꺼내는 패턴이 역전파의 핵심입니다.


## 6-4. 딕셔너리(Dict)와 f-string
키(Key)-값(Value) 쌍으로 데이터를 저장하는 자료구조입니다. `state_dict` 라는 이름의 딕셔너리에 모델의 모든 가중치(파라미터)를 보관합니다.

```python
# 딕셔너리 생성
state_dict = {
    'wte': matrix(vocab_size, n_embd),    # 토큰 임베딩 가중치
    'wpe': matrix(block_size, n_embd),    # 위치 임베딩 가중치
    'lm_head': matrix(vocab_size, n_embd) # 출력 분류 가중치
}

# 값 접근: 대괄호 + 키 이름
tok_emb = state_dict['wte'][token_id]
```

### f-string을 이용한 동적 키 생성
```python
# microgpt.py 250~255줄: 레이어 번호에 따라 키 이름을 동적으로 생성
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    # i=0 이면 키가 'layer0.attn_wq' 가 됨
    # i=1 이면 키가 'layer1.attn_wq' 가 됨
```
**핵심 포인트**: `f'문자열 {변수}'` 형태의 f-string은 문자열 안에 변수를 직접 삽입하는 파이썬 문법입니다. 레이어가 여러 개인 신경망에서 가중치를 체계적으로 관리하기 위한 핵심 패턴입니다.


## 6-5. 집합(Set)과 `in` 연산의 성능
집합(Set)은 **중복을 허용하지 않는** 자료구조이며, 특정 원소가 포함되어 있는지를 **매우 빠르게(O(1))** 확인할 수 있습니다.

```python
# 리스트에서 'in' 연산: 처음부터 끝까지 하나씩 비교해야 함 (느림, O(n))
visited_list = [1, 2, 3, 4, 5]
if 3 in visited_list:  # 최악의 경우 5번 비교
    print("찾음")

# 집합에서 'in' 연산: 해시 기법으로 즉시 찾음 (빠름, O(1))
visited_set = {1, 2, 3, 4, 5}
if 3 in visited_set:  # 단 1번에 확인 완료
    print("찾음")
```

### 💻 `microgpt.py` 적용 (219~231줄)
```python
visited = set()     # 빈 집합 생성

def build_topo(v):
    if v not in visited:  # O(1) 속도로 '이미 방문했는가?' 확인
        visited.add(v)    # 집합에 추가 (중복 삽입은 자동 무시됨)
        for child in v._children:
            build_topo(child)
        topo.append(v)
```
**핵심 포인트**: 계산 그래프에서는 하나의 변수가 여러 곳에서 재사용될 수 있습니다(예: 같은 가중치가 여러 연산에 참여). `Set`을 사용하지 않으면 같은 노드가 중복 등록되어 그래디언트가 잘못 계산되고, 리스트 기반 `in` 연산은 수만 개의 노드에서 심각한 성능 저하를 야기합니다.

---

이 챕터까지 마스터하면, `microgpt.py`에 등장하는 **모든 파이썬 문법적 장벽**이 사라지게 됩니다! 이제 Phase 2 (수학 기초)로 돌아가서, 남은 수학적 개념들을 보충할 차례입니다.

---
| ← [이전 챕터 (Chapter 3, 4, 5)](03_chapter_03_04_05.md) | [목록으로 (Plan)](01_plan.md) | [다음 Phase (Phase 2) 계획서](../phase02_math/01_plan.md) → |
