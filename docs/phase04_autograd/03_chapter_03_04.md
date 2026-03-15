# Chapter 3. DFS를 활용한 체인룰 계산기 (`backward`)

`Value` 클래스들의 곱셈, 덧셈 트리 맨 마지막에서 `loss.backward()` 를 한 번만 부르면 그 전위의 모든 연산과 가중치(파라미터)들이 자신의 미분값(gradient)을 알아내게 되는 마법입니다.
* 지금까지 배운 **위상 정렬(Topological sorting) + 역전파 체인룰(Chain Rule)**을 `microgpt.py` 218~240줄 코드로 완전히 합쳐낸 실전 코드입니다.

## 3-1. 위상 정렬과 재귀 함수 (DFS)
가장 끝 점(나침반) `loss` 에서 시작하여 가지(`_children`)를 부모 방향으로 거슬러 올라가 끝까지 전부 탐색(DFS)합니다. 그리고 더이상 부모가 없으면(파라미터 초기값 등), 비로소 리스트(`topo`)에 자신을 거꾸로 쌓아 나갑니다.

```python
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)  # 방문 표기
                for child in v._children:
                    build_topo(child) # 자원봉사 중: 끝도 없이 파고드는 재귀함수
                # 나(v)의 부모 탐색이 모두 끝난 후에야 비로소 명부에 이름 올림
                topo.append(v)
        
        build_topo(self) # 이 self 는 항상 맨 마지막에 태어난 결과 노드(loss)임
```

## 3-2. 글로벌 그래디언트의 누적 (역방향 파도치기)
잘 세워진 `topo` 리스트들을 거꾸로(`reversed`) 훑고 내려가며 자식들에게 수돗물(기울기)을 틀어주는 `for` 루프문입니다. 
`child.grad += (내 로컬 그래디언트) * (부모의 글로벌 그래디언트 v.grad)` 공식을 그대로 넣습니다!

```python
        # 처음 출발지인 Loss 자신 입장에선, 자기가 변하는 비율이 당연히 1배임
        self.grad = 1.0
        
        # topo 순번을 뒤집어, 가장 나중에 들어온 놈(최종 조상=loss)부터 거꾸로 역으로!
        for v in reversed(topo):
            # v 가 아까 _children 에 저장해뒀었던 자신만의 (자식, 그 자식에게 미치는 영향력)
            for child, local_grad in zip(v._children, v._local_grads):
                # 미분값의 연쇄법칙 누적! (+= 로 합산 쌓임)
                child.grad += local_grad * v.grad
```
> ✅ **실제 계산 그래프 순회 예제**
> `loss = (a + 1) * b`
> 1. `b` 노드와 덧셈 `(+)` 노드가 가장 마지막 `topo` 배정에 들어갑니다.
> 2. `loss.grad = 1.0` 으로 초기화.
> 3. `reversed` 를 통해 가장 먼저 `loss` 노드가 루프를 돌고, 
> 4. `(+)` 와 `b`의 `.grad` 속성에 `local_grad * 1.0`을 더해줍니다.
> 5. 그다음 `(+)` 노드가 이 루프를 돌아, 자신의 자식인 `a` 와 `1` 에게 위에서 받은 그래디언트와 로컬 그래디언트를 곱해(`1.0 * v.grad`) 다시 넘겨줍니다!


# Chapter 4. 신경망(NN)으로의 확장 및 검증

이제 이렇게 멋지게 자동 미분되는 장난감 1개(스칼라 `Value`)를 완성했으니, 이것들을 리스트나 행렬로 다발로 묶으면 그대로 파이토치(PyTorch)와 동일한 능력을 갖는 거대한 파라미터(모델 지식) 저장소가 되는 것입니다! 

## 4-1. 행렬 연산(Linear)의 구현

우리가 파이토치에서 자주 쓰는 선형 레이어(`nn.Linear`)를 `microgpt.py` 에서는 단순 파이썬 리스트 내포기로 완성해 버립니다.
```python
# 선형 변환식 행렬 Wx 의 곱합 연산
def linear(x, w):
    # 각각의 x(벡터), w(가중치 점선들)를 모두 Value로 처리합니다. 
    # w 행렬 내부에는 Value 객체들이 [nout, nin] 형태로 다발로 차있습니다.
    # sum(wi * xi) 안의 * 연산과 + 연산 덕분에 암흑 속에 거대한 DAG가 짜여집니다.
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w] 
```

## 4-2. 소프트맥스(Softmax) 확률 변환
마지막 모델의 결과물을 확률 분포(0~1 사이)로 쪼개기 위한 소프트맥스(Softmax) 역시, 내장된 `exp()` 지수 오버로딩과 덧셈, 나눗셈 오버로딩들에 의해 파이써닉하게 한 줄로 그 지도를 그릴 수 있습니다.
```python
def softmax(logits):
    max_val = max(val.data for val in logits) # 스케일 안전판
    # logits [Value, Value, Value] 들이 각자 매직메서드 .exp() 를 발사!
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

## 4-3. 손실 역전파 계산과 옵티마이저 (Adam) 학습
* **1단계(Forward)**: 천 번의 스텝 동안 매번 `loss` (1개)를 계산!
* **2단계(Backward)**: 
  ```python
  loss.backward()  # 우리가 짠 체인 룰 엔진! 전체 파라미터들이 .grad 값을 알아냄.
  ```
* **3단계(Optimizer/Adam)**: 
  알아낸 그 `grad`(틀린 방향의 비례 수치) 값을 이용해, 파라미터(지식 저장소)의 뇌세포(`p.data`) 신경들을 학습률(`lr_t`)만큼 조금씩 깎습니다! **오차를 줄이는 방향으로!**

  ```python
  for p in params: 
      # 학습률과 모멘텀(Adam)을 반영해 가중치 갱신
      p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
      p.grad = 0  # 다음 판을 위해 기울기는 다시 0으로 초기화
  ```

---

### 🎉 **축하합니다!**
이렇게 **"Phase 1: 파이썬 (객체/매직 메서드)" -> "Phase 2: 수학 기초 (편미분/연쇄 법칙)" -> "Phase 3: 계산 그래프 알고리즘 (위상정렬/역전파)" -> "Phase 4: 자동 미분 엔진 구현 (Value/역방향 탐색/행렬화)"** 까지의 학습이 완료되었습니다.

이제 복잡하고 거대한 프레임워크인 텐서플로(TensorFlow)나 파이토치(PyTorch)의 소스코드를 열어보아도 "아, 결국 바닥 코어에는 내가 만든 `Value` 같은 클래스와 `_children` 관계, 트리 순회 및 곱셈 누적이 숨어 있구나!"라는 것을 쉽게 떠올릴 수 있을 것입니다. 
이 4단계 플랜과 문서를 통해 안드레이 카파시(Andrej Karpathy)가 `microgpt`로 전하고자 하는 진정한 딥러닝 코어 알고리즘의 원리를 깨달으시기 바랍니다.
