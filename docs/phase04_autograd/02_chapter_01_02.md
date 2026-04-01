# Chapter 1. 단일 스칼라 노드 객체화 (Value 클래스 뼈대)

우리가 작성한 `microgpt.py` 31번째 줄부터는 본격적인 **Autograd Engine (자동 미분 엔진)** 의 심장부인 `Value` 클래스가 정의되어 있습니다.

## 1-1. 상태의 캡슐화 (객체 초기화 지점)
학습 과정에서 쓸 미분값을 담는 변수(`grad`)와 값 유지력(`data`), 그리고 어디서부터 태어났는지 기억하는 족보(`_children`)를 파이썬 내장 클래스로 보호합니다.

```python
class Value:    
    def __init__(self, data, children=(), local_grads=()):
        self.data = data             # 자신의 스칼라 값 (순전파 중에 계산됨)
        self.grad = 0                # 자신에 대한 손실함수의 미분 값 (역전파 중 채워짐)
        self._children = children    # 자신을 만든 부모 노드 (A*B=C 였다면 C._children=(A,B))
        self._local_grads = local_grads # A와 B가 C에게 미치는 지역 미분값들
```

## 1-2. 메모리 최적화와 `__slots__`
`microgpt.py` 32줄에 있는 최적화입니다. GPT를 학습시키려면 `q, k, v` 선형 층 같은 행렬만 수백 개가 필요하며, 그 내부의 점(`element`) 하나하나가 모두 `Value` 객체로 파편화되어 메모리에 상주합니다!
* `__slots__ = ('data', 'grad', '_children', '_local_grads')`를 선언하면 파이썬이 해시맵(`__dict__`) 대신 고정 길이 튜플로 인스턴스를 관리하므로, 메모리 자원 낭비를 1/5 수준으로 철저하게 최소화할 수 있습니다.


# Chapter 2. 연산 노드 생성기 구현 (매직 메서드 오버로딩)

값을 가지고 태어난 뼈대(`Value`)끼리 스스로 수학 연산(`+`, `*`)을 시키면, **"오버로딩 된 파이썬 매직 메서드들을 통해 순전파 진행과 동시에 _children 지도까지 하나로 엮어버린다"**는 사실이 머신러닝 코드의 최고의 미학입니다.

## 2-1. 수학 기호의 객체화 (`__add__`, `__mul__`)
`a + b`를 하게 될 때 내부적으로는 파이썬이 `a.__add__(b)`를 실행합니다. 

```python
    def __add__(self, other):
        # 1. other 형 변환 방어 (아래 2-2 참조)
        other = other if isinstance(other, Value) else Value(other)
        
        # 2. 값(data)끼리 더하여 새로운 껍데기(Value) 생성!
        # [Phase 2]에서 배운 대로 덧셈의 로컬 미분값은 공평하게 (1, 1) 입니다!
        return Value(
            self.data + other.data, 
            children=(self, other), 
            local_grads=(1, 1)
        )
```

> ✅ **실제 덧셈 객체 연산 예시**
> `a = Value(3.0)`, `b = Value(2.0)` 으로 만들고 `c = a + b` 를 수행합니다.
> 결과인 `c`는 단순 스칼라 `5.0`이 아닙니다!
> * `c.data` = `5.0`
> * `c._children` = `(a, b)`
> * `c._local_grads` = `(1, 1)`

## 2-2. 파이썬 타입 방어와 우항 연산자 지원
코드 상에서 실수 연산을 훨씬 부드럽게 만들기 위해, 상대방 피연산자(`other`)가 파이썬 기본 상수(`int`나 `float`)일지라도 에러가 나지 않도록 유연한 형태의 랩핑 분기점 작성이 필수입니다.

```python
    # 상대가 왼쪽에서 덧셈/곱셈을 걸어왔을 때(ex: 3.0 + a) 대처
    def __radd__(self, other):
        return self + other  # 교환 법칙 가능. 기존 자신의 __add__ 로 토스!
        
    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        # 뺄셈은 그냥 '자신 + (부호가 뒤집힌 other)'와 같은 원리!
        return self + (-other)
```

* **이렇게 `Value` 클래스를 감싸 두기만 하면, 복잡히 얽힌 행렬 점곱이나 자연로그 등 우리가 아는 모든 파이썬 수학 기호를 `Value` 변수 사이에 막 던져도, 에러 없이 거대한 방정식 트리를 자동 구축(Build)하게 됩니다.**

---
| [목록으로 (Plan)](01_plan.md) | [다음 챕터 (Chapter 3, 4)](03_chapter_03_04.md) → |
