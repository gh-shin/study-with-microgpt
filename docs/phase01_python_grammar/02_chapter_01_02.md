# Chapter 1. 파이썬과 데이터 다루기 기초

우리가 작성할 `microgpt.py`의 `Value` 클래스는 기본적으로 계산 수식에 들어가는 수많은 "데이터(숫자)"들을 담는 통입니다. 따라서 파이썬이 데이터를 어떻게 취급하는지 정확히 이해하고 넘어가야 합니다.

## 1-1. 변수와 스칼라(Scalar) 값
수학에서 스칼라(Scalar)는 단순히 1차원의 크기만을 가지는 변수를 뜻합니다. 신경망 내부에서는 수많은 스칼라 값들이 더해지고 곱해집니다.

```python
# 파이썬에서의 스칼라 값 선언 예시
a = 1.0     # 실수 (float)
b = 2       # 정수 (int)
c = a + b   # 사칙 연산
print(c)    # 출력: 3.0
```
**핵심 포인트**: 앞으로 `microgpt.py`에서 보게 될 `self.data` 나 `self.grad` 같은 속성에는 모두 이렇게 스칼라(float) 값이 저장된다는 점을 기억하세요. 딥러닝 연산의 최소 단위가 됩니다.

## 1-2. 기본 자료구조: 리스트(List)와 튜플(Tuple)
단일 값뿐만 아니라 여러 개의 값을 묶어서 관리하는 파이썬의 표준 자료구조입니다.
`Value` 클래스에서는 어떤 노드들이 결합되었는지를 기억하는 `_children` 등에 데이터를 보관하기 위해 여러 개의 리스트와 튜플이 쓰입니다.

```python
# 리스트 (List, 대괄호 사용): 데이터를 넣고 뺄 수 있음 (수정 가능, Mutable)
topo = []            # 역전파 순서를 담을 빈 리스트
topo.append(10)      # 요소를 마지막에 추가
topo.append(20)

# 튜플 (Tuple, 소괄호 사용): 일단 만들면 수정할 수 없음 (수정 불가, Immutable)
children = (a, b)    # a와 b를 하나로 묶음
```
**핵심 포인트**: `__init__` 함수에서 `self._children = children` 형태로 저장이 될 때, 빈 괄호 `()`가 튜플을 뜻한다는 점을 잘 알고 넘어가야 합니다. 객체의 계보(족보)를 수정하지 못하게 안전하게 보관하기 위해 튜플을 주로 씁니다.

## 1-3. 제어문(if, for, while)과 함수(def)
계산의 흐름을 통제하거나 반복시키며, 자주 쓰는 코드를 재사용 가능한 단위로 분리합니다.

```python
# for 문을 이용한 반복 탐색
# 리스트에 있는 값들을 순회
visited = {1, 2, 3}      # 집합(Set) 형태로 중복을 방지
node = 2
if node in visited:      # if 문에 의한 논리 제어
    print("이미 방문했습니다.")

# 함수 (def)
def calculate_forward(x_val, y_val):
    result = x_val * y_val
    return result

# 함수 내부의 함수 (중첩 함수, Nested function)
# => 이후 Value.backward() 메서드 내부에 쓰이는 build_topo()를 이해하기 위한 핵심
def backward():
    visited = set()   # 바깥쪽 함수의 변수
    def build_topo(v):
        if v not in visited:
            visited.add(v)  # 안쪽 함수에서 바깥쪽 함수의 변수에 접근 가능(클로저 성질)
    build_topo(10)
```
**핵심 포인트**: `microgpt.py`의 218~235번째 줄에 위치한 `def backward():` 내부에 중첩되어 정의된 `def build_topo(v):` 의 사용 패턴입니다. 위와 같이 중첩 함수를 쓰면, 바깥 함수의 로컬 변수(예: `visited`, `topo`)를 별도의 인자 전달 없이 공유하여 코드를 간결하게 짤 수 있습니다.


# Chapter 2. 객체지향 프로그래밍 (OOP) 입문

`Value` 클래스의 본질은 데이터를 담는 객체입니다. 객체지향의 근본적인 철학은 **"상태(Data)와 그 상태를 다루는 행동(Method)을 하나로 묶어서 관리하자"** 입니다. 단순 텍스트 숫자는 기울기를 가질 수 없지만, `Value` 객체는 기울기를 가질 수 있습니다.

## 2-1. 클래스(Class)와 객체(Object)의 개념
- **클래스 (Class)**: 객체를 생성하기 위한 틀 또는 도면 (예: 붕어빵 틀)
- **객체 (Object)**: 클래스에 의해 메모리에 할당된 실제 데이터 덩어리 (예: 만들어진 붕어빵)

```python
class Node:
    pass

node_a = Node() # Node 클래스 기반으로 만들어진 객체 node_a
node_b = Node() # Node 클래스 기반으로 만들어진 객체 node_b
```

## 2-2. 생성자(`__init__`)와 `self`
생성자 `__init__`는 객체가 최초에 만들어질 때(인스턴스화) 자동으로 호출되어 **상태를 초기화**하는 역할을 합니다. 가장 중요한 예약어가 바로 `self`인데, 이는 **생성된 객체 자기 자신**을 지칭합니다.

```python
class Value:
    # __init__: 초기화 역할을 담당하는 매직 메서드
    def __init__(self, data, _children=()):
        self.data = data         # 자신의 값(data)을 외부로부터 받아서 저장
        self.grad = 0.0          # 어떤 노드든 태어날 때 초기 기울기(grad)는 0으로 시작
        self._children = _children  # 이 노드를 낳은 부모 노드들 저장

# 사용 (Usage)
# 숫자 2.0이 들어가 Value라는 '껍데기'를 입게 됨
a = Value(2.0)
print(a.data)  # 2.0
print(a.grad)  # 0.0
```
**핵심 포인트**: 스칼라 값(`a = 2.0`) 대신 `a = Value(2.0)` 형태로 객체화함으로써 얻는 가장 큰 이점은 바로 `a.grad` 같은 부수적인 데이터를 숫자와 꼭 붙여서(마치 꼬리표처럼) 들고 다닐 수 있게 되었다는 점입니다! 우리는 딥러닝에서 단순 숫자뿐만 아니라 그 숫자의 '기울기'가 무조건 필요하기 때문입니다.

## 2-3. 인스턴스 메서드 (Instance Methods)
클래스 내부에 정의된 함수를 뜻하며, 첫 번째 인자로 무조건 `self`를 받습니다. 자기 자신이 가진 상태 데이터를 읽거나 조작하기 위한 용도로 쓰입니다.

```python
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
    
    # 인스턴스 메서드: 외부 명령에 의해 자신의 내부 상태를 변경
    def update_gradient(self, delta):
        self.grad += delta

a = Value(5.0)
a.update_gradient(0.1) # 내부적으로 update_gradient(a, 0.1) 처럼 동작
print(a.grad)          # 0.1
```
**핵심 포인트**: 위 코드처럼 `Value` 객체들은 단순 연산만 수행하는 것이 아니라 `backward()`같은 인스턴스 메서드를 지니며, 스스로의 자식 노드들을 찾아가서(`self._children`) 기울기를 전달하는 능동적인 작업 체계를 가집니다.
