# MicroGPT Study Project 🧠

이 프로젝트는 Andrej Karpathy의 [microgpt.py](https://karpathy.github.io/2026/02/12/microgpt/)를 기반으로, 딥러닝과 트랜스포머의 원리를 가장 원자적인 수준에서 학습하기 위한 교육용 리포지토리입니다.

의존성 없는 순수 파이썬만으로 구현된 GPT 알고리즘을 분석하고, 그 기술적 배경이 되는 수학과 알고리즘을 4단계의 학습 단계를 통해 마스터하는 것을 목표로 합니다.

## 🚀 프로젝트 개요

- **대상 코드**: `src/microgpt.py` (Karpathy의 원본 스크립트에 상세한 한국어 주석이 추가된 버전)
- **핵심 목표**: 
  - 스칼라 기반 자동 미분 엔진(Autograd)의 이해 및 구현
  - 계산 그래프(Computation Graph)와 역전파(Backpropagation)의 동작 원리 습득
  - 트랜스포머(Transformer) 아키텍처 및 어텐션(Attention) 메커니즘 분석
  - Adam 옵티마이저와 학습 루프의 실전 구현

---

## 📚 학습 로드맵 (Learning Phases)

상세 내용은 `docs/` 폴더 내의 각 Phase 문서를 참고해 주세요.

### [Phase 1: 파이썬 객체지향 및 고급 문법](docs/phase01_python_grammar/01_plan.md)
계산 그래프를 코드로 표현하고, 수학 연산 기호를 파이썬 객체에 적용하기 위한 필수 문법을 학습합니다.
- **주요 내용**: 매직 메서드(Dunder Methods) 오버로딩, `__slots__` 최적화, 재귀 호출과 DFS, 리스트 내포 및 Lambda 함수.

### [Phase 2: 미분의 본질과 연쇄 법칙](docs/phase02_math/01_plan.md)
머신러닝의 엔진 역할을 하는 기초 수학과 미분법을 다룹니다.
- **주요 내용**: 다항식/로그/지수 함수의 미분, 다변수 함수와 편미분, 연쇄 법칙(Chain Rule), 크로스 엔트로피 손실 함수의 수학적 원리.

### [Phase 3: 딥러닝 코어 이론](docs/phase03_deep_learning/01_plan.md)
수학적 연산을 자료구조(그래프)로 변환하여 컴퓨터가 학습할 수 있는 형태로 구성하는 법을 배웁니다.
- **주요 내용**: 계산 그래프(DAG), 위상 정렬(Topological Sort), 역전파 알고리즘, ReLU 활성화 함수, 토큰화와 임베딩, 어텐션과 트랜스포머 아키텍처.

### [Phase 4: Autograd Engine 실전 구현](docs/phase04_autograd/01_plan.md)
이전 단계에서 배운 모든 지식을 결합하여 `Value` 클래스를 직접 구현하고 GPT 모델을 완성합니다.
- **주요 내용**: `Value` 클래스 뼈대 구축, 연산 노드 생성기 구현, DFS 기반 `backward` 엔진, 가우시안 초기화, Adam 옵티마이저 및 학습 루프 분석.

---

## 💻 핵심 소스 코드: `microgpt.py`

`src/microgpt.py` 파일은 약 400줄의 코드로 이루어진 완전한 GPT 알고리즘입니다.

- **Value 클래스**: 모든 연산의 기초가 되는 자동 미분 노드
- **gpt 함수**: 트랜스포머 레이어, RMSNorm, Multi-head Attention 구현
- **학습 루프**: 데이터 로딩, 토큰화, 순전파, 역전파, 파라미터 업데이트(Adam)의 전 과정

## 🛠️ 시작하기

1. **환경 구성**: 별도의 라이브러리 설치가 필요 없습니다. 순수 파이썬(Python 3.x) 환경이면 충분합니다.
2. **코드 실행**:
   ```bash
   python src/microgpt.py
   ```
3. **학습 방법**: `docs/01_summary.md`를 먼저 읽고, Phase 1부터 순서대로 이론과 코드를 대조하며 학습하시기 바랍니다.

---

## 🔗 관련 링크
- [Andrej Karpathy's MicroGPT Blog Post](https://karpathy.github.io/2026/02/12/microgpt/)
- [MicroGPT GitHub Repository](https://github.com/karpathy/microgpt)

---
"모든 복잡한 알고리즘은 아주 단순한 원자적 원리들의 집합입니다." - Happy Coding! 🚀
