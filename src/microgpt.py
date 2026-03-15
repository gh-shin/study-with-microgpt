"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python. (순수하고 의존성이 없는 파이썬으로 GPT를 학습시키고 추론을 실행하는 가장 원자적인 방법)
This file is the complete algorithm. (이 파일은 완전한 알고리즘입니다)
Everything else is just efficiency. (그 외의 모든 것은 효율성일 뿐입니다)
@karpathy
"""
import os
import math
import random
random.seed(42)

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
# (문서(예: 이름 리스트)의 리스트인 데이터셋 docs가 있게 하라)

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karphthy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
# (문자열을 정수 시퀀스("토큰")로 변환하고 다시 되돌리는 토크나이저가 있게 하라)
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1 (데이터셋의 유니크한 문자들이 토큰 ID 0..n-1이 됨)
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token (특별한 문장 시작(BOS) 토큰을 위한 토큰 ID)
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS (유니크한 토큰의 총 개수, +1은 BOS를 위한 것)
print(f"vocab size: {vocab_size}")

# Let there be Autograd to recursively apply the chain rule through a computation graph
# (계산 그래프를 통해 연쇄 법칙을 재귀적으로 적용하는 자동 미분(Autograd)이 있게 하라)
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage (메모리 사용량을 위한 파이썬 최적화)

    def __init__(self, data, children=(), local_grads=()):
        """
        Initialize a Value node that represents a scalar in the computation graph.
        (계산 그래프에서 스칼라를 나타내는 Value 노드를 초기화합니다)

        Role: Acts as an atomic unit for tracking values and gradients.
        (역할: 값과 그래디언트를 추적하는 원자 단위 역할을 합니다)

        Effect: Stores data, gradients, and connectivity for automatic differentiation.
        (효과: 자동 미분을 위해 데이터, 그래디언트 및 연결 관계를 저장합니다)

        Usage: v = Value(data, children=(a, b), local_grads=(1, 1))
        (쓰임: v = Value(data, children=(a, b), local_grads=(1, 1)))
        """
        self.data = data # scalar value of this node calculated during forward pass (순전파 중에 계산된 이 노드의 스칼라 값)
        self.grad = 0 # derivative of the loss w.r.t. this node, calculated in backward pass (역전파 중에 계산된 이 노드에 대한 손실의 미분값)
        self._children = children # children of this node in the computation graph (계산 그래프에서 이 노드의 자식들)
        self._local_grads = local_grads # local derivative of this node w.r.t. its children (자식들에 대한 이 노드의 로컬 미분값)
    
    def __add__(self, other):
        """
        Perform addition between this Value and another value.
        (이 Value와 다른 값 사이의 덧셈을 수행합니다)

        Role: Implements the addition operator for the computation graph.
        (역할: 계산 그래프를 위한 덧셈 연산자를 구현합니다)

        Effect: Creates a new Value node that depends on the inputs with a local gradient of 1.
        (효과: 로컬 그래디언트가 1인 입력을 기반으로 한 새로운 Value 노드를 생성합니다)

        Usage: c = a + b
        (쓰임: c = a + b)
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        """
        Perform multiplication between this Value and another value.
        (이 Value와 다른 값 사이의 곱셈을 수행합니다)

        Role: Implements the multiplication operator for the computation graph.
        (역할: 계산 그래프를 위한 곱셈 연산자를 구현합니다)

        Effect: Creates a new Value node with local gradients based on the values of the inputs.
        (효과: 입력값에 기반한 로컬 그래디언트를 가진 새로운 Value 노드를 생성합니다)

        Usage: c = a * b
        (쓰임: c = a * b)
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    
    def __pow__(self, other):
        """
        Raise this Value to the power of a constant.
        (이 Value를 상수의 거듭제곱으로 만듭니다)

        Role: Implements power operations (x**n) where n is a numeric constant.
        (역할: n이 수치 상수인 거듭제곱 연산(x**n)을 구현합니다)

        Effect: Adds a power node to the graph with the appropriate power rule derivative.
        (효과: 적절한 거듭제곱 규칙 미분값을 가진 거듭제곱 노드를 그래프에 추가합니다)

        Usage: b = a**2
        (쓰임: b = a**2)
        """
        return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self):
        """
        Calculate the natural logarithm of this Value.
        (이 Value의 자연로그를 계산합니다)

        Role: Provides the ln(x) function for the computation graph.
        (역할: 계산 그래프를 위해 ln(x) 함수를 제공합니다)

        Effect: Creates a node with a 1/x local derivative.
        (효과: 1/x 로컬 미분값을 가진 노드를 생성합니다)

        Usage: b = a.log()
        (쓰임: b = a.log())
        """
        return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self):
        """
        Calculate the exponential (e^x) of this Value.
        (이 Value의 지수(e^x)를 계산합니다)

        Role: Provides the exp(x) function for the computation graph.
        (역할: 계산 그래프를 위해 exp(x) 함수를 제공합니다)

        Effect: Creates a node where the local derivative is equal to the node's value.
        (효과: 로컬 미분값이 노드의 값과 동일한 노드를 생성합니다)

        Usage: b = a.exp()
        (쓰임: b = a.exp())
        """
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self):
        """
        Apply the Rectified Linear Unit (ReLU) activation function.
        (Rectified Linear Unit (ReLU) 활성화 함수를 적용합니다)

        Role: Introduces non-linearity into the network.
        (역할: 네트워크에 비선형성을 도입합니다)

        Effect: Zeroes out negative values and provides a gradient of 1 for positive values.
        (효과: 음수 값을 0으로 만들고 양수 값에 대해 1의 그래디언트를 제공합니다)

        Usage: b = a.relu()
        (쓰임: b = a.relu())
        """
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self):
        """
        Negate this Value. (이 Value의 부호를 바꿉니다)
        Role: Implements unary negation (-x). (역할: 단항 부호 반전(-x)을 구현합니다)
        Effect: Returns a new Value with data multiplied by -1. (효과: 데이터에 -1을 곱한 새로운 Value를 반환합니다)
        Usage: b = -a (쓰임: b = -a)
        """
        return self * -1
    def __radd__(self, other):
        """
        Perform right-side addition. (우항 덧셈을 수행합니다)
        Role: Enables addition when the left operand is not a Value (e.g., float + Value). (역할: 왼쪽 피연산자가 Value가 아닐 때(예: float + Value) 덧셈을 가능하게 합니다)
        Effect: Redirects to __add__. (효과: __add__로 리다이렉트합니다)
        Usage: b = 1.0 + a (쓰임: b = 1.0 + a)
        """
        return self + other
    def __sub__(self, other):
        """
        Perform subtraction. (뺄셈을 수행합니다)
        Role: Implements the subtraction operator (x - y). (역할: 뺄셈 연산자(x - y)를 구현합니다)
        Effect: Returns a new Value representing the difference. (효과: 차이를 나타내는 새로운 Value를 반환합니다)
        Usage: c = a - b (쓰임: c = a - b)
        """
        return self + (-other)
    def __rsub__(self, other):
        """
        Perform right-side subtraction. (우항 뺄셈을 수행합니다)
        Role: Enables subtraction when the left operand is not a Value (e.g., float - Value). (역할: 왼쪽 피연산자가 Value가 아닐 때(예: float - Value) 뺄셈을 가능하게 합니다)
        Effect: Returns other + (-self). (효과: other + (-self)를 반환합니다)
        Usage: b = 1.0 - a (쓰임: b = 1.0 - a)
        """
        return other + (-self)
    def __rmul__(self, other):
        """
        Perform right-side multiplication. (우항 곱셈을 수행합니다)
        Role: Enables multiplication when the left operand is not a Value (e.g., float * Value). (역할: 왼쪽 피연산자가 Value가 아닐 때(예: float * Value) 곱셈을 가능하게 합니다)
        Effect: Redirects to __mul__. (효과: __mul__로 리다이렉트합니다)
        Usage: b = 2.0 * a (쓰임: b = 2.0 * a)
        """
        return self * other
    def __truediv__(self, other):
        """
        Perform division. (나눗셈을 수행합니다)
        Role: Implements the division operator (x / y). (역할: 나눗셈 연산자(x / y)를 구현합니다)
        Effect: Returns self * other**-1. (효과: self * other**-1을 반환합니다)
        Usage: c = a / b (쓰임: c = a / b)
        """
        return self * other**-1
    def __rtruediv__(self, other):
        """
        Perform right-side division. (우항 나눗셈을 수행합니다)
        Role: Enables division when the left operand is not a Value (e.g., float / Value). (역할: 왼쪽 피연산자가 Value가 아닐 때(예: float / Value) 나눗셈을 가능하게 합니다)
        Effect: Returns other * self**-1. (효과: other * self**-1을 반환합니다)
        Usage: b = 1.0 / a (쓰임: b = 1.0 / a)
        """
        return other * self**-1

    def backward(self):
        """
        Perform a backward pass to calculate gradients using the chain rule.
        (연쇄 법칙을 사용하여 그래디언트를 계산하는 역전파를 수행합니다)

        Role: Triggers automatic differentiation for all nodes in the graph relative to this node.
        (역할: 이 노드를 기준으로 그래프의 모든 노드에 대해 자동 미분을 트리거합니다)

        Effect: Updates the .grad attribute for every contributing node in topological order.
        (효과: 위상 정렬 순서에 따라 기여하는 모든 노드의 .grad 속성을 업데이트합니다)

        Usage: loss.backward()
        (쓰임: loss.backward())
        """
        topo = []
        visited = set()
        def build_topo(v):
            """
            Build a topological ordering of all nodes reachable from v.
            (v에서 도달 가능한 모든 노드의 위상 정렬 순서를 구축합니다)
            Role: Helper function for the backward pass to ensure correct order of differentiation.
            (역할: 올바른 미분 순서를 보장하기 위해 역전파를 돕는 보조 함수입니다)
            Effect: Populates the 'topo' list with nodes in order.
            (효과: 'topo' 리스트를 순서대로 노드로 채웁니다)
            Usage: build_topo(self) (쓰임: build_topo(self))
            """
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model (모델의 지식을 저장하기 위한 파라미터 초기화)
n_layer = 1 # depth of the transformer neural network (number of layers) (트랜스포머 신경망의 깊이(레이어 수))
n_embd = 16 # width of the network (embedding dimension) (네트워크의 너비(임베딩 차원))
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters) (어텐션 윈도우의 최대 컨텍스트 길이 (참고: 가장 긴 이름은 15자))
n_head = 4 # number of attention heads (어텐션 헤드의 수)
head_dim = n_embd // n_head # derived dimension of each head (각 헤드의 파생된 차원)
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value] (파라미터를 하나의 list[Value]로 평탄화)
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next (모델 아키텍처 정의: 토큰과 파라미터를 다음에 올 토큰의 로짓으로 매핑하는 함수)
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU (GPT 중에서도 축복받은 GPT-2를 따르되, 약간의 차이점을 둠: layernorm -> rmsnorm, 바이어스 없음, GeLU -> ReLU)
def linear(x, w):
    """
    Apply a weighted sum (linear transformation) to an input vector.
    (입력 벡터에 가중치 합(선형 변환)을 적용합니다)

    Role: Computes y = Wx where W is the weight matrix and x is the vector.
    (역할: W가 가중치 행렬이고 x가 벡터인 y = Wx를 계산합니다)

    Effect: Projects an embedding into a new dimensional space or calculates logits.
    (효과: 임베딩을 새로운 차원 공간으로 투영하거나 로짓을 계산합니다)

    Usage: out = linear(input_vec, weights)
    (쓰임: out = linear(input_vec, weights))
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """
    Convert a vector of logits into a probability distribution.
    (로짓 벡터를 확률 분포로 변환합니다)

    Role: Normalizes output scores into values between 0 and 1 that sum to 1.
    (역할: 출력 점수를 합이 1이 되는 0과 1 사이의 값으로 정규화합니다)

    Effect: Enables classification and calculation of cross-entropy loss.
    (효과: 분류 및 크로스 엔트로피 손실 계산을 가능하게 합니다)

    Usage: probs = softmax(model_logits)
    (쓰임: probs = softmax(model_logits))
    """
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """
    Apply Root Mean Square Layer Normalization to the input vector.
    (입력 벡터에 Root Mean Square Layer Normalization을 적용합니다)

    Role: Stabilizes training by normalizing activations without using mean subtracting.
    (역할: 평균 차감 없이 활성화 값을 정규화하여 학습을 안정화합니다)

    Effect: Scales elements to maintain a consistent variance across layers.
    (효과: 레이어 전반에 걸쳐 일관된 분산을 유지하도록 요소를 스케일링합니다)

    Usage: normalized_x = rmsnorm(x)
    (쓰임: normalized_x = rmsnorm(x))
    """
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """
    Perform a single forward pass step of the GPT transformer model.
    (GPT 트랜스포머 모델의 단일 순전파 단계를 수행합니다)

    Role: Maps a sequence of tokens to prediction logits for the next token.
    (역할: 토큰 시퀀스를 다음에 올 토큰의 예측 로짓으로 매핑합니다)

    Effect: Processes input through attention and MLP blocks while building the graph.
    (효과: 그래프를 구축하면서 어텐션 및 MLP 블록을 통해 입력을 처리합니다)

    Usage: logits = gpt(current_token, current_pos, kv_cache_k, kv_cache_v)
    (쓰임: logits = gpt(current_token, current_pos, kv_cache_k, kv_cache_v))
    """
    tok_emb = state_dict['wte'][token_id] # token embedding (토큰 임베딩)
    pos_emb = state_dict['wpe'][pos_id] # position embedding (위치 임베딩)
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding (결합된 토큰 및 위치 임베딩)
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection (참고: 잔차 연결을 통한 역전파로 인해 불필요하지 않음)

    for li in range(n_layer):
        # 1) Multi-head Attention block (1) 멀티 헤드 어텐션 블록)
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # 2) MLP block (2) MLP 블록)
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer and its buffers (축복받은 옵티마이저 Adam과 그 버퍼들이 있게 하라)
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer (첫 번째 모멘트 버퍼)
v = [0.0] * len(params) # second moment buffer (두 번째 모멘트 버퍼)

# Repeat in sequence (순차적으로 반복)
num_steps = 1000 # number of training steps (학습 단계 수)
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides (단일 문서를 가져와 토큰화하고 양쪽에 BOS 특별 토큰으로 감쌈)
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss (토큰 시퀀스를 모델에 순전파하여 손실까지 계산 그래프를 구축)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low. (문서 시퀀스에 대한 최종 평균 손실. 당신의 손실이 낮기를)

    # Backward the loss, calculating the gradients with respect to all model parameters (손실을 역전파하여 모든 모델 파라미터에 대한 그래디언트 계산)
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients (Adam 옵티마이저 업데이트: 해당 그래디언트를 기반으로 모델 파라미터 업데이트)
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay (선형 학습률 감소)

    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')