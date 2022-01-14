### Chapter 3 - 신경망

[TOC]

> 컴퓨터가 수행하는 복잡한 처리도 퍼셉트론으로 (이론상) 표현할 수 있지만 가중치를 설정하는 건 사람이 수동으로 해줘야 함 

>:point_right: 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력을 가진 신경망을 이용해 해결



#### 퍼셉트론에서 신경망으로

##### 신경망의 예

- <img src="https://blog.kakaocdn.net/dn/QfyS1/btqH7cdVOs5/GhDWPa5tGZOTjsEK5FXc9K/img.png" alt="02-1. 신경망 (1) - 3층 신경망 순전파 구현" style="zoom: 33%;" />

  - 입력층 : 가장 왼쪽 줄 (입력이 들어오는 층)

  - 출력층 : 가장 오른쪽 줄 (출력되는 층)

  - 은닉층 : 중간 줄 (눈에 보이지 않는 층)

    > 위 그림의 신경망은 3층으로 구성되지만 가중치를 갖는 층은 2개뿐이라 '2층 신경망'이라고 한다. 문헌에 따라 차이는 있을 수 있다.

##### 퍼셉트론 복습

<img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211228011020885.png" alt="image-20211228011020885" style="zoom:50%;" />

- b : **편향**(뉴런이 얼마나 쉽게 활성화되느냐를 제어)
- w1,w2 : 각 신호의 가중치 (신호의 영향력을 제어)

- h(x)함수는 입력이 0을 넘으면 1을 돌려주고 그렇지 않으면 0을 돌려줌

- 활성화 함수 

  - 입력 신호의 총합을 출력 신호로 변환하는 함수
  - 입력 신호의 총합이 활성화를 일으키는지를 정하는 함수

  - <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211228011410419.png" alt="image-20211228011410419" style="zoom:67%;" />
    - 가중치가 곱해진 입력 신호의 총합을 계산하고, 그 합을 활성화 함수에 입력해 결과를 내는 2단계로 처리
    - <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211228011644350.png" alt="image-20211228011644350" style="zoom: 43%;" />
      - 기존 뉴런의 원을 키워 그 안에 활성화 함수의 처리 과정을 그려 넣음
  - 단순 퍼셉트론 : 단층 네트워크에서 계단 함수(임계값을 경계로 출력이 바뀌는 함수)를 활성화 함수로 사용한 모델
  - 다층 퍼셉트론 : 신경망(여러 층으로 구성되고 시그모이드 함수 등의 매끈한 활성화 함수를 사용하는 네트워크)


------

#### 활성화 함수(Activation function)

##### 시그모이드 함수(sigmoid function)

- 신경망에서 자주 이용하는 활성화 함수
- <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231153423938.png" alt="image-20211231153423938" style="zoom:70%;" />
- 

##### 계단 함수(Heaviside function) 구현하기

- ```python
  def step_function (x): # x라는 넘파이 배열 준비
  	y = x > 0 # 넘파이 배열에 부등호 연산 수행 -> bool 배열 생성
      return y.astype(np.int) # 배열 y의 원소를 bool에서 int형으로 바꿔줌 (True->1, False->0)
  ```

##### 계단 함수의 그래프

- ```python
  import numpy as np	
  import matplotlib.pylab as plt
  
  def step_function(x):
      return np.array(x > 0, dtype = np.int) # x의 원소가 0보다 크면 1, 작으면 0 리턴
  
  x = np.arange(-5.0, 5.0, 0.1)
  y = step_function(x)
  plt.plot(x,y)
  plt.ylim(-0.1, 1.1)
  plt.show()
  ```

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231160107815.png" alt="image-20211231160107815" style="zoom:80%;" />

##### 시그모이드 함수 구현하기

- ```python
  def sigmoid(x):
  	return 1 / (1 + np.exp(-x)) # 인수 x가 넘파이 배열이어도 올바른 결과 도출 (<- 브로드캐스트 기능)
  ```

  - 브로드캐스트 기능 : 넘파이 배열과 스칼라값의 연산을 넘파이 배열의 원소 각각과 스칼라값의 연산으로 바꿔 수행하는 것

    <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211128183859899.png" alt="image-20211128183859899" style="zoom:67%;" />

- ```python
  x = mp.arange(-5.0, 5.0, 0.1)
  y = sigmoid(x)
  plt.plot(x, y)
  plt.ylim(-0.1, 1.1)
  plt.show()
  ```

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231160718809.png" alt="image-20211231160718809" style="zoom:67%;" />

##### 시그모이드 함수와 계단 함수 비교

- '매끄러움'의 차이 - 계단 함수는 0또는 1이지만 시그모이드 함수는 연속적인 값이 나타남
- 비슷한 점 - 둘다 입력이 작을 때의 출력은 0에 가깝고(혹은 0), 입력이 커지면 출력이 1에 가까워지는(혹은 1) 구조

##### matplotlib 이용하기

- 정사각형 그리기

  ```python
  import matplotlib.pylab as plt
  import numpy as np
  
  #(1,1) -> (-1,1) -> (-1,-1) -> (1,-1) -> (1,1)
  x = np.array([1,-1,-1,1,1])
  y = np.array([1,1,-1,-1,1])
  plt.plot(x,y)
  ```

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220114233103976.png" alt="image-20220114233103976" style="zoom:40%;" />



------

#### 비선형 함수

- 계단 함수와 시그모이드 함수는 둘다 **비선형 함수**
  - **선형 함수** : 출력이 입력의 상수배만큼 변하는 함수, 1개의 직선
  - **비선형 함수** : 직선 1개로 그릴 수 없는 함수
- 선형 함수의 문제
  - 층을 아무리 깊게 해도 '은닉층이 없는 네트워크'로도 똑같은 기능을 할 수 있음
  - 여러층으로 구성하는 이점을 살릴 수 없음

------

##### ReLU함수

- 시그모이드보다 최근엔 **ReLu** 함수를 주로 사용함

- <img src="https://machinelearningmastery.com/wp-content/uploads/2018/10/Line-Plot-of-Rectified-Linear-Activation-for-Negative-and-Positive-Inputs.png" alt="A Gentle Introduction to the Rectified Linear Unit (ReLU)" style="zoom: 28%;" />

- <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231171830798.png" alt="image-20211231171830798" style="zoom:67%;" />

- ```python
  def relu(x):
  	return np.maximum(0,x) # maximum : 두 입력 중 큰 값 반환
  ```

------

#### 다차원 배열의 계산

##### 다차원 배열

- ```python
  >>> A = np.array([[1,2], [3,4], [5,6]])
  >>> np.ndim(A) # 배열의 차원 수 출력
  2
  >>> A.shape # 배열의 형상 출력 - 형태를 통일 시키기 위해 tuple로 형태로 반환
  (3,2)
  ```

##### 행렬의 곱

- `np.dot(A,B)` : 행렬 A와 B의 곱
  - 첫번째 행렬의 원소 수(열 수)와 두번째 행렬의 0번째 차원의 원소 수(행 수)가 같아야 연산이 됨

##### 신경망에서의 행렬 곱

- <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231180808202.png" alt="image-20211231180808202" style="zoom:70%;" />
- `Y = np.dot(X,W)` 을 통해 단번에 결과 Y를 계산할 수 있음

------

##### 3층 신경망 구현하기

##### 3층 신경망<img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231182532649.png" alt="image-20211231182532649" style="zoom:67%;" />

- 입력층(0층) - 노드 2개
- 첫번째 은닉층(1층) - 노드 3개
- 두번째 은닉층(2층) - 노드 2개
- 출력층(3층) - 노드 2개

##### 표기법 설명

- 가중치(w)와 은닉층 뉴런(a) 표기법
  - 기호 오른쪽 위에 윗첨자 - 층수
  - 기호 오른쪽 아래 아랫첨자 두개 - 차례로 다음 층 뉴런과 앞 층 뉴런의 인덱스 번호

- 예시<img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231182728551.png" alt="image-20211231182728551" style="zoom:67%;" />

  - 입력층(0층)의 뉴런 x2에서 다음 층 뉴런 a1(1)으로 향하는 엣지위에 가중치를 표시

- <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231184526077.png" alt="image-20211231184526077" style="zoom:67%;" />

  - 편향의 입력 신호는 항상 **1**

  - 편향은 오른쪽 아래 인덱스가 하나만 있음(앞 번호 생략)

    - 입력층(0층)의 편향 뉴런이 하나라 **어디서 왔는지 단번에 알 수 있기 때문에 생략!**

  - a식을 간소화

    <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231184840948.png" alt="image-20211231184840948" style="zoom:50%;" />

    <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231184916460.png" alt="image-20211231184916460" style="zoom:80%;" />

  - 구현하기

    <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231185334955.png" alt="image-20211231185334955" style="zoom:80%;" />

    - X와 W1의 대응하는 원소수가 일치하므로 행렬 곱셈 가능

- 활성화 함수를 명시한 뉴런 - 입력층(0층)에서 은닉층(1층)으로의 신호전달<img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231185434314.png" alt="image-20211231185434314" style="zoom:80%;" />

  - (step 1) a = 가중치가 곱해진 입력 신호와 편향의 총합을 계산

  - (step 2) a를 활성화 함수(시그모이드 함수) h()에 넣고 y를 출력

  - 구현하기

    ```python
    Z1 = sigmoid(A1) # 활성화 함수 h()로 변환된 신호를 z로 표기
    
    print(A1) # [0.3, 0.7, 1.1]
    print(Z1) # [0.57444252, 0.66818777, 0.75026011]
    ```

- 은닉층(1층)에서 은닉층(2층)으로의 신호 전달)

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231190125268.png" alt="image-20211231190125268" style="zoom:80%;" />

  - 구현하기

    ```python
    W2 = np.array([[0.1, 0.4] , [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    
    print(Z1.shape) # (3,)
    print(W2.shape) # (3,2)
    print(B2.shape) # (2,)
    
    A2 = np.dot(Z1,W2) + B2
    Z2 = sigmoid(A2)
    ```

    - 은닉층(1층)의 뉴런이 새로운 입력신호가 되어 전달

- 은닉층(2층)에서 출력층(3층)으로의 신호 전달

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231190515949.png" alt="image-20211231190515949" style="zoom:80%;" />

  - 구현하기

    ```python
    def identity_function(x):
    	return x
    
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3) #혹은 Y = A3
    ```

    - 출력 뉴런의 **활성화 함수**로 **항등함수**인 identity_function()을 정의
    - 은닉층의 활성화 함수는 h()이고 출력층의 활성화 함수는 σ()로 표시해 서로 다름을 명시

##### 구현 정리

```python
import numpy as np

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def identity_function(x):
	return x

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1,W2) +b2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2,W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [0.31682708, 0.69627909]
```

- `init_network()` : 가중치와 편향을 초기화하고 그 값(각 층에 필요한 매개변수)을 딕셔너리 변수인 network에 저장하는 함수
- `forward()` : 입력 신호를 출력으로(순방향, 순전파) 변환하는 함수

------

#### 출력층 설계하기

- 신경망은 분류와 회귀 문제에 이용 가능
  - 분류 :point_right: 항등 함수(identity function)
  - 회귀 :point_right: 소프트맥스 함수

- 기계학습 문제는 **분류**와 **회귀**로 나뉨
  - 분류 : 입력 데이터가 어느 클래스에 속하는가?
    - ex) 사진 속 인물의 성별 분류
  - 회귀 : 입력 데이터에서 (연속적인) 수치를 예측
    - ex) 사진 속 인물의 몸무게를 예측하는 문제

##### 항등 함수와 소프트맥스 함수 구현하기

- 항등 함수 

  - y = f(x) = x
  - 입력값 = 출력값

- 소프트맥스 함수

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231212951001.png" alt="image-20211231212951001" style="zoom: 33%;" />

  - n : 출력층의 뉴런 개수

  - yk : 출력층의 k번째 뉴런의 출력

  - 분자 : 입력 신호 ak의 지수 함수

  - 분모 : 모든 입력 신호 a의 지수 함수의 합

  - <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231213106909.png" alt="image-20211231213106909" style="zoom:67%;" />

    - 출력은 모든 입력 신호로부터 화살표를 받음 = 출력층의 각 뉴런이 모든 입력 신호에서 영향을 받음

  - 구현하기

    ```python
    def softmax(a):
    	exp_a = np.exp(a) # 지수 함수
    	sum_exp_a = np.sum(exp_a) # 분모_지수 함수의 합
    	y = exp_a / sum_exp_a # softmax 함수
    	
    	return y
    ```

##### 소프트맥스 함수 구현 시 주의점

- 오버플로우 문제

  - softmax function가 사용하는 지수 함수는 쉽게 아주 큰 값을 내뱉음

    => 큰 값끼리 나눗셈을 하면 결과 수치가 **불안정**해짐

    :question:컴퓨터가 표현할 수 있는 값이 한정되어 있어서

- 소프트맥스 함수 개선

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231214012610.png" alt="image-20211231214012610" style="zoom:60%;" />

  - 변형 1) C라는 임의의 정수를 분모 분자에 곱함
  - 변형 2) C를 지수함수 안으로 옮겨 log C로 만듦
  - 변형 3) log C를 C'로 치환

  :point_right: 소프트맥스의 지수 함수를 계산할 때, 어떠한 정수(C')를 더하든 빼든 결과가 바뀌지 않음을 의미

  - 오버플로우를 막을 목적이므로 입력 신호 중 **최대값**을 이용하는 것이 일반적임

- 개선된 소프트맥스 함수 구현

  ```
  def modified_softmax(a):
  	c = np.max(a) # 최대값 반환
  	exp_a = np.exp(a-c) # 오버플로우 대책 지수함수
  	sum_exp_a = np.sum(exp_a) # 지수함수의 합
  	y = exp_a / sum_exp_a # 소프트맥스 함수
  ```

  - 최대값(c)를 빼줌으로써 지수함수에 들어갈 값의 스케일을 줄임

    :point_right: 오버플로우 대책 마련!

##### 소프트맥스 함수의 특징

- 출력 총합이 **1** :point_right: 소프트맥스 함수의 출력을 '확률'관점으로 해석 가능(비중함수)

  - 문제를 확률적(통계적)으로 대응할 수 있게 됨

  ex) 위의 코드의 출력을 확률 관점으로 보면

  - y[0]의 확률 => 1.8%
  - y[1]의 확률 => 24.5%
  - y[2]의 확률 => 73.6%
    - 두번째 노드(y[2])의 확률이 가장 높으니 답은 2번째 클래스라고 할 수 있음
    - 객관적인 지표로써 73.6%확률로 2번째 클래스, 24.5%확률로 1번째 클래스, 1.8%확률로 0번째 클래스라는 확률적인 결론도 낼 수 있음

- 소프트맥스 함수를 적용해도 각 원소의 대소 관계는 변하지 않음

  - 이유 : exp()가 단조 증가 함수라서
    - f(x) = exp(x)
    - a < b => f(a) < f(b)
    - <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/Exp.svg/1200px-Exp.svg.png" alt="지수 함수 - 위키백과, 우리 모두의 백과사전" style="zoom:20%;" />

- 신경망으로 분류할 때는 출력층의 소프트맥스 함수 생략 가능

  - 신경망을 이용한 분류에서는
    - 가장 큰 출력을 내는(확률이 높은) 뉴런에 해당하는 클래스로만 인식
  - 신경망으로 분류 문제를 풀 때는 출력층의 소프트맥스 함수 생략
    - 가장 큰 노드 입력과 가장 큰 노드 출력의 위치 고정 => np.exp()계산에 필요한 cost를 줄이기 위해 생략

  - 주의:exclamation:
    - 기계학습의 문제 풀이
      1. 학습(learning)
         - 모델 학습
         - 네트워크의 파라미터 조정
         - 출력층에서 소프트맥스 함수 **사용**
      2. 추론(inference)
         - 학습된 모델로 미지의 데이터에 대한 추론(분류)를 수행
         - 모델이 잘 학습됐는지 test-set으로 검증
         - 컴퓨팅 자원 효율을 위해 소프트맥스 함수 **생략**

##### 출력층의 뉴런 수 정하기

- 출력층의 뉴런 수는 풀려는 문제에 맞게 적절히 정해야 함

  - 분류(classification) 문제 

    - 분류하고 싶은 클래스 개수 = 뉴런 개수

    - ex) 0~9 숫자 이미지 분류 문제
      - 분류 클래스 개수 = 출력층의 뉴런 개수 = 10개
      - <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231223426765.png" alt="image-20211231223426765" style="zoom: 50%;" />
        - 뉴런의 색이 진할수록 높은 출력값을 의미
        - 해당 NN(Neural Network)모델은 입력층의 이미지를 숫자2로 판단함



#### 손글씨 숫자 인식

- 추론 과정을 신경망의 **순전파**라고 함
- 이번 절에선 사전에 학습된 모델을 사용해 학습 단계를 건너 뜀

##### MNIST 데이터 셋

- MNIST 

  - 손글씨 숫자 이미지 집합으로 기계학습 분야에선 유명..!
  - 0부터 9까지의 숫자 이미지로 구성됨
    - 훈련 이미지 : 60k장
    - 시험 이미지 : 10k장
  - 특징 
  
    - gray-level 이미지
  - size : 28 x 28
    - 픽셀 : 0~255
  - 각 이미지에 실제 숫자가 레이블로 붙어 있음

- MNIST 데이터셋 파일을 다운 받아 dataset디렉토리에 저장

  | 파일                       | 목적                                                         |
  | -------------------------- | ------------------------------------------------------------ |
  | train-image-idx3-ubyte.gx  | 학습 셋 이미지 - 55000개의 트레이닝 이미지, 5000개의 검증 이미지 |
  | train-labels-idx1-ubyte.gz | 이미지와 매칭되는 학습 셋 레이블                             |
  | t10k-images-idx3-ubyte.gz  | 테스트 셋 이미지- 10000개의 이미지                           |
  | t10k-labels-idx1-ubyte.gz  | 이미지와 매칭되는 테스트 셋 레이블                           |

##### MNIST 데이터셋 이미지 가져오기

- ![image-20220105002018229](C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220105002018229.png)
  - `sys.path` 
    - 파이썬 모듈이 저장되어 있는 위치를 반환
    - System이 해당 경로들을 scope하고 있다는 의미 => 해당 위치에 있는 모든 모듈을 경로 설정없이 바로 불러올 수 있음!
  - `os.pardir` 
    - 부모 디렉터리 경로

##### 코드 설명

- `mnist.py` 
  - `load_mnist()` : MNIST 데이터셋의 이미지를 **numpy**배열로 변환해주는 함수 

##### IDX file format

- MNIST 데이터셋 확장자
  - .idx1-ubyte : 라벨링 파일 (0-9)
  - .idx3-ubyte : 이미지 파일 (0-255)

##### load_mnist

- ```python
  def load_mnist(normalize=True, flatten=True, one_hot_label=False):
      """MNIST 데이터셋 읽기
  
      Parameters
      ----------
      normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
      one_hot_label :
          one_hot_label이 True면, 레이블을 원-핫(one-hot) 배열로 돌려준다.
          one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
      flatten : 입력 이미지를 1차원 배열로 만들지를 정한다.
  
      Returns
      ------
      (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
      """
      if not os.path.exists(save_file):
          init_mnist()
  
      with open(save_file, 'rb') as f:
          dataset = pickle.load(f)
  
      if normalize:
          for key in ('train_img', 'test_img'):
              dataset[key] = dataset[key].astype(np.float32)
              dataset[key] /= 255.0
  
      if one_hot_label:
          dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
          dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
  
      if not flatten:
           for key in ('train_img', 'test_img'):
              dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
  
      return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
  
  if __name__ == '__main__':
      init_mnist()
  
  ```

  - load_mnist는 함수 객체임

  - `load_mnist(normalize=True, flatten=True, one_hot_label=False):`

    - load_mnist() 함수는 위와 같은 default parameter를 가지고 있음

  - `normalize`

    - 입력 이미지의 pixel value를 0.0~1.0 사이 값으로 정규화
      - normalize == False이면 데이터 원형 그대로 0~255 사이 값 유지

  - `flatten` 

    - 입력 이미지를 평탄하게(1차원 배열로) 만듦
      - flatten==False이면 데이터 원형 그대로 1x28x28의 3D배열
      - flatten==True이면 배열을 일자로 늘려서 784개의 원소로 이루어진 1D배열

  - `one_hot_label`

    - 레이블을 원-핫 인코딩형태로 저장(라벨링 데이터의 저장 형식을 설정)
      - one_hot_label==False이면 데이터 원형 그대로 라벨링 값을 정수 형태로 저장
      - one_hot_label==True이면
        - 원래 라벨링이 '7' : [0,0,0,0,0,0,0,1,0,0]

    :question:원-핫 인코딩 (one-hot encoding)

    - 한 놈만 핫하다(= 한 놈만 정답이다)
    - 정답을 뜻하는 원소만 1, 나머진 모두 0

- 직렬화 (serialization)

  - 자료구조(객체)를 파일로 저장하기 위해 변환하는 것
  - 객체를 이진(binary)이나 텍스트 포맷으로 변환( 형식 변환 )
    - 구조화된 데이터(ex. 인메모리 구조체)를 저장하거나 전송하기 위한 형식으로 변환하기 위한 개념
  - 변환된 데이터는 나중에 동일 시스템(os)이나 타 시스템에서 재구성해 객체를 복제할 수 있음

- pickle 모듈

  - 파이썬 언어로된 가장 기본 **직렬화 컨버터**모듈

    - 서로 다른 코딩언어/자료형 간 객체를 직렬화 시키는 도구

  - 파이썬 언어로된 버퍼 프로토콜이기도 함

  - 일시적으로 존재하는 객체는 pickle 모듈을 사용할 수 없음

    ex) function, method, class, pip ...

  - `pickle.load(<직렬화 된 객체>)` : 직렬화된 객체를 복원하는 메소드

##### MNIST 이미지 출력하기

- ```python
  import numpy as np
  import sys, os
  sys.path.append(os.pardir) # 부모 디렉토리의 모든 파일 위치 추가
  
  from dataset.mnist import load_mnist # MNIST 데이터 읽어서 numpy.ndarray 객체로 변환시키는 모듈
  
  from PIL import Image # PIL : Python Image Library
  
  def img_show(img):
  	pil_img = Image.fromarray(np.unit8(img))
  	pil_img.show()
  
  (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) # 데이터 읽어옴 (IDX format -> numpy.ndarray 객체)
  
  x_train # 60000장의 이미지를 28x28로 평활화 시킨 것
  t_train # 각 원소는 레이블을 나타냄
  
  img = img.reshape(28,28)
  img_show(img) # 이미지 출력
  ```

  - 주의할 점
    - flatten=True로 설정해서 이미지 객체를 1차원 넘파이로 불러왔을 때 이미지로 출력하려면 다시 원래 형태인 28x28로 변형해야함
      - `np.reshape( , )`
    - OpenCV 이외의 PIL모듈로 이미지 출력할 땐 넘파이 객체를 PIL용 데이터 객체로 변환해야 함
      - `Image.fromarray()`

##### MNIST 숫자 분류 신경망 구조

- 입력층 

  - 뉴런 784개 ( 28 x 28 사이즈 이미지 => 784 x 1 배열로 평활화)

- 은닉층 2개 - 저자가 임의로 정함

  - 첫번째 layer : 뉴런 50개
  - 두번째 layer : 뉴런 100개

- 출력층 

  - 뉴런 10개 (0-9)

- <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220105011821029.png" alt="image-20220105011821029" style="zoom:80%;" />

- MNIST 데이터셋을 활용한 추론을 수행하는 신경망 구현

  ```python
  import numpy as np
  import pickle
  import sys, os
  sys.path.append(os.pardir) # 부모 디렉토리의 모든 파일 위치 추가
  
  from dataset.mnist import load_mnist # MNIST 데이터 읽어서 numpy.ndarray 객체로 변환시키는 모듈
  
  #(1) MNIST 데이터를 np.ndarray 객체로 불러오기
  def get_data():
      (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
      return x_test, t_test 
  
  #(2) 사전 학습된 가중치 파일 불러오기
  def init_network():
      with open('sample_weight.pkl',mode='rb') as f:
          network = pickel.load(f)
         
      return network # dictionary 형태의 객체 
  
  #(3) 추론
  def predict(network, x):
      W1, W2, W3 = network['W1'].network['W2'].network['W3'] #weight 가중치
      b1, b2, b3 = network['b1'].network['b2'].network['b3'] #bias 편향
      
      a1 = np.dot(x, W1) + b1
      z1 = sigmoid(a1)
      a2 = np.dot(z1,W2) + b2
      z2 = sigmoid(a2)
      a3 = np.dot(z2, W3) + b3
      y = softmax(a3)
      
      return y
  
  # 활성화 함수 정의
  def sigmoid(x):
      return 1 / (1+np.exp(-x))
  
  def softmax(x):
      if x.ndim == 2: 
          x = x.T # 전치
          x = x - np.max(x, axis =0) # 각 열의 최댓값을 원소마다 빼서 지수함수의 스케일 줄임
          y = np.exp(x) / np.sum(np.exp(x), axis = 0) # softmax함수
          
  # main()
  x, label = get_data()
  network = init_network() # 사전 학습된 가중치 불러오기
  
  accuracy_cnt = 0 # 정확하게 예측한 것 카운트
  
  for i in range(len(x)):
      y = predict(network, x[i]) # x[i] : 평활화된 i번째 이미지
      print("확률 = ", y) #출력된 클래스별 확률
      print("\n")
      max_idx = np.argmax(y) # 값이 가장 큰 원소의 인덱스 
      
      if max_idx == label[i]:
          accurac_cnt += 1
  ```

  - 출력된 확률값 다루기

    - 확률이 높은 클래스 선택!

  - 레이블 == 신경망 예측 클래스

    => accuracy_cnt 증가

    ```python
    print("Accuracy: " + str(float(accuracy_cnt) / len(x))) # 정확도 계산
    ```

    Accuracy : 0.9352

    

##### 정규화 & 전처리

- 정규화 (Normalization) : 데이터를 특정 범위로 변환하는 처리
- 전처리 (Pre-processing) : 데이터에 입력 전 특정 변환을 가하는 것
- 데이터 전처리로 정규화 시키는 장점?
  - 식별 능력 개선
  - 학습 속도 향상
  - 해석 용이
- 현업에서는 데이터 전체의 분포를 고려해 전처리를 가함



:memo:Quiz

1. 다음은 계단 함수의 그래프를 그려주는 코드이다. 이 중 일부를 설명하시오

   ```python
   def step_function(x):
   	return np.array(x >0 , dtype = np.int)
   ```

   :point_right: 배열 x의 각 원소가 0보다 크면 1을, 0보다 작으면 0을 리턴

2. 

