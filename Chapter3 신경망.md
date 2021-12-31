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

![img](http://mathurl.com/render.cgi?y%20%3D%20%5Cbegin%7Bcases%7D0%28b+%20w_%7B1%7Dx_%7B1%7D%20+%20w_%7B2%7Dx_%7B2%7D%5Cleq0%29%0A%20%26%20%20%20%5C%5C%201%28%20b%20+%20w_%7B1%7Dx_%7B1%7D%20+%20w%7B2%7Dx%7B2%7D%5Cgeq0%29%0A%20%26%20%20%0A%5Cend%7Bcases%7D%0A%5Cnocache)

- b : **편향**(뉴런이 얼마나 쉽게 활성화되느냐를 제어)
- w1,w2 : 각 신호의 가중치 (신호의 영향력을 제어)

<img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211228011020885.png" alt="image-20211228011020885" style="zoom:50%;" />

- h(x)함수는 입력이 0을 넘으면 1을 돌려주고 그렇지 않으면 0을 돌려줌

- 활성화 함수의 등장

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

#### 활성화 함수

##### 시그모이드 함수 

- 신경망에서 자주 이용하는 활성화 함수
- <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211231153423938.png" alt="image-20211231153423938" style="zoom:70%;" />

##### 계단 함수 구현하기

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
      return np.array(x > 0, dtype = np.int)
  
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

- 시그모이드보다 최근엔 **ReLu** 함수 

- <img src="https://machinelearningmastery.com/wp-content/uploads/2018/10/Line-Plot-of-Rectified-Linear-Activation-for-Negative-and-Positive-Inputs.png" alt="A Gentle Introduction to the Rectified Linear Unit (ReLU)" style="zoom: 33%;" />

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

