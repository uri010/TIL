### Chapter 4 - 신경망 학습

- 학습 

  - 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득

- 손실 함수 ( =비용 함수 ) 

  - 머신 러닝의 가늠자

  - Loss function(=cost function)

  - 손실 함수의 결과값을 최소화 시키는 가중치 매개변수를 찾는 것이 목적!

    (손실 함수가 최소화되는 시점 -> 최적화된 가중치 매개변수)

#### 데이터에서 학습한다!

##### 데이터 주도 학습

- "기계학습은 데이터가 생명이다."

- 기계학습

  - 사람의 개입을 최소화하고 수집한 데이터로부터 패턴을 찾으려 시도

  - 주어진 데이터를 활용해 분석

    - 데이터에서 **특징**을 추출하고 그 특징의 패턴을 기계학습 기술로 학습

    - 특징

      - 입력 데이터에서 본질적인 데이터를 정확하게 추출할 수 있도록 설계된 변환기

      - 예시 - SIFT, SURF, HOG ...

    - 학습

      - 추출되니 특징의 패턴을 ML 기술이 학습
      - 예시 - SVM, KNN ...

  - 모아진 데이터로부터 규칙을 찾는 건 '기계'가 특징은 '사람'이 설계

- 학습법

  - 사람이 생각한 알고리즘
  - 사람이 생각한 특징 ( SIFT, HOG 등 ) -> 기계학습  (SVM, KNN 등)
  - 신경망(딥러닝)
    - 데이터를 **있는 그대로** 학습
    - 특징도 '기계'가 스스로 학습
    - 처음부터 결과를 출력할 때까지 사람의 개입이 없음 => end-to-end

##### 훈련 데이터와 시험 데이터

- 데이터셋 나누기
  - 데이터 = 훈련 데이터(training data) + 시험 데이터(test data)
  - 훈련 데이터 
    - 학습 단계에 사용함
  - 시험 데이터 
    - 훈련된 모델의 성능을 평가할 때 사용
- 나누는 이유?
  - 우리가 원하는 것은 **범용적**으로 사용할 수 있는 일반화된 모델이기 때문.
  - 시험 데이터는 범용 능력을 제대로 평가하기 위한 데이터
- 오버피팅 
  - 한 데이터에만 지나치게 최적화된 상태
  - 편견 덩어리, dogma가 심한 상태

#### 손실 함수

- 신경망 학습에서는 현재의 상태를 '하나의 지표'로 표현
  - 어떤 것의 상태나 성능을 객관적으로 평가하기 위함
- 손실 함수
  - 신경망 성능의 '나쁨 정도'를 평가하는 지표
  - 종류
    - MSE or CEE
  - 손실 함수에 마이너스만 곱하면 '얼마나 좋으냐'라는 지표로 나타낼 수 있음

##### 오차제곱합(SSE)

- 가장 많이 쓰이는 손실 함수

- <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220123165506578.png" alt="image-20220123165506578" style="zoom:50%;" />

  - k : 데이터의 차원 수 
    - 벡터 형태로 정리됨
    - ex) Y출력 = [y1, y2, y3, .... , y10] => k = 10
  - yk : 신경망의 출력 (신경망이 추정한 값)
  - tk : 정답 레이블 by one-hot encoding

- 사용 예시

  ```python
  import numpy as np
  
  def sum_squares_error(y, t):
  	return 0.5 * np.sum((y-t)**2)
  
  # 정답은 2
  t = [0,0,1,0,0,0,0,0,0,0] 
  y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 2일 확률이 가장 높다고 추정함 
  sum_squares_error(np.array(y), np.array(t))
  # 출력 : 0.097500...
  ```

##### 교차 엔트로피 오차(CEE)

- <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220123170042650.png" alt="image-20220123170042650" style="zoom:50%;" />

  - log는 밑이 e인 자연로그(ln = loge)
  - k : 데이터의 차원 수
  - yk : 신경망의 출력 ( 신경망이 추정한 값 )
  - tk : 정답 레이블 by one-hot encoding
    - 정답일 때의 출력(1이 있는 위치)이 전체 값을 결정
    - SSE의 경우 모든 뉴런의 출력이 손실함수 결과에 영향을 줬음

- 사용 예시

  ```python
  import numpy as np
  
  def cross_entropy_error(y, t):
      epsilon = 1e-7 # 엄청 작은 수 (엡실론)
      return -np.sum(t * np.log(y + epsilon))
  
  y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
  t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  cross_entropy_error(np.array(y), np.array(t))
  # 출력 : 0.5108...
  ```

  - epsilon
    - np.log() 함수에 0을 입력하면 마이너스 무한대를 뜻하는 -inf가 되어 계산이 불가
    - 아주 작은 값인 epsilon을 더해 0이 되지 않도록 해줌

##### 미니배치 학습

- 배치 처리 vs 미니배치 학습

  - 배치 처리 - 입력 데이터를 한번에 처리
  - 미니배치 학습 - 모든 데이터 중 일부 데이터의 손실 함수만 구하는 것
  - 미니배치 학습을 통해 근사치를 구할 수 있음

- 미니배치 학습 예시

  ```python
  import sys, os
  sys.path.append(os.pardir)
  import numpy as np
  from dataset.mnist import load_mnist
  
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
  
  print(x_train.shape) # (60000, 784) = (데이터 개수, 이미지 배열 28x28)
  print(t_train.shape) # (60000, 10) = (데이터 개수, 레이블 클래스 개수)
  ```

- 미니배치 고르기 예시

  ```python
  train_size = x_train.shape[0] # 전체 데이터 개수( 전체 batch ) 
  batch_size = 10 # 모집단 수
  
  batch_mask = np.random.choice(train_size, batch_size)
  print(batch_mask) # 0 ~ 59999 숫자 중 임의로 10개 고름
  
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]
  
  print(x_batch.shape) # (10, 784)
  print(t_batch.shape) # (10, 10)
  
  np.random.choice(60000, 10) # [0,60000] 수 중에서 무작위로 10개 추출
  ```

##### (배치용) 교차 엔트로피 오차 구현하기

- 데이터 하나에 대한 CEE 

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220123173713687.png" alt="image-20220123173713687" style="zoom:50%;" />

- 전체 데이터(batch용)에 대한 CEE

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220123173735827.png" alt="image-20220123173735827" style="zoom:50%;" />

- (배치용) 교차 엔트로피 오차 구현하기

  ```python
  def cross_entropy_error(y,t):
      if y.ndim == 1:
          t = t.reshape(1,t.size)
          y = y.reshape(1,y.size)
      
      batch_size = y.shape[0]
      return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
  ```

  - t가 0인 원소는 교차 엔트로피 오차도 0이므로 그 계산은 무시해도 좋음
  - 정답에 해당하는 신경망의 출력만으로 교차 엔트로피 오차를 계산할 수 있음
  - `y[np.arange(batch_size),t]`
    - 각 데이터의 정답 레이블에 해당하는 신경망의 출력을 추출
    - 위 예시에서는 [y[0.2], y[1.7], y[2.0], y[3.9], y[4,4]]인 넘파이 배열 생성

##### 왜 손실 함수를 설정하는가?

- 우리가 원하는 건 주어진 데이터를 **정확하고 정밀하게 예측하는 모델**을 만드는 것
  - 궁극적인 목표는 높은 **정확도**를 끌어내는 매개변수를 찾는 것
  - 정확도가 아닌 손실 함수의 값을 통해 우회적으로 평가하는 이유 => 미분과 관련됨
- 미분의 역할
  - 최적의 매개변수(편향, 가중치)를 탐색할 때 손실 함수의 값을 가능한 한 작게 하는 매개변수 값을 찾아야 함
  - 이때, 매개변수의 미분값(=기울기)을 계산하고, 그 **미분값을 단서**로 해서 매개변수 값을 서서히 갱신하는 과정을 **반복**
  - 손실함수의 미분이란
    - 가중치 매개변수의 값을 아주 조금 변화 시켰을 때, 손실 함수가 어떻게 변하냐 의 의미
  - 미분값이
    - 음수 - 가중치 매개변수를 양의 방향으로 변화시킴
    - 양수 - 가중치 매개변수를 음의 방향으로 변화시킴
    - 0 - 갱신 멈춤
- 정확도를 지표로 삼으면 안되는 이유
  - 미분 값이 대부분의 장소에서 0이 되어 매개변수를 갱신할 수 없기 때문
    - <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220123181030930.png" alt="image-20220123181030930" style="zoom:67%;" />
    - 정확도 함수 f(x) = x/100 -> f'(x) = 1 /100 = 0.01로 상수 함수라 변화가 없음
    - 매개변수를 약간만 조정해서는 정확도가 개선되지 않으며 개선되더라도 불연속값으로 바뀌어버림
- 손실 함수를 지표로 삼는 이유
  - 매개변수의 값이 조금 변하면 손실 함수 값도 연속적으로 변함

