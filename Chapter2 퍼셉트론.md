#### Chapter 2 - 퍼셉트론

##### 퍼셉트론이란?

- 다수의 신호를 받아 하나의 신호(강물처럼 흐름이 있는 것을 상상)를 출력

- 이 책에선 1을 '신호가 흐른다', 0을 '신호가 흐르지 않는다'로 씀

- <img src="https://t1.daumcdn.net/cfile/tistory/99BDCE4D5B98A1022C" alt="01. 퍼셉트론 - Perceptron" style="zoom:70%;" /> x1,x2는 입력 신호, y는 출력 신호, w1과 w2는 가중치, 원은 뉴런 혹은 노드

  - 입력 신호가 뉴런에 보내질 때 가중치가 곱해짐

    - 뉴런에서 보내온 신호의 총합이 정해진 한계(**임계값**-theta)을 넘을 때만 1을 출력 - '뉴런이 활성화한다'라고도 표현

    <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211224113326941.png" alt="image-20211224113326941" style="zoom:67%;" />

    

##### 단순한 논리 회로

- AND gate 

  - 만족하는 매개변수 조합은 무한히 많음 (w1, w2, θ)

    ex) (0.5, 0.5, 0.7) or ( 1.0, 1.0, 1.0) 등등 일 때 AND gate 만족

- NAND gate 

  - AND gate를 구현하는 매개변수의 부호를 모두 반전하면 NAND gate가 됨

    ex) (-0.5, -0.5, -0.7) 
  
- OR gate

  - 하나 이상이 1이면 출력이 1이 되는 원리

  - w1과 w2가 각각 θ보다 크면 될듯?

    ex) (0.5, 0.5, 0.3)



##### 퍼셉트론 구현하기

- 가중치와 편향 도입

  - θ를 -b로 치환

    ![img](http://mathurl.com/render.cgi?y%20%3D%20%5Cbegin%7Bcases%7D0%28b+%20w_%7B1%7Dx_%7B1%7D%20+%20w_%7B2%7Dx_%7B2%7D%5Cleq0%29%0A%20%26%20%20%20%5C%5C%201%28%20b%20+%20w_%7B1%7Dx_%7B1%7D%20+%20w%7B2%7Dx%7B2%7D%5Cgeq0%29%0A%20%26%20%20%0A%5Cend%7Bcases%7D%0A%5Cnocache)

    - -b를 **편향**(bias)라 하며 w1,w2는 그대로 가중치(weight)

  - NumPy로 구현

    ```python
    >>> import numpy as np
    >>> x = np.array ( [0,1] ) # 입력 
    >>> w = np.array([0.5,0.5]) # 가중치
    >>> b = -0.7 # 편향
    >>> w * x
    array([0. , 0.5])
    >>> np.sum(w*x)
    0.5
    >>> np.sum(w*x) +b
    -0.19999999999 # 대략 -0.2(부동소수점 수에 의한 연산 오차)
    ```

    - 각 배열의 원소끼리 곱한 값에 편향을 더함

- 가중치와 편향 구현하기

  - AND gate

    ```python
    def AND(x1, x2):
    	x = np.array([x1, x2])
    	w = np.array([0.5, 0.5])
    	b = -0.7 # -theta를 b로 치환
    	tmp = np.sum(w*x) + b # 가중치와 편향을 이용해 식 구현
    	if tmp <= 0:
    		return 0
    	else:
    		return 1
    ```

  - NAND gate

    ```python
    def NAND(x1, x2):
    	x = np.array([x1,x2])
    	w = np.array([-0.5, -0.5])
    	b = 0.7
    	tmp = np.sum(w*x) +B 
    	if tmp <= 0:
    		return 0
    	else:
    		return 1
    ```

  - OR gate

    ```python
    def OR(x1, x2):
        x = np.array([x1,x2])
        w = np.array([w1,w2])
        b = -0.2
        tmp = np.sum(w*x) + b
        if tmp <= 0:
        	return 0
        else:
        	return 1
    ```

    

- 편향과 가중치

  > 가중치는 각 입력 신호가 결과에 주는 영향력(중요도)를 조절, 편향은 뉴런이 얼마나 쉽게 활성화(결과로 1을 출력)하느냐를 조절하는 매개변수



##### 퍼셉트론의 한계

- XOR gate
  - **베타적 논리합**이라는 논리 회로
  - x1과 x2중 한 쪽이 1일 때만 1을 출력
  - 직선 하나로 영역을 나눌 수 없음! :point_right: XOR gate의 영역을 곡선을 나누면 됨
- 선형과 비선형
  - 선형 - 직선, 비선형 - 곡선



##### 다층 퍼셉트론이 충돌한다면

- 다층 퍼셉트론 : 퍼셉트론을 층을 쌓아 비선형 영역을 분리

- 기존 게이트 조합

  - XOR gate - AND, NAND, OR을 조합

    <img src="https://media.vlpt.us/images/citizenyves/post/698e42f0-66da-4d87-aa5f-f292fb7ac5c4/image.png" alt="Perceptron - 다층 퍼셉트론 XOR 게이트(python 구현)" style="zoom:40%;" />

    ```python
    def XOR(x1, x2):
    	s1 = NAND(x1,x2) # 전에 만들어 뒀던 함수 이용
    	s2 = OR(x1, x2)
    	y = AND(s1, s2)
    	return y
    ```

    <img src="https://media.vlpt.us/images/citizenyves/post/041d474a-9cd7-48e8-86d4-8f04cb5726b2/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-11-16%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2012.59.42.png?w=768" alt="Perceptron - 다층 퍼셉트론 XOR 게이트(python 구현)" style="zoom:50%;" />

    - 다층 구조의 네트워크
      1. 0층의 두 뉴런이 입력 신호를 받아 1층의 뉴런으로 신호를 보냄
      2. 1층의 뉴런이 2층의 뉴런으로 신호를 보내고, 2층의 뉴런은 y를 출력



