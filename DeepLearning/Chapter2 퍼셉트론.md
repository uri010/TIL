#### Chapter 2 - 퍼셉트론

##### 뉴런이란?

<img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220106022529934.png" alt="image-20220106022529934" style="zoom:80%;" />

- Dendrite : 이웃 뉴런에서 전기 신호를 받음
- Synapse : 다른 뉴런과 Dendrite의 연결 부위에 있으며 **전기 신호의 세기를 재조정**함
- Soma(cell body) : Dendrite로부터 받은 여러 전기신호를 합침
- Axon : Soma의 전위가 일정 이상이 되면 이웃 뉴런으로 전기 신호를 보냄

##### 퍼셉트론이란?

- 다수의 신호를 받아 하나의 신호(강물처럼 흐름이 있는 것을 상상)를 출력

- 이 책에선 1을 '신호가 흐른다', 0을 '신호가 흐르지 않는다'로 씀

- <img src="https://t1.daumcdn.net/cfile/tistory/99BDCE4D5B98A1022C" alt="01. 퍼셉트론 - Perceptron" style="zoom:70%;" /> 

  - x1,x2는 입력 신호

  - y는 출력 신호

  - w1과 w2는 가중치 (신호의 세기를 재조정하는 Synapse의 역할)

  - 원은 뉴런 혹은 노드
  
  - 입력 신호가 뉴런에 보내질 때 가중치가 곱해짐
  
    - 뉴런에서 보내온 신호의 총합이 정해진 한계(**임계값**-theta)을 넘을 때만 1을 출력 - '뉴런이 활성화한다'라고도 표현
  
    <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211224113326941.png" alt="image-20211224113326941" style="zoom:67%;" />
  
- 수학적인 의미로는 선으로 영역을 **분리**



##### 단순한 논리 회로

- 값의 정의
  - 전류가 흐른다 = 1 = True
  - 전류가 흐르지 않는다 = 0 = False

- AND gate 

  - 만족하는 매개변수 조합은 무한히 많음 (w1, w2, θ)

    ex) (0.5, 0.5, 0.7) or ( 1.0, 1.0, 1.0) 등등 일 때 AND gate 만족

- NAND gate 

  - AND gate를 구현하는 매개변수의 부호를 모두 반전하면 NAND gate가 됨

    ex) (-0.5, -0.5, -0.7) 
  
- OR gate

  - 하나 이상이 1이면 출력이 1이 되는 원리

  - 가중치보다 임계값을 더 낮게 잡아줌

    ex) (0.5, 0.5, 0.3)
  
- XOR gate

  - 배타적 논리합
  - 논리값이 다를 때만 True, 같으면 False
  - **단층 퍼셉트론으로 구현하기 힘듦** :point_right: 다층 퍼셉트론으로 해결!



##### 논리회로의 기하학적 의미

- AND gate

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220106200540274.png" alt="image-20220106200540274" style="zoom:23%;" />

  - 0.5x1 + 0.5x2 >= 0.7 :point_right: x1/1.4 + x2/1.4 >= 1
  - 파란색 영역이 True값이 나오는 영역
  - 빨간색 점은 False값, 검정색 점은 True값

- NAND gate

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220106201132361.png" alt="image-20220106201132361" style="zoom:25%;" />

  - -0.5x1 - 0.5x2 <= 0.7에서 영역을 바꾸기 위해 등호 방향 바꿈 :point_right: -0.5x1 -0.5x2 >= 0.7

- OR gate

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220106201425060.png" alt="image-20220106201425060" style="zoom: 25%;" />

  - 0.5x1 + 0.5x2 >= 0.2 :point_right: x1/0.4 + x2/0.4 >= 1
  - 파란색 영역이 True값이 나오는 영역
  - 빨간색 점은 False값, 검정색 점은 True값
  - 임계점이 AND보다 낮음

- XOR gate 

  - 직선으로 영역을 나눌 수 없음

    :point_right: 곡선으로 나눔

    <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220106201956320.png" alt="image-20220106201956320" style="zoom: 25%;" />

##### 퍼셉트론 구현하기

- 가중치와 편향 도입

  - -θ를 b로 치환

    ![img](http://mathurl.com/render.cgi?y%20%3D%20%5Cbegin%7Bcases%7D0%28b+%20w_%7B1%7Dx_%7B1%7D%20+%20w_%7B2%7Dx_%7B2%7D%5Cleq0%29%0A%20%26%20%20%20%5C%5C%201%28%20b%20+%20w_%7B1%7Dx_%7B1%7D%20+%20w%7B2%7Dx%7B2%7D%5Cgeq0%29%0A%20%26%20%20%0A%5Cend%7Bcases%7D%0A%5Cnocache)

    - b를 **편향**(bias)라 하며 w1,w2는 그대로 가중치(weight)

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

- 논리회로 구현하기

  - `and_gate.py`

    ```python
    def AND(x1, x2):
    	x = np.array([x1, x2]) # 입력
    	w = np.array([0.5, 0.5]) # 가중치
    	b = -0.7 # -theta를 b로 치환
    	tmp = np.sum(w*x) + b # 가중치와 편향을 이용해 식 구현 0.5x1 + 0.5x2 - 0.7
    	if tmp <= 0:
    		return 0
    	else:
    		return 1
    
    # 다른 파일에서 이 파일을 불러와 쓸 때 아래 부분은 무시
    if __name__ == '__main__': 
        for xs in [{0,0}, {1,0}, {0,1}, {1,1}]:
           	y = AND(xs[0], xs[1])
            pint(str(xs) + ' -> ' + str(y)
     # (0,0) -> 0
     # (1,0) -> 0
     # (0,1) -> 0
     # (1,1) -> 1
    ```

  - `nand_gate.py`

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

  - `or_gate.py`

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

  - **가중치**(weight)는 각 입력 신호가 결과에 주는 영향력(중요도)를 조절
  - **편향**(bias)은 뉴런이 얼마나 쉽게 활성화(결과로 1을 출력)하느냐를 조절하는 매개변수



##### 다층 퍼셉트론이 충돌한다면

- 다층 퍼셉트론 : 퍼셉트론을 층을 쌓아 비선형 영역을 분리

- 기존 게이트 조합

  - XOR gate - AND, NAND, OR을 조합

    <img src="https://media.vlpt.us/images/citizenyves/post/698e42f0-66da-4d87-aa5f-f292fb7ac5c4/image.png" alt="Perceptron - 다층 퍼셉트론 XOR 게이트(python 구현)" style="zoom:40%;" />

    ```python
    from and_gate import AND # and_gate.py 파일의 AND함수 import
    from or_gate import OR
    from nand_gate import NAND
    
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



:memo:Quiz

1. 가중치 w1 = 0.3, w2 = 0.5와 임계값 θ = 0.6을 가지는 퍼셉트론을 생각하자. 이는 AND 게이트, NAND 게이트, OR 게이트 중 어느 것인지 답하고 이유를 설명하시오.

   :point_right: AND 게이트, 0.3x1 + 0.5x2 >= 0.6이라는 식을 생각해볼 때 x1과 x2가 모두 1일 때만 0.6이라는 임계값을 넘고 나머지는 모두 임계값을 넘지 못하기 때문이다.

   

2. 단층 퍼셉트론으로는 XOR 게이트를 구현할 수 없음을 기하학적으로 설명하시오.

   :point_right: 좌표평면 상에 있는 점 (1,1),(0,0)과 (1,0),(0,1)을 직선으로 나눌 수 없기 때문이다.

   

3. 적절한 가중치 w1, w2와 임계값 θ를 잡아 퍼셉트론을 이용하여 조건명제 p->q를 구현하시오.

   :point_right: <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220106204739624.png" alt="image-20220106204739624" style="zoom: 33%;" />

   

