## Chapter 1 헬로 파이썬

##### 프로그래밍 언어

> python 3.8.8



##### 사용할 외부 라이브러리

> matplotlib & NumPy



##### 아나콘다 배포판

- 배포판이란?

  : 필요한 라이브러리를 한 번에 설치할 수 있게 모아둔거

- 아나콘다는 데이터 분석에 중점을 둔 배포판



##### 인터프리터(interpreter)란?

- 소스코드를 컴파일 없이 한 줄 씩 읽어 내려가며 실행하는 프로그램
- python은 인터프리터 언어



##### 기본 문법들 

- 산술연산

  - `+`,`-`,`*`,`/`,`**` : 덧셈, 뺄셈, 곱셈, 나눗셈, 거듭제곱

- 자료형

  - `type()` : 특정 데이터의 자료형 출력

- 변수

  - python은 **동적언어**라서 따로 변수 자료형을 명시하지 않아도 됨
    - 변수의 자료형을 상황에 맞게 자동으로 결정
    - 자동 형변환 ex) 정수 x 실수 = 실수

- 주석 

  - `#`  : 한 줄 주석
  - `""" """`, `''' '''` : 여러 줄 주석
  - `Ctrl + /` or `ALT + 3/4` : 주석 단축키 
  - 주의 사항 
    - python은 들여쓰기로 if문이나 함수 등의 범위를 인식하기 때문에 **주석도 들여쓰기**를 해줘야 어느 함수의 주석인지 구분할 수 있음

- 리스트

  - `리스트명 = [요소1, 요소2, ...]` 

- 딕셔너리

  - `딕셔너리명 = { 키 : 값 } `

- bool

  - True 혹은 False 값을 가짐
  - `and`, `or`, `not` 연산자 사용 가능

- if문

  - ```python
    if 조건문1:
        ...
    elif 조건문2:
    	...
    else:
    	...
    ```

- for문

  - ```python
    for ... in 데이터집합:
    	...
    ```

    - 리스트 등 데이터 집합의 각 원소에 차례로 접근 가능

- 함수

  - ```python
    def 함수이름(parameter1, parameter2...):
    	 ...
    ```



##### 클래스

- 개발자가 직접 클래스를 정의하여 독자적인 자료형, 전용 함수와 속성 정의

- ```python
  class 클래스 이름:
      def __init__(self, parameter, ...): # 생성자
      	...
      def 메서드 이름1(self, parameter, ...): # 첫번째 인수는 자기 자신을 나타내는 self를 씀
          ...
      def 메서드 이름2(self, parameter, ...):
          ...
  ```

- `self.name`처럼 self다음에 속성 이름을 써서 **인스턴스 변수**를 작성하거나 접근 가능



##### NumPy

- 가져오기

  - ```python
    import numpy as np
    ```

- 배열 생성

  - ```python
    배열이름 = np.array(리스트) # 리스트를 인수로 받아 numpy.ndarray형태를 반환
    ```

- 산술 연산

  - 같은 크기의 배열에서 원소별(element-wise)로 산술 연산(+,-,*,/) 수행
  - 배열과 스칼라값의 산술 연산도 가능

- 브로드캐스트

  - 형상이 다른 배열끼리 계산할 수 있게 배열이 확대되는 기능
  - <img src="https://blog.kakaocdn.net/dn/x7p0b/btqBphEV3u0/FOdFjojXbRIxo8jxpXN8gK/img.jpg" alt="img" style="zoom:80%;" />

- 원소 접근

  - 인덱스로 접근

    ```python
    X = np.array([[51,55],[14,19],[0,4]])
    X[0] # 0 행
    X[0][1] # (0,1) 위치의 원소
    ```

    ```
    array([51,55])
    55
    ```

  - for문으로 접근

    ```python
    for row in 배열이름:
    ```

  - 인덱스를 배열로 지정

    ```python
    X = X.flatten() # X를 1차원 배열로 변환(평탄화)
    print(X)
    X[np.array([0,2,4])] # 인덱스가 0, 2, 4인 원소 얻기
    ```

    ```
    [51 55 14 19 0 4]
    arry([51, 14, 0])
    ```

  - 특정 조건 

    ```python
    X > 15
    X[X>15]
    ```

    ```
    array([ True, True, False, True, False, False]), dtype=bool)
    array([51, 55, 19])
    ```



##### matplotlib

- **matplotlib**은 그래프를 그려주는 라이브러리로 데이터를 시각화할 수 있음

- sin, cos함수 그리기

  ```python
  import numpy as np
  import matplotlib.pyplot as plt # 그래프 그리는 모듈
  
  y1 = np.sin(x)
  y2 = np.cos(x)
  
  #그래프 설정
  plt.plot(x,y1,label="sin") # x와 y1으로 구성된 sin 그래프
  plt.plot(x, y2, linestyle = "--", label = "cos") # cos 함수는 점선으로 그리기
  plt.xlabel("x") # x축 이름
  plt.ylabel("y") # y축 이름
  plt.title('sin & cos') # 그래프 제목
  plt.legend() # plot의 label을 범례로 사용, plt.legend(('sin','cos'))로도 사용 가능
  
  plt.show() # 그래프 그리기
  ```

  <img src="C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20211221112801941.png" alt="image-20211221112801941" style="zoom:70%;" />

- 이미지 표시

  ```python
  improt matplotlib.pyplot as plt
  from matplotlib.image import imread
  
  img = imread('image.jpg') # 이미지 읽어오기, 경로 지정 주의
  
  plt.imshow(img)
  plt.show()
  ```

  

