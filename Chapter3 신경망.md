#### Chapter 3 - 신경망

> 컴퓨터가 수행하는 복잡한 처리도 퍼셉트론으로 (이론상) 표현할 수 있지만 가중치를 설정하는 건 사람이 수동으로 해줘야 함 

>:point_right: 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력을 가진 신경망을 이용해 해결



##### 퍼셉트론에서 신경망으로

- 신경망의 예

  - <img src="https://blog.kakaocdn.net/dn/QfyS1/btqH7cdVOs5/GhDWPa5tGZOTjsEK5FXc9K/img.png" alt="02-1. 신경망 (1) - 3층 신경망 순전파 구현" style="zoom: 33%;" />

    - 입력층 : 가장 왼쪽 줄 (입력이 들어오는 층)

    - 출력층 : 가장 오른쪽 줄 (출력되는 층)

    - 은닉층 : 중간 줄 (눈에 보이지 않는 층)

      > 위 그림의 신경망은 3층으로 구성되지만 가중치를 갖는 층은 2개뿐이라 '2층 신경망'이라고 한다. 문헌에 따라 차이는 있을 수 있다.

- 퍼셉트론 복습

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



##### 활성화 함수

- 