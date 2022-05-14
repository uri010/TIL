# Git사용법

## GitHub 연동

1. github에 respository 생성

2. local에 연결할 폴더 생성

3. git bash에서 해당 폴더로 이동

   ```
   $ cd test
   ```

4. 해당 폴더에 git 저장소 생성(초기화)

   ```
   $ git init
   ```

5. repository 연결

   ```
   $ git remote add origin https://github.com/username/repositoryname.git
   ```

6. 연결 확인

   ```
   $ git remote -v
   ```
   
7. 기본 branch master에서 main으로 바꾸기

   ```
   $ git branch -M main
   ```

8. add, commit 후 main branch에 push하기

   ```
   $ git push -u origin main
   ```

## 트러블 슈팅

#### git push 에러

![image-20220420111814699](C:\Users\a9681\AppData\Roaming\Typora\typora-user-images\image-20220420111814699.png)

- 원인

  - 데이터 유실 등 문제가 있을 수 있는 부분이 있어 git에서 처리 되지 않도록 에러를 띄우는 것
  - 내 경우엔 해당 github repository에 전에 올려놓은 파일이 있었고 새로 local 폴더를 만들어 다시 연결시켜준 것이라 데이터 유실의 문제가 있었음

- 해결방법

  ```
  $ git push -u origin +master
  ```
  
  
#### github push token 에러
- 원인
   - Mac에서는 github 연동할 때 token을 등록해줘야 했었다..!

- 해결방법
   - https://hyeo-noo.tistory.com/184 
   - 사이트 참고하깅

  
