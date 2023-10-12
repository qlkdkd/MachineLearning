# Linear Regression Model Part1
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/b2cb579b-19c3-4020-8523-6e5bd0640df3)
* 그래프 주제: 집 크기에 따른 가격, **1250feet^2의 집의 적합한 가격 추측**
    * 이러한 상황에서 우리는 데이터에 맞는 모델을 찾기 위해 직선을 하나 그어서 그에 대응하는 가격을 추측할 수 있음.
* **이 방법을 선형 회귀(Linear regression)이라 함.**

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/65bb32a7-1705-47dc-a774-4dff5987e016)
* Training set: 위의 표와 같이 우리에게 주어지는 값들
* 위의 예시를 적용하면 각 집 크기에 해당하는 가격의 집합을 Training Set이라 함 ->이것을 활용하여 1250feet^2에 해당하는 가격을 얼마인지 알아내야 함.

 ### 용어
* $x$: 입력, 특징
* $y$: 출력 , "target" variable
* $(x, y)$ : 하나의 훈련 예제
* $(x^{(i)}, y^{(i)})$: i번째 훈련 예제
    * 예시)첫 번째: $x^{(1)}=2104, y^{(1)}=400 -> (x^{(1)}, y^{(1)})=(2104, 400)$
* m: 훈련 예제의 수
