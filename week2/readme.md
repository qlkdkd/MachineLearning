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

---

# Linear Regression Model Part 2
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/28372b4f-d2ff-4512-a981-d9fed8e88235)
f함수 표방법:
$$f_{w, b}(x)=wx+b$$
* 간단히 $f(x)=wx+b$라고 표현해도 됨.
* 함수 이름: 선형 함수(linear Function) ->**Linear Regression(선형 회귀)**

---

# Cost Function
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/960f3386-6e73-4dac-b353-030e9d160665)
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/d9fcc5a5-3030-45de-8421-a64995528c4b)
* 비용 함수는 주어진 훈련 세트에 가장 적합한 일차(선형)함수를 알아낼 수 있게 해준다. 앞서 설펴본 함수에서 우리는 매개변수(Parameter)에 해당하는 $w$, $b$를 결정해야 한다. 우리가 선택하는 매개변수의 값에 따라서 다른 f함수를 가지게 된다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/9f2eb031-5584-40a8-926c-a6b1e8c14aa9)
### 우리가 선택한 파라미터가 데이터와 잘 일치하는 지 알 수 있는 법
* 최소: $(f(x)-y)^2$ 가 되는 것이 합리적
      * $f(x)-y$ 가 최소가 된다면 음수값이 나올 수 있음.
* 실제 훈련 세트는 1~m까지 존재하기 때문에 전부 더해줌.
$$J_{w, b}=\frac{1}{2m}\sum_{i=1}^{m}(f_{w, b}(x_{(i)})-y^{(i)})^2$$
* 우리가 구한 J함수가 바로 **비용함수**(Cost function)이다. 이 함수는 제곱 오차 함수 또는 평균 제곱 오차라고도 불린다. 비용함수는 대부분의 회귀 문제에서 적절하고 통상적인 방법이지만, 다른 비용함수들 또한 적절하다.

---
