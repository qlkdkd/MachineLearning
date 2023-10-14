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

# Cost Function Intuition
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/0362c63d-6894-495a-94d6-de1bc10fba78)
### Simplified
* 우선, 직관적으로 접근하기 위해서 $w=0$ 으로 설정하여 f함수를 단순화한다. 훈련 세트도(1, 1), (2, 2), (3, 3)이라고 가정해보자.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/90fef331-0b21-464e-999c-bddc4da76041)
* w=0이 될 경우, $J(w)=\frac{1}{2m}\sum_{i=1}{m}(f_w(x^{(i)}-y^{(i)})^2=\frac{1}{2m}(0^2+0^2+0^2)=0$가 되어버려 훈련 세트와 일치하지 않는다. 따라서 w!=0이다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/ed71087d-8050-492a-bc80-a907377f75d0)
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/5817be2d-db9d-4168-9a43-b5a32c118791)
* w=1일 경우 $J(w)=\frac{1}{2m}\sum_{i=1}{m}(f_w(x^{(i)}-y^{(i)})^2=x$이 되어 훈련세트와 일치한다.

---

# Visualizing the Cost Function
이제 $b$가 존재하는 다변수 선형 회귀를 생각해보자.
* 함수: $f_{w, b}(x)=wb+x(b\neq 0)$
* 파라미터: $w, b$
* 비용함수: $J(w, b)=\frac{1}{2}\sum_{i=1}{m}(f_{w, b}(x^{(i)}-y^{(i)})^2$
* 목표: $minimize_{w, b}J(w, b)$
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/30892934-8a42-4aa1-9df4-0e94f97ee336)

이제 파라미터가 2개가 되었고, 이전보다 J함수의 그래프는 더욱 복잡해질 것이다.
마찬가지로 포물선과 같은 형태이긴 하지만 아래와 같이 3차원 그래프로 나타낼 수 있을 것이다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/ba2e07f7-52f8-4d73-b9f2-3b00dcf89e2d)

이처럼 비용함수를 표현하면 w, b, J(w, b) 총 3가지를 표현해야하므로 위와 같이 3차원의 그래프가 된다.
우리에게 필요한 비용함수의 값은 (x, y)평면으로부터의 높이가 된다.
 우리는 비용함수를 더 편리하게 보기 위해서 위와 같은 3차원 도형이 아닌 3차원의 그래프를 (x, y)평면으로 정사영시킨 
등고선 그래프(Contour Plots, Contour Figures)를 활용할 것이다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/53c01e65-1704-4488-b80d-d422ac289562)

---

# Visualization examples
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/7f3ba07f-9ded-4253-b222-03a6ed4747d6)
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/524dcc87-4303-49a9-88f9-a8564434edaf)

# Gradient Descent
* 경사하강법(gradient descent)은 비용함수의 최솟값을 구하는 알고리즘이다(즉, 최소가 되도록 하는 w, b를 구하는 알고리즘이다.). 이 알고리즘은 선형 회귀 뿐만 아니라 대부분의 머신러닝에서 실제로 사용되는 알고리즘이다.

* 경사하강법 알고리즘은 다음과 같은 방법으로 진행된다.
      * $J(w, b)$(선형회귀나 아무 함수)
      * 원하는 결과: $min_{w, b}J(w, b)$
* w와 b로 시작($w=0, j=0$으로 시작)
* $J(w, b))$를 줄이기 위해 $w$, $b$를 계속 변경하여 최소한으로 끝날때까지 진행

* 즉, 임의의 초기값으로 시작하여 최소의 비용함수의 값을 찾을 때 까지 w, b를 변경시킨다. 임의의 초기값을 기준으로 최소가 되는 점을 찾아내는 알고리즘이다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/a0d6e859-9bc1-43ce-ad55-ec150cf807b9)

---

# Implementing Gradient Descent
### 경사하강법 알고리즘
$$w=w-\alpha\frac{\partial}{\partial w}J(w, b)$$
$$b=b-\alpha\frac{\partial}{\partial b}J(w, b)$$

   * =: 할당 연산자
   * $\alpha$: 학습률
   * \frac{\partial}{\partial w(또는 b)}$: 미분 계수
* 올바른 예시
$$tmp_w=w-\alpha\frac{\partial}{\partial w}J(w, b)$$ 
$$tmp_b=b-\alpha\frac{\partial}{\partial b}J(w, b)$$
($w=tmp_w$)
($b=tmp_b$)
* 잘못된 예시
$$tmp_w=w-\alpha\frac{\partial}{\partial w}J(w, b)$$($w=tmp_w$)
$$tmp_b=b-\alpha\frac{\partial}{\partial b}J(w, b)$$($b=tmp_b$)

---

# Gradient Descent Intuition
* $J(w)$
* $w=w-\alpha\frac{\partial}{\partial w}J(w)$
* $min_wJ(w)$
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/9eb3a8da-0c4c-4a88-a151-53441a66ac88)
