# Motivation
어떻게 classification 문제를 추론할 수 있을까? 우선 우리는 기존에 배웠던 linear regression(선형회귀) 모델을 가져와서 classification 문제에 적용을 해볼 것이다. 종양 크기에 대한 악성/양성 구분 문제를 예시로 살펴보자.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/97e27281-7b15-4a8d-be12-2059b16d8040)

기존에 배운 가설 함수를 통하여 $f_{w, b}=w^{(i)}x$로 나타낼 수 있고, 아래와 같이 데이터가 표현된다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/e2db3b2f-8860-452e-bd7f-d493589992a0)
### 파란그래프
이렇게 구한 가설함수의 임계값을 0.5로 설정하여, 가설함수의 값이 0.5 이상이면 '악성', 그보다 작으면 '양성'이라고 판별했다.
```python
if (f(w, b)<0.5): y=0
 else: y=1
```
### 녹색 그래프'
기존에 데이터에서 아주 큰 종양의 데이터가 추가되었다. 이때, 선형회귀의 결과는 초록색 직선이 되지만, threshold=0.5로 잡을 경우에 분류가 제대로 되지 않는다.
이처럼 분류 문제에 선형회귀 알고리즘을 사용하게 되면 실패하는 경우가 많다.

또 다른 문제로는, 분류 문제는 0과 1로만 분류가 되어야 하지만, 선형 회귀를 사용하게 되면 1보다 크더나 0보다 작은 값을 예측값  y로 추론할 수도 있다.

따라서 우리는 분류 문제를 해결하기 위해 선형회귀가 아닌 로지스틱 회귀에 대해서 배울 것이고, 이는 가설 함수가 0과 1 사이의 값만 취할 수 있도록 한다.

---

# Logistic Regression
우리는 로지스틱 회귀의 표현 모델에 대해서 알아볼 것이다. 앞서 말했듯이 로지스틱 회귀의 가설 함수는 0과 1사이의 값만 내보내는 형태가 되어야 한다.
즉, $0<=ff_{w, b}(x)<=1$의 조건을 만족해야 한다.

위의 조건을 만족해햐하는 상황에서 기존에 세운 가설 함수 $f_{w, b}(x)$는 0과 1 사이의 범위를 만족하지 않고, 잘못된 예측을 할 수 있기 때문에 적합하지 았다.
따라서 로지스틱 회귀에서는 $f_{w, b}(x)=g((w, b)x)$로 가설함수를 표현한다. 여기서 $g(z)=\frac{1}{1+e^{-z}}$이며, 정리하여 나타내면 아래와 같다.

$$f_{\overrightarrow{w}, b}(\overrightarrow{x})=g(\overrightarrow{w}\cdot\overrightarrow{x}+b)=\frac{1}{1+e^{-(\overrightarrow{w}\cdot\overrightarrow{x}+b)}}$$

이 그래프는 0과 1 사이의 값으로만 표현되며, 시그모이드 함수 또는 로지스틱 함수라고도 불린다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/e22acefb-f685-4f02-bbb6-b5ebfabbb1e8)

이 가설함수의 출력값은 주어진 입력값 x에 대해서 y가 1일 확률을 의미한다.
만약 $f(x)=0.7$이면, 해당 입력값이 악성 종양일 확률이 70%이라는 의미다.
일반적으로 다음과 같이 표현한다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/b3a615a4-b414-41b3-b1a8-9287caf50b4f)

또한 $f(x)=0.7$을 거꾸로 말하면 양성 종양일 확률이 30%라는 의미이며, 일반적으로 $f(x)=P(y=9|x;\overrightarrow{w}, b)$로 표현한다.

---

# Decision Boundary
