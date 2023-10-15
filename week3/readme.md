# Multiple Features
우리가 앞서 배운 선형회귀에서는 하나의 feature를 가진 x(집의 넓이)가 있었고, 그 x로 y(집의 가격)을 예측하는 것이었다.
 하지만, 집의 가격을 결정하는 요소(feature)로 '침실의 개수' 또는 집이 지어진 지 얼마나 오래되었는지도 알고 있다면
 가격을 결정하는 더 많은 요소(features)를 갖게 된다. 
이처럼 실제 상황에서는 하나의 변수로만은 예측하기 어려운 경우가 많고, 이번 시간에 여러 개의 변수를 이용한 **다변수 선형회귀**를 알아볼 것이다.

즉, 집의 가격을 예측하는 문제에서 집의 넓이뿐만 아니라, 방의 개수, 층수, 그리고 건물의 연식을 고려하여서 집의 가격을 예측할 것이다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/053269bd-31df-4b94-9108-112f997817d8)


* 변수가 하나일 때 함수는 $f(x)=wx+b$이었지만, 이제 변수가 여러개이고 함수를 일반화하면 다음과 같다.
$$f_{w, b}(x)=w_1x_1+w-2x_2+w_3x_3+...+w_nx_n$$
* 우리는 이 식을 행렬곱을 통하여 함수를 아래와 같이 간략하게 나타낼 수 있다. 편의상 $x_0^{(i)}=1, (i\in 1, 2, ..., m)$으로 정의하고,
* $x$, $w$가 모두 n+1차원이 되도록 하였다.
$$f_{\overrightarrow{w}, b}(\overrightarrow{x})=\overrightarrow{w}\cdot\overrightarrow{x}+b=w_1x_1+w_2x_2+...+w_nx_n$$

---

#  Vectorization Part 1
### 매개변수와 요소
* $\overrightarrow{w}=[w_1, w_2, w_3], (n=3)$
* $\overrightarrow{x}=[x_1, x_2, x_3], (n=3)$
#### 파이썬(넘파이)
```python
w=np.array([1.0, 2.5, -3.3])# w[0]=1.0, w[1]=2.5, w[2]=-3.3
b=4
x=np.array([10, 20, 30])# w[0]=10, w[1]=20, w[2]=30
f=np.dot(w, x)+b
```
#### 만약 벡터화 없이 코드를 짠다면...
$f_{\overrightarrow{w}, b}(\overrightarrow{x})=\sum_{j=1}^{n}w_jx_j+b$
```python
f=w[0]*x[0]+w[1]*x[1]+w[2]*x[2]+b
```
```python
f=0
for j in range(0, n):
  f=f_w[j]*x[j]
f=f+b
```

---

# Vectorization Part2
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/0a68b3c9-c5f8-4f81-96be-ad51b6ddf206)

### 다변수 경사하강법
매개변수
$$\overrightarrow{w}=(w_1, w_2, ..., w_16)$$
미분계수
$$\overrightarrow{d}=(d_1, d_2, ..., d_16)$$
```python
w=np.array([0.5, 1.3, ..., 3.4])
d=np.array([0.3, 0.2, ..., 0.4])
```
compute $w_j=w_j-0.1d_j$ for $f=1...16$

#### 벡터화 없이 계산
```python
w1=w1-0.1*d1
w2=w2-0.1*d2
...
w16=w16-0.1*d16
```
#### 벡터화로 계산
```python
w=w-0.1*d
```

---

# Gradient Descent for Multiple Regression
우리는 지금 다변수를 갖는 경우의 함수에 대해서 살펴보았다. 이제 우리는 가설 함수의 매개변수들을 어떻게 지정하는지 알아볼 것이고, 
경사하강법을 어떻게 적용할 것인지 살펴 볼 것이다.

기존에 요소가 1개인 경우에는 아래와 같은 경사 하강법을 적용하면 되었다.
$$w=w-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_w(x^{(i)})-y^{(i)})x^{(i)}$$
$$b=b-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_b(x^{(i)})-y^{(i)})$$

하지만 다변수 선형회귀의 경우에는 아래와 같은 경사하강법을 적용하면 된다.
* 반복
    * $w_j=w_j-\alpha\frac{\partial}{\partial w_j}J(\overrightarrow{w}, b)$
    * $b=b-\alpha\frac{\partial}{\partial b}J(\overrightarrow{w}, b)$
        * $J=\overrightarrow{w}\cdot\overrightarrow{x}+b$
        * $\overrightarrow{w}, j$는 지속적으로 업데이트 해야함
     
---

# Feature Scaling Part 1
Feature Scailing은 경사하강법을 잘 활용할 수 있는 방법 중 하나이다.
Feature Scailing의 개념은 서로 다른 요소라도 비슷하거나 같은 범위에 있다면 경사하강법은 더욱 빠르게 수렴할 수 있다는 것이다.
* $price=w_1x_1+w_2x+2+b$
    * $x_1$=size(feet^2), range: 300~2000
    * $x_2$=number of bedrooms, range: 0~5
* House: $x_1=2000$, $x_2=5$, price=\$500k

### 매개변수들의 크기
#### w1=50, w2=0.1, b-50
* $price=50\*2000+0.1\*5+50=$ \$100050.5k
#### w1=0.1, w2=50, b=50
* $price=0.1\*2000k+50\*5+50=$ \$500k -> 더 합리적임

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/f5c24c1b-ae2d-4757-b4e6-7cb3540cb350)
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/0224987b-0230-4d95-ad26-c07a10c5cacd)
* 위의 그림을 보면 예제 1을 보면 2개의 요소가 갖는 범위의 차이가 많이 나서 비용함수는 많이 찌그러진 타원의 모양을 갖게 되며, Global optima를 구하는데 오랜 시간이 걸리게 된다. 하지만 오른쪽 예제 2에서는 각 요소의 값(범위)를 최댓값이나 최댓값-최솟값 등과 같은 값으로 나누어진 값으로 치환하여 계산하면 비용함수의 그래프는 원에 가까운 모양이 될 것이고, 더 적은 선형하강법 연산으로 Global Optima를 찾을 수 있다.

# Checking Gradient Descent for Convergence
### 경사하강법
$$w_j=w_j-\alpha-\frac{\partial}{\partial w_j}J(\overrightarrow{w}, b)$$
$$b=b-\alpha-\frac{\partial}{partial b}J(\overrightarrow{w}, b)$$

### 정확히 작동하고 있는 경사하강법을 확실하게 하기\
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/f20ba301-e67e-4bff-ba64-9e4ace048fb3)
* 목표: $min_{\overightarrow{w}, b}, b}J(\overrightarrow{w}, b)$, $J(\overrightarrow{w}, b)$는 모든 반복 후 감소해야 한다.
* $J(\overrightarrow{w}, b)$는 숫자가 증가할수록 0에 수렴

#### 자동수렴 테스트
* $\epsilon=0.001$로 가정함
* 만약 $J(\overrightarrow{w}, b)$가 한 번의 반복에서 $\epsilon$만큼 감소하면 수렴을 선언함(광역 최솟값의 근삿값을 찾기 위한 매개변수 $\overrightarrow{w}, b$를 찾아야 함)

# Choosing the Learning Rate
이번에 우리는 디버깅(Debuggin) 작업을 통해 Gradient Descent가 적절하게 동작하기 위한 방법과 Learning Rate를 선택하는 방법에 대해서 알아볼 것이다. 이전에 배웠듯이 Gradient Descent의 역할은 Cost Function J를 최소화하는 파라미터를 찾는 것이며, 이것에 대한 디버깅 작업은 Gradient Descent를 적용할 때마다 변화하는 Cost Function J(\overrightarrow{w}, b)를 그래프로 그리면서 올바르게 감소하는지 확인하는 작업이다. Gradient Descent가 올바르게 동작한다면 그래프는 아래와 같은 모양이 될 것이다.

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/743ec032-1540-4692-9fa0-8474f36fd616)

* 비용함수가  진동하거나 양의 방향으로 상승하는 경우, 학습률이 너무 큰 것이다.
* 비용함수가 충분히 작은 경우, 학습률과 비용함수는 모든 반복에서 감소해야 한다.
* 만약 학습률이 너무 작으면, 경사하강법은 한 점에 모이기 위해 수많은 반복을 해야한다.

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/f206088f-2eef-4c4b-89cf-89cb4d3df827)
위 그래프처럼 매 경사하강법(매 반복)마다 비용함수의 값은 감소해야 한다. 그리고 반복이 진행될 수록 비용함수의 기울기는 0으로 수렴하게 될 것이다. 보통 몇 번의 반복이 필요한 지 알 수 없으므로 보통은 그래프를 그리며 수렴하는지를 확인한다. 또한, 수렴하는지 아닌지 자동으로 검사하는 알고리즘인 자동 수렴 테스트를 사용하여 수렴 여부를 파악할 수도 있다.

# Feature Engineering
이번에는 Feature를 간단하게 선택하는 방법과 적절한 feature의 선택으로 강력한 학습 알고리즘을 만드는 방법에 대해서 알아보자.
### 기능 엔지니어링
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/d9b957a7-e4ee-4c3e-b77a-4ebfe346010b)
* $f_{\overrightarrow{w}, b}(\overrightarrow{x})=w_1x_1+w_2x_2+b$($x_1$=가로길이, $x_2$=세로길이)
* $x_3=x_1x_2$, ($x_3$=넓이=가로*세로)
* 기능 엔지니어링: 직관을 사용하여 원래 기능을 변환하거나 결합하여 새로운 기능을 설계합니다.

# Polynomial Regression
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/3e3c3e4c-58b6-466b-a68d-73b640142938)
위와 같은 데이터 세트는 선형으로 표현하기에는 어려움이 있다. 이 데이터 세트를 표현할 수 있는 다른 모델들이 있는데, 그 중 하나는 2차 함수 모델로 표현할 수 있다. 하지만 2차 함수 모델은 최대값을 찍고 감소하는 형태이기 때문에, 집의 크기가 일정 크기 이상으로 커지면 집값이 감소하는 이상한 모델이 된다. 이는 데이터 세트와 잘 맞지 않을 것이다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/14145c2c-31d3-4880-befb-057e3ede6b6f)
비슷한 방식으로 제곱근 형태의 요소도 만들 수 있다.

주의할 점은 이렇게 요소를 만들면 feature scailing이 더 중요해진다는 것이다. 예를 들어, $x$의 원래 범위가 1~1000이였다면, 2차항의 범위는 10^6, 3차항의 범위는 10^9이 된다.


