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

기존에 요소가 1개인 경우에는 아래와 같은 선형 하강법을 적요하면 되었다.
$$b=b-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_b(x^{(i)})-y^{(i)})$$
