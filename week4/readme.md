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

또한 $f(x)=0.7$을 거꾸로 말하면 양성 종양일 확률이 30%라는 의미이며, 일반적으로 $f(x)=P(y=0|x;\overrightarrow{w}, b)$로 표현한다.

---

# Decision Boundary
위의 가설함수 $f_{\overrightarrow{w}, b}(x)=g((\overrightarrow{w}, b)x), g(z)=\frac{1}{1+e^{-z}}$에서 
우리는 가설 함수의 값>=0.5이면 y=1, 가설함수의 값<=0.5이면 y=0이라고 추론할 수 있을 것이다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/4f9d8d53-d901-48c4-9e5d-98650b9cac72)

다음으로 $f_{overrightarrow{w}, b}(\overrightarrow{x})=g(z)=g(w_1x+w_2x_2+b)$의 가설 함수에 대해서 살펴보자.
만약 우리가 최적의 매개변수 w, b 를 찾았고, 그 값이 $w_1=1, w_2=1, b=-3$이라면 다음과 같다.

$z=x_1+x_2+3>=0$

따라서 $x_1+x_2>=3$이면 1로 분류하고, 그렇지 않으면 0으로 분류된다. 이는 아래 분홍색 선으로 표현된다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/ae15f3f8-12f5-4719-a256-32919775ecf4)
이때 결정 경계선은 $x_1+x_2-3=0$이다. 이제 새로운 데이터가 추가됬을 때, $x_1$과 $x_2$값을 찍어보고, 결정 경계선보다 위쪽에 찍히면 1, 아래쪽으로 찍히면 0으로 분류하면 되는 것이다.

결정 경계선은 w, b에 의해서 결정되는 것이다. 훈련 데이터는 매개변수 w, b를 결정하는 데 사용될 뿐, 결정 경계선에 직접적으로 영향을 끼치지 않는다.

### Non-liniar decision boundaries
위의 예제는 classification이 선형적으로 분리되어 있는 데이터이다. 그러나 선형으로 클래스를 구분할 수 없는 경우도 있다. 예를 들어 아래와 같이 결정 경계선을 설정해야 되는 경우도 있다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/4fdee821-951e-46cc-924d-dc9cd294a845)

이런 경우에 사용할 수 있는 한 가지 방법은 차수를 높혀서 다항식으로 표현하는 것이다. 이 방법을 사용하면 비선형 결정 경계선도 표현할 수 있다. 만약 우리가 이미 적절한 매개변수 값을 구했고, 그 값이 $b=-1, w_1=0, w_2=0, w_3=1, w_4=1$이라면 다음과 같이 예측할 수 있다.

```python
if (-1+x_1**2 + x_2**2)>=0: y=1
else: y=0
```
즉, 결정 경계선은 $x_1^2+x_2^2=1$이다.

또한, 아래와 같이 더욱 복잡한 결정 경계선을 갖는 경우도 있는데, 이러한 경우에도 차수를 높혀서 표현할 수 있다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/1c54eb4a-2312-4b4c-8737-1b2fd7b785c8)

---

# Cost Function for Logistic Regression
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/5779399d-c01c-41d5-ac9e-d0c79103a6d9)
이제 로지스틱 회귀를 하기 위해서 필요한 w, b값을 구하는 방법에 대해 알아보자. 
앞에서 배웠던 선형 회귀를 위한 비용함수는 $J(w, b)=\frac{1}{2m}\sum_{i=1}{m}(f_{w, b}(x^{(i)})-y^{(i)})$이며, 우리는 아래와 같이 표현할 수도 있다.
$$J(\overrightarrow{w}, b)=\frac{1}{2m}\sum_{i=1}{m}(f_{\overrightarrow{w}, b}(x^{(i)})-y^{(i)})^2$$

로지스틱 회귀 문제를 위해 
$f_{\overrightarrow{w}, b}(\overrightarrow{x})=\frac{1}{1+e^{-(\overrightarrow{w}\cdot\overrightarrow{x}+b)}$
를 대입하면 문제가 없을 것 같지만, 우리는 이 비용함수를 사용할 수 없다. 왜냐하면, 위의 과정으로 대입한 비용함수 $J(\overrightarrow{w}, b)$는 볼록함수가 아니기 때문이다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/ad301e1f-1623-4e3f-be25-a6d395a01590)

선형 회귀에서의 비용함수는 볼록함수 구조로 경사하겅법을 연산하면 수렴하면서 최솟값을 찾을 수 있었지만, 로지스틱 회귀에서는 많은 극소점을 가지기 때문에 최솟값을 보장할 수 없다. 따라서 우리는 로지스틱 회귀를 위한 비용함수를 다음과 같이 정의한다.

### Logistic loss function
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/35afe9f3-ef6d-460d-a1f4-6c25184898e1)

이처럼 비용함수는 y=1이거나 y=0일때만 정의되어 있다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/a4d8c646-af81-4fc1-95a9-527ad2722407)

y=1인 경우에 $f_{\overrightarrow{w}, b}(\overrightarrow{x})=1$이면 비용=0이지만, $f_{overrightarrow{w}, b}(\overrightarrow{x})=0$ 이면 비용이 무한대라는 것을 알 수 있다.
이것이 의미하는 것을 이해하는 것은 중요하다. 잘못된 예측을 한 경우이므로 매우 큰 비용의 패널티가 생기는 것을 의미한다.(종양을 판단함에 있어서, 악성 종양이지만 양성일 확률이 높다고 예측할 수 있다.)

반대로 y=0인 경우에는 그래프가 반대로 그려진다. $f_{\overrightarrow{w}, b}(\overrightarrow{x})=1$이라면 비용이 무한대라는 것을 알 수 있고,  $f_{\overrightarrow{w}, b}(\overrightarrow{x})=0$이면 비용이 0이라는 것을 알 수 있다. 위와 마찬가지로 $f_{\overrightarrow{w}, b}(\overrightarrow{x})=1$이라 판단하는 경우에는 위 그래프가 y=0인 클래스의 그래프이므로 예측이 완전히 잘못되었고, 그에 대한 패널티로 비용이 무한대가 되는 것을 볼 수 있다. 반대로 $f_{overrightarrow{w}, b}(\overrightarrow{x})=0$인 경우에는 y=0이라는 올바른 예측을 하고 있고, 비용이 0이라는 것을 볼 수 있다.

y는 오직 1 또는 0이라는 값만 가지므로, 나누어져 있는 비용함수를 다음과 같이 표현 가능하다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/f7d20dea-b0cb-47c5-8c90-1715e7264ad9)

즉, 하나의 식으로 표현해도, 우리는 y=0 or y=1인 클래스에 대해서만 고려하기 때문에 결과적으로 동일하다. 또한, 다른 비용함수들도 많지만, 해당 모델이 Maximum likehood estimation 원리로 유도되었고, 매개변수 w, b를 구하는데 꽤 효율적이기 때문에 많이 사용된다. 
정리하면 아래와 같이 비용함수를 구할 수 있다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/ba40b36b-56fa-4b1e-b7da-cb6825b1f897)
우리는 $J(\overrightarrow{w}, b)$를 최소로 만드는 최적의 w, b를 찾고, 이후에 새롭게 주어진 입력값 x를 어떤 클래스로 분류할지 판단하려면 $f_{\overrightarrow{w}, b}(\overrightarrow{x})=\frac{1}{1+e^{-(\overrightarrow{w}\cdot\overrightarrow{x}+b)}}$의 값이 0.5 보다 큰지 작은지 확인하면 된다.
