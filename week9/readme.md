# TensorFlow implementation

## 텐서플로우를 이용한 신경망 훈
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/3548665f-9a09-4068-92b0-8cdebf103f39)
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([
    Dense(unit=25, activation='sigmoid')# 첫번째 은닉층(25개의 뉴런, 시그모이드 함수 계산)
    Dense(unit=15, activation='sigmoid')# 두번째 은닉층(15개의 뉴런, 시그모이드 함수 계산)
    Dense(unit=1, activation='sigmoid')# 세번째 은닉층(1개의 뉴런, 시그모이드 함수 계산)
])

form tensorflow.keras.losses import BinaryCrossentropy# loss 함수 가져오기

model.compile(loss=BinaryCrossentropy())

model.fit(X, Y, epochs=100)# X, Y: 입력, epochs: X, Y의 훈련 횟수
```

---

# Training Details

1. 함수 정하기($f_{\vec{w}, b}(\vec{x})=?$)
2. 손실/비용함수 정하기($L(f_{\vec{w}, b}(\vec{x}), y)$, $J(\vec{w}, b)=\frac{1}{m}\sum_{i=1}^{m}L(f_{\vec{w}, b}(\vec{x}^i), y^i)$)
3. $J(\vec{w}, b)$에 대해 경사하강법 진행

```python
# logistic regression 활용
import numpy as np

# 1. 함수 정하기
z=np.dot(w, x)+b

# 2. 손실/비용 함수
f_x=1/(1+np.exp(-z))

# 3. 경사하강법
w=w-alpha*dj_dw
b=b-alpha*dj+db

```

```python
# 텐서플로우를 이용한 신경망 훈련

# 1. 함수 정하기
model=Sequential([Dense(...) Dense(...), Dense(...)])

# 2. 손실/비용함수
model.compile(loss=BinaryCrossentropy())

# 3. 경사하강법
model.fit(X, y, epochs=100)
```
