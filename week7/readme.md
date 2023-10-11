[지난주 내용 이어서...](https://github.com/qlkdkd/MachineLearning/blob/main/week6/readme.md)
# Forward prop in a single layer

## forward prop(coffee roasting model)
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/17a0d154-8489-419b-9344-7a41bd744233)

$\overrightarrow{x}=$ `np.array([200, 17)]`

$a_1^{[2]}=g(\overrightarrow{w}_1^{[2]}\cdot \overrightarrow{a}^{[1]}+b_1^{[2]}=$
```python
w2_1=np.array([-7, 8])
b2_1=np.array([3])
z2_1=np.dot(w2_1, ba1)+b2_1
a2_1=sigmoid(z2_1)
```

$a_1^{[1]}=g(\overrightarrow{w}_1^{[1]}\cdot \overrightarrow{x}+b_1^{[1]})=$
```python
w1_1=np.array([1, 2])
b1_1=np.array([-1])
z1_1=np.dot(w1_1, x)+b
a1_1=sigmoid(z1_1)
```

$a_2^{[1]}=g(\overrightarrow{w}_2^{[1]} \cdot \overrightarrow{x} +b_2^{[1]})= $
```python
w1_2=np.array([-3, 4])
b1_2=np.array([1])
z1_2=np.dot(w1_2, x)+b
a1_2=sigmoid(z1_2)
```

$a_3^{[1]}=g(\overrightarrow{w}_3^{[1]}\cdot \overrightarrow{x}+b_3^{[1]}=$
```python
w1_3=np.array([5, -6])
b1_3=np.array([2])
z1_3=np.dot(w1_3, x)+b
a1_3=sigmoid(z1_3)
```

->
```python
a1=np.array([a1_1, a1_2, a1_3])
```

---

# General implementation of forward propagation
## Forward prop in NumPy
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/9e632fb4-3b68-4837-bf06-74f9ed98de5f)
`W=np.array([[1, -3, 5], [2, 4, -6]])`

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/395b7429-5131-4045-b626-dd37590bb2a4)
`b=np.array([-1, 1, 2])`

$\overrightarrow{a}^{[0]}=\overrightarrow{x}$=
`a_in=np.array([-2, 4])`

```python
def dense(a_in, W, bm g):
  units=W.shape[1]#[0, 0, 0]
  a_out=np.zeros(units)
  for j in range(units):#0, 1, 2
    w=W[:, j]
    z=np.dot(w, a_in)+b
    a_out[j]=g(z)
  return a_out
#note: g() is defined outside of dense()
#(see optional lab for details)
```
```python
def sequential(x):
  a1=dense(x, W1, b1, g)
  a2=dense(a1, W2, b2, g)
  a3=dense(a2 W3, b3, g)
  a4=dense(a3, W4, b4, g)
  f_x=a4
  return f_x
```

> Capital `W` refers to a matrix
---

# Is there a path to AGI?
* AI
    * ANI(Artificial Narrow Intelligence[협의의 인공지능]): E.g., 스마트스피커, 자율주행 자동차, 웹서치, 공장에서 사용할 수 있는 ai
    * AGI(Artificial General Intelligence[일반적인 인공지능]): 인간이 할 수 있는 모든 것

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/cc6e455c-55f6-422d-8e99-9d629c33c8d8)
* 실제 뉴런
    * 입력->출력: 뉴런에서 뉴런으로
* 수학적으로 간단히 재현된 뉴런 모델
    * 입력->출력:
 
## 뉴런 신경망과 뇌
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/4c42e5c1-a2c0-4ed0-bf29-1e1e097c3355)
* 실제로 인간 뇌와 인공신경망의 거리는 아직 멀다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/6d77b7c6-04f4-416a-811b-5df033192030)
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/0fd0604c-e35f-465c-bbd5-14c8336e31b8)

하나의 학습 알고리즘 가설: 모든 지능의 행동이 시뮬레이션이 가능하다는 가설. 하나만 이해하면 모든 지능을 구현할 수 있다.
* 원리: 시각, 촉각 등의 정보를 뇌로 이동. 만약 청각 또는 촉각을 끊고 시각만 연결했는데도 청각/촉각적 기억을 한다.(체성감각피질)

## Sensor representations in the brain
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/8e8fa4e7-9e40-433e-a845-7107469f55b1)
* '혀에 전극을 주면 흑백 이미지를 학습할 수 있을까'라는 실험이 진행된 적 있음
* '사람들도 청각을 통해 인지할 수 있을까?'
* 햅틱 벨트: 가야되는 곳의 방향을 몸으로 기억할 수 있을까?
* 제 3의 눈을 뇌에 직접 인식하면 놀랍도록 적응력이 있다.

---

# How neural networks are implemented efficiently
## For loops vs vetorizetion
```python
# for loop
X=np.array(p200, 17)]
W=np.array([[1, -3, 5], [-2, 4, -6]])
b=np.array([-1, 1, 2])

def dense(a_in, W, b):
  a_out=np.zeros(units)
  for j in ragne(units):
    w=w[:, j]
    z=np.dot(W, X)+b[j]
    a[j]=g(z)
  return a
```
a

```python
#vectorization
X=np.array(p200, 17)]
W=np.array([[1, -3, 5], [-2, 4, -6]])
B=np.array([-1, 1, 2])

def dense(A_in, W, B):
  Z=np.matmul(A_in, W)# np.matmul=matrix multiplication
  A_out=g(Z)
  return A_out
```
[[1, 0, 1]]

# Matrix multiplication
## dot products
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/f946dff1-d93e-445d-b87a-93fcf9ae00a9)

## Vector matrix multiplication
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/e6f1c67e-9a46-4a3a-aaa6-015fe11e55e4)

## matrix matrix multiplication
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/165c0b7d-7baa-49ab-ac40-9a90af110aba)


# Matrix multiplication rules
## Matrix multiplication rules
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/24449364-498e-48df-a4aa-2449f05b7804)
* 2by3 mul 3by2
* $\overrightarrow{a}_1^T \overrightarrow{w}_1 =(1\times 3)+(2 \times 4)=11$
* row3 col2
    * $\overrightarrow{a}^T_3\overrightarrow{w_2}=(0.1 \times 5)+(0.2 \times 6)=1.7$
* row2 col3
    * $overrightarrow{a_2^T}\overrightarrow{w_3}=(-1 \times 7)+(-2 \times 8)=-23$
* $Z=A^TW=$![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/4ebf44bf-49dd-4883-aae4-d09753cdc89a)

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/1a8334d4-9e01-463b-afa7-d79a3c1751e5)

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/f590c46a-057f-4f38-8ea0-3722a1f45366)

