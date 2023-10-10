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
