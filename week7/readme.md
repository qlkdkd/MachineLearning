# Forward prop in a single layer

## forward prop(coffee roasting model)
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/17a0d154-8489-419b-9344-7a41bd744233)
$\overrightarrow{x}$ `np.array([200, 17)]`

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
