# Neurons and the brain

## Neural networks
* Origins: Algorithms that try to mimic the brain(1950's)

* Used in the 1980's and early 1990's. Fell out of favor in the late 1990's

* Resurgence from around 2005.

* speech -> images(2012, ImageNet) -> text(자연어처리)(NLP)

---

## Neurons in the brain
![](image1.png)

![](image2.png)

> dendrites: 수상돌기
> axon: 축석돌기

> 인공 뉴런 모델
> 여러 개의 뉴런 모델이 여러개의 입력을 받고, 숫자를 출력함

![](image3.png)
* 가로축: 데이터 양(Big Data 이용), 세로축: 학습 알고리즘의 정확성(컴퓨터 프로세서나 GPU가 성능을 좌우함)
* 기존 전통적 알고리즘은 아무리 많은 데이터가 있어도 정확성에 한계가 있다.
* 신경망이 커질수록(뉴런 수 증가) 성능이 계속 증가한다.

---

## Demand Prediction(수요 예측)
![](image.png)
* 기존 로지스틱 회귀를 사용하여 수요 예측
* $x=$price, $a=f(x)=\frac{1}{1+e^{-(wx+b)}}$

> $a$: 출력, top seller가 될 확률, $x$: 입력, 가격

![](image-1.png)
* 가격, 배송비 -> 경제성 |
* 마케팅 -> 인지도       | -> Top Seller가 될 확률
* 재료값 -> 품질         |

 (입력값)  (인공 뉴런)        (출력값)

 (4개)      (3개)             (1개)
  
            (층): 여러개의 뉴런을 갖고 있음.
            (출력 층): 마지막 뉴런
경제성, 인지도, 품질: 활성화 값. 생물학적 뉴런을 출력을 내보낼 때 연결되어있는 다른 뉴런을 활성화 시킬 때 사용됨
* 4개의 숫자가 입력되면 이 신경망 계층에서는 3개의 숫자가 계산되어 나오고, 3개의 숫자값이 출력층에 들어가서 하나의 값을 출력함.

![](image-2.png)

* 입력값: $\overrightarrow{x}$, 은닉층(hidden layer)(경제성, 인지도, 품질) -> $\overrightarrow{a}$, 출력층(output layer) $a$
* 4개의 입력 -> 3개의 출력, 3개의 입력 -> 1개의 출력(최종 결과)
> 잠재적으로 사용하고자 하는 특징들의 종류가 계산을 통해서 스스로 파악됨 -> 신경망이 강력한 알고리즘이라는 설명
![](image-3.png)

* 출력 층은 로지스틱 회귀를 풀고 있다. -> Top Seller가 될 확률 $a$를 출력.

* 좀 더 복잡한 특징 $x_1, x_2$를 계산할 특징(feature engineering) -> 수동적으로 계산

* 인공신경망의 특징: 자동적으로 학습을 통해 판단

![](image-4.png)
* $\mathbf{x}$가 입력되면 3개의 뉴런이 있는 첫 번째 은닉층에 들어가면 출력으로 $\mathbf{a}$가 나오고, $\mathbf{a}$가 2개의 뉴런이 있는 두 번째 은닉층에 들어가면 두번째 출력, 그리고 출력층에 들어가서 최종 결과 $a$ 출력.
> **Multilayer Perceptron (MLP)** 

---

## Example: Recognizing Images
![Alt text](image-5.png)
* 이미지 크기: 1000(px) $\times$ 1000(px)
* 밝기값: 0~255
* 각각의 행을 하나의 벡터로 만들기: $\mathbf{x}=\{197, 185, 203, ..., 187, 214\}$

![](image-6.png)
* 방금 전 벡터를 입력
* 첫번째 은닉층의 결과: 해당 방향의 직선을 찾음
* 두번째 은닉층의 결과: 얼굴의 일부를 찾아냄. 눈이 있는지, 코가 있는지, 귀의 모양은 어떤지...
* 세번째 은닉층의 결과: 얼굴 모양을 찾아냄
* 출력층: 해당 인물이 일치할 확률을 출력함.
  
![Alt text](image-7.png)
* 자동차 예시

---

## Neural network layer
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/675e5807-8fb2-41b2-a1d7-62ce0267706d)
* Layer0: 입력값 $\mathbf{x}=[197, 184, 136, 214]$를 세 개의 뉴런에 입력
* Layer1
   * 첫 번째 은닉층의 매개변수: $\mathbf{w}_1, b_1$
   * 출력값: $a=g(\mathbf{w}\cdot\mathbf{x}+b_1)=0.3$
   * 두 번째 은닉층의 매개변수와 출력값: 매개변수: ", 출력값: " $=0.7$
   * 세 번째 은닉층 " : 매개변수: " , 출력값: $a=g(\mathbf{w_3}\cdot z + b_3$=0.2
      * g(z)=\frac{1}{1+e^{-(z)}}
   * 최종 출력값: $\mathbf{a}^{[1]}=[0.3, 0.7, 0.2]$
      * 최종 출력값의 출처 은닉층을 알려주기 위해 출력값의 윗첨자에 $[n]$을 써준다.

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/cd8f4b41-93a6-4d92-b522-a7b455dc598f)
* Layer2
   * 입력값: $a^{[1]}$
   * 출력값의 매개변수: $\mathbf{w_1}+b_1$
   * 출력값: $\mathbf{a}_1=g(\mathbf{w}_1\cdot \mathbf{a^{[1]}}+b1)$

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/79ffc678-353f-4691-9245-d8fa6435334b)
* 출력값: $a^{[2]}=0.84$
* 결과: $a^{[2]}가 0.5보다 크거나 같나?$
   * 참: $y=1$, 거짓: $y=0$

---

## More complex neural networks

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/c0845268-4c8c-480b-951b-cb439266f6a5)
* 4층 구조 신경망: 3개의 은닉층, 1개의 출력층층
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/290c513e-3257-4cc9-8642-7da8abdfba76)
* $a_n^{[3]}=g(\mathbf{w}_1^{[3]}+\\mathbf{a}^{[2]}+b_n$
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/670ccd50-d15b-4fbc-9cff-6d3c623a2246)
* 빈칸: $\mathbf{w}_2^{[3]}$, $b_2^{[3]}=a^{[2]}=g(\mathbf{w}^{[2]} \cdot\ mathbf{a}^{[2]}+b^{[2]})$
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/3c09e955-4162-489c-a3e5-baf2d3fdbfc5)
* $a^{[l]}=g(\mathbf{w}^{[l]}\cdot\mathbf{a}^{[l-1]}+b^{[l]})$
* 입력값 $\mathbf{x}$는 $a^{[0]}$ 

---

## Inrerence: making predictions(forward propagation)
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/4a1461fc-df65-46a2-9171-54fe270f0e5a)
* 입력값: 1개, 은닉층: 2개(1단계: 25개, 2단계: 15개), 출력층: 1개($a$가 '1'일 확률)
* $a^{[1]}$은 25개의 값을 가진 벡터이다.
  
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/d06984d3-fce6-4510-97bc-01fd98007d95)
* $a^{[2]}$는 15개의 값을 가진 벡터이다. 이는 계속 활용된다.
  
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/e4989214-1bc5-493e-a38e-398d9fed5e02)
* 결과 $\mathbf{a}^{[3]}$은 $f(x)$라고 써도 된다. 값은 0~1 사이에서 나온다.
* 그동안 써 왔던 계산 방법을 **'순방향 전파'** 라고 한다. 그 반대 방향은 '역방향 전파'라고 한다.
* 보통 출력값으로 가까워질 때 뉴런의 개수는 점점 감소하는 경향이 있다.

---

## Inference in Code
### 예시: Coffee roasting
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/d72f5b99-403c-434d-89d1-86e0a49ec0ba)
* x축: 온도, y축: 시간, X: 좋은 커피, O: 나쁜 커피
* 삼각형 내: 맛이 좋음
* 온도가 낮음 || 시간이 짧음: 덜 조리됨, 온도가 높음 || 시간이 오래됨: 오래 조리됨
* 입력값 $\mathbf{x}=[200, 17]$

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/81f3fc4a-46c9-49a0-a4cd-a84d18b58106)
```python
x=np.array([[200.0, 17]] #입력값
layer_1=Dense(units=3, activation='sigmoid')#Dense함수 사용, 유닛: 3개([[0.2, 0.7, 0.3]])
a1=layer_1(x)#x: 입력값
```

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/91a693a5-9497-47e7-85dd-5093254b76ea)
```python
layer2_=Dence(units=1, activation='sigmoid')#유닛 1개([0.8])
a2=layer2(a1)#입력값: a1
```

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/61e65cbd-9f35-4b44-ab28-b2a8b2a70106)
```python
if a2>=0.5: yhat=1
else: yhat=0
```

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/c2eb63bf-82c5-4ccd-8274-3663dee56e4e)
* x: 25개, np.array형태
* layer_1에 Dense함수 사용 -> 출력값은 a1
* -> a1을 layer_2에 삽입 -> a2출력
   * layer_2에 Dense함수 사용
* ....

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/ee60a386-077a-4088-b05f-cc8677fc0546)

---

## Data in TensorFlow
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/39e678cc-d7b9-4d29-869f-0c499e76f138)

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/1f5389a1-df97-470f-9cc5-6a9365fafd51)
$$ \begin{bmatrix}
1 &2  &3  \\
4 &5  &6  \\
\end{bmatrix}$$
->
`x=np.array([[1, 2, 3], [4, 5, 6]]`

$$\begin{bmatrix}
0.1& 0.2
-3& 4
-.5& -.6
7& 8
\end{bmatrix}$$
->
`x=np.array([[0.1, 0.2], [-3, 4], [-0.5, -0.6], [7, 8]])`

![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/4b916f4d-65d6-478e-89b8-46c70e40c55c)

