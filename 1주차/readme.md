[퀴즈 보기](https://github.com/qlkdkd/MachineLearning/blob/main/1%EC%A3%BC%EC%B0%A8/Quiz.md)
# What is Machine Learning?
## Machine Learning
"명시적으로 프로그래밍하지 않고도 컴퓨터에 학습할 수 있는 능력을 부여하는 연구 분야"
-작가 사무엘(1959)

## Question
만약 체커 프로그램이 오직 10번의 게임(열번 대신 천번)을 스스로 한다면, 훨신 적은 게임에게, 얼마나 이것의 성능을 향상시켰는가?
* 좀 더 잘 만들었더라면 좋았을 텐데
* 상황을 더 악화시켰을 것입니다

## Machine learning algorithms
* 지도학습
* 비지도학습
* 추천 시스템
* 강화 학습
> 학습 알고리즘 적용을 위한 실용적인 조언

---

# Supervised Learning Part1
## 지도학습
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/f1199807-1c64-4ffb-8dd5-09bb02da7483)
"정답"을 통해 배운다."

입력(X)| 출력(Y)| 어플리케이션
---|---|---
이메일|스팸(0/1)|스팸 필터링
오디오|text transcript|음성 인식
영어|스페인어|기계 번역
광고, 사용자 정보|클릭 여부(0/1)|온라인 광고
이미지, 라디오 정보|다른 차의 위치|자율주행차
핸드폰 이미지|결함 여부{0/1)|비주얼 검사

## Regression: Housing price prediction
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/cfa058a6-93d7-4af8-bdb4-d149874e7ff4)
* x: 집의 크기, y: 가격
* 데이터셋: 빨간 X표시 -> (750 feet^2, 150000$) 집의 가격을 예측 가능
* **회귀**는 연속적인 값을 예측한다.
---

# Supervised Learning Part2
## Classifiaction: Breast cancer detection(분류: 유방암 예측)
### 분류: 미리 정의되었고, 가능성이 있는 클래스 레이블(class label)중 하나를 예측하는 것이다. 즉, 이산된 값을 추정한다.
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/d23d59e3-3da0-480a-b19a-724103ef650b)
* 빨간 점: 유방암 존재, 파란 점: 정상
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/adbb3429-f081-4a2e-aec9-0ba00bfcb2d6)
#### 다중분류
* 빨간 x: 종양 타입1
* 주황 세모: 종양 타입2
* 파랑 원: 정상

### 다른 조건도 추가 가능
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/9701a583-ac7f-400d-9fa5-e4ae94aa27c0)
* 추가된 조건
    * 나이
    * 종양 두께
    * 세포 사이즈의 균일성
    * 세포 모양의 균일성

## 주어진 "올바른 값"에 대해 배우기
* 회귀: 연속적인 값 예측
* 분류: 이산적인 값 예측

# Unsupervised Learning Part1
![image](https://github.com/qlkdkd/MachineLearning/assets/71871927/1c8c6cf6-698c-456a-b353-6dcb9e721550)
* 비지도학습은 알고 있는 출력 값(label)이나 제공하는 데이터 셋 없이 학습하는 머신 러닝을 의미한다. 오직 입력 데이터만으로 데이터에서 지식을 추출할 수 있어야 한다. 따라서, Pre diction result에 대한 피드백이 없으며 잘못된 지식을 추측하더라도 교정해줄 '선생님'이 없다.
* 비지도 학습은 가지고 있는 데이터 변수들 간의 관계를 가지고 군집화(clustering)하여 어떠한 구조인지 추측한다.
* 예시: 구글 뉴스, DNA 배열 등

# Unsupervised Learning Part2
* 군집화: 그룹의 비슷한 데이터에 서로 지목한다.
* 이상 감지(Anomaly detection): 비슷하지 않은 데이터에 지목한다.
* 차원 축소 감소(Dimenstionality reduction): 더 적은 수를 이용하여 데이터를 압축한다.

## 질문: Of the following examples, which woult you address using an unsupervised learning algorithm?
* Given email labeled as spam/not spam, learn a spam filter(X)->지도학습
* Given a set of news articles found on the web, group them into sets of articles about the same story(O)->군집화
* Given a database of customer data, automatically discover market segments and group customers into different market segments.(O)-> 군집화
* Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not(X)-> 지도학습
