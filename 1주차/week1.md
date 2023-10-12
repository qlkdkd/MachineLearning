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
* **회귀**는 무한히 가능한 출력들을 예측함
