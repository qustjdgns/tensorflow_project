# League of Legends 기여도 분석 모델 (PCA Statistical Model)
---

# 1. 프로젝트 개요
---
주제: 라인별 기여도 정의 및 경기 데이터 해석

목표: 단순 승패나 KDA 를 넘어선, 객관적이고 통계적인 '개인 라인전 기량(Contribution)' 평가 모델 구축 

데이터: 챌린저 티어 매치 데이터 (Timeline 포함), Early/Mid Game (8~20분) 데이터 집중 분석  

방법론: PCA (주성분 분석) 를 활용한 비지도 학습 가중치 산출 (Label-Free Modeling)

데이터 스냅샷 : 8분, 10분, 12분, 15분, 20분

# 2. 데이터 전처리 및 피처 선정 (Feature Engineering)
---

## 2-1. 왜 '상대 격차(Diff)' 인가?
---


<img width="3999" height="2232" alt="image" src="https://github.com/user-attachments/assets/3a7650b2-342a-41cc-b8d9-eaa39bf5df66" />



> 설명: 내 지표가 높아도 상대방이 더 높으면 의미가 없다.  또한 기여도는 상대방과의 벡터 차이(Difference) 로 정의된다.

리그 오브 레전드는 상대평가 게임이다.  
내가 CS 를 잘 획득했어도 상대가 더 많이 획득했다면 불리한 상황이다.

따라서 본 모델은 절대 지표 대신 아래의 Diff Features 를 핵심 입력값으로 사용하여 **"맞라인 상대보다 얼마나 우월한가?"** 를 측정한다.

### Diff Features 표
---
| Feature | 정의 |
|--------|------|
| Diff Gold | (내 골드 - 상대 골드) / 시간(분) |
| Diff XP | (내 경험치 - 상대 경험치) / 시간(분) |
| Diff CS | (내 CS - 상대 CS) / 시간(분) |
| Rel Damage | 해당 시간대/라인의 중앙값(Median) 대비 딜량 비율 |

---
## 2-2. 피처 선정의 타당성 (대리 변수 효과)
---
본 모델은 소수의 핵심 변수(Gold, XP) 에 집중한다.  
그 이유는 '대리 변수(Proxy Variable)' 효과 때문이다.

Gold & XP 는 솔로 킬, 로밍 성공, 포탑 채굴, 디나이(Deny) 등 모든 유리한 플레이의 최종 결과값이다.

따라서 수많은 세부 지표(와드, CC기 등) 를 넣지 않아도,  Gold/XP/Damage 변수만으로 퍼포먼스를 설명하기에 충분하며,  
이는 PCA 분석 시 해당 피처들의 가중치가 높게 산출된 것으로 입증된다.

---
# 3. 기여도 모델링 방법론: PCA (Principal Component Analysis)
---
"왜 머신러닝(회귀분석) 을 쓰지 않았는가?"

승패(Win) 라벨을 학습에 사용하면,  **'패배했지만 잘한 유저(SVP)'** 를 제대로 평가할 수 없기 때문이다.

따라서 본 프로젝트는 승패 정보를 모르는 상태에서 **데이터의 분산만을 이용해 가중치를 찾는 비지도 학습(PCA)** 을 적용했다.

---
## 3-1. 모델 아키텍처 (Input - Model - Output)
---

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/080f7b80-8da2-4329-be57-a1c60dcea94c" />



1. Input ($\vec{x}$):  
4차원 벡터 [Diff Gold, Diff XP, Diff CS, Rel Damage] (Standard Scaling 적용)

2. Model (PCA):  
각 라인별 데이터의 공분산을 분석하여, 정보를 가장 많이 담고 있는 제1 주성분(PC1) 축을 찾았다.  
이 과정에서 각 라인별 최적 가중치($\vec{w}$) 가 데이터 기반으로 자동 산출된다.

3. Output (Contribution):

$$
\text{Contribution} = \text{Normalized(PC1)} - (\text{Deaths} \times \text{Penalty})
$$

도출된 점수를 중앙값 1.0 기준으로 정규화하고,  안정성 지표인 데스 페널티를 적용한다.

---
## 3-2. PCA 가중치 해석 및 도메인 지식 검증
---

<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/a63fecc4-7a07-4ac7-998a-64eb42b67136" />


> x축: 피처 / y축: 가중치 / 색상: 라인별 구분

결과 분석:

UNGLE (주황색): Gold, CS보다 XP Diff(경험치) 가중치가 압도적으로 높게 산출되었음. 
이는 정글러의 레벨링이 갱킹 성공률 및 운영 주도권에 가장 중요하다는 도메인 지식과 일치함.

ADC/MID (빨강/초록): Gold Diff와 CS Diff가 가장 높은 비중을 차지함.
이는 코어 아이템을 빠르게 확보하는 것이 딜러진의 핵심임을 증명함.

SUP (보라색): 상대적으로 Gold/XP 가중치가 다른 라인보다 낮게 형성됨.

의의: PCA는 수동으로 가중치를 설정하지 않았음에도 불구하고, 각 라인의 역할 특성에 따라 가장 합리적인 가중치를 통계적으로 도출했음.

---
# 4. 분석 결과 및 시각화 (Analysis Result)
---
## 4-1. 라인별 기여도 분포 (Violin Plot)
---

  <img width="1200" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/46df4986-e7c2-42ac-9c9c-70348cfee411" />
  

> 사용한 스냅샷 : 8분, 10분, 12분, 15분, 20분
> /x축: 포지션 / y축: 기여도 점수 / 파랑: 승리팀, 빨강: 패배팀  

결과: 승리 팀은 기여도 중앙값이 1.2 이상, 패배 팀은 0.8 이하에 분포한다. 

해석: 승패 라벨을 학습하지 않았음에도 분포가 명확히 분리되는 것은, 모델이 정의한 '기여도' 가 실제 승리에 결정적임을 시사한다.

---
## 4-2. 개인별 기여도 일관성 (Consistency Plot)
---

<img width="1200" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/aeb0be57-8698-42b0-8a8d-f020efefc7c3" />

> x축: 매치 번호 / y축: 기여도 / 점선: 기준선(1.0), 초록선: 내 평균  

분석: 분석 대상(Bonnie#0314)의 평균 기여도는 0.62로 측정되었다.

해석: 기준점(1.0) 대비 다소 낮은 수치를 기록하고 있으며, 특히 패배한 경기(Red dots)에서 기여도가 크게 낮아지는 경향이 있다. 이는 라인전 단계에서 안정성을 보완해야 함을 시사한다.

---
## 4-3. 소환사 유형 군집화 (Clustering)
---

 <img width="1400" height="1027" alt="Figure_4" src="https://github.com/user-attachments/assets/077a82e8-40f3-47db-a16a-a3db01e6685b" />
 
> x축: 평균 기여도(Performance) / y축: 표준편차(Risk)

분석: 4개의 라인(TOP, JUNGLE, MID, BOTTOM) 모두에서 'High Mean, Low Std (안정형)' 군집과 'High Mean, High Std (캐리형)' 군집이 명확히 구분된다.

결과: 본 모델을 통해 단순 승률이 아닌, 플레이어의 성향(공격적/안정적)을 유형화할 수 있음을 확인했다.

---
## 4-4. 경기 시간대별 기여도 변화 (Match Timeline)
---

<img width="1200" height="600" alt="Figure_5" src="https://github.com/user-attachments/assets/f59f4e7b-de82-4f11-a814-1c44f21087c5" />

> x축: 경기 시간(분) / y축: 누적 기여도 / 실선: 승리팀, 점선: 패배팀  

분석: 초반 8분 라인전 단계에서 발생한 미세한 격차(Diff) 가 15분, 20분으로 갈수록 스노우볼링되어 기여도 격차가 벌어지는 현상이 관측된다.

의의: 8~20분의 스냅샷 데이터가 게임의 승패 흐름을 효과적으로 설명함을 입증한다.

---
## 4-5. 모델 검증 (ROC Curve & Confusion Matrix)
---

 <img width="1200" height="500" alt="Figure_6" src="https://github.com/user-attachments/assets/0644d1b9-3ae3-44e7-ac11-d65860d9c9fd" />

> AUC Score: 0.721  

결과 해석: 기여도 모델 학습 시 승패 정보를 사용하지 않았음에도, 

산출된 기여도 점수만으로 승패를 **약 72%** 의 확률로 정확히 분류해냈다. 

이는 모델의 통계적 유의성을 강력하게 뒷받침한다.

---
## 4-6. 최상위 아마추어 vs 프로게이머 (Radar Chart)
---

<img width="600" height="600" alt="Figure_7" src="https://github.com/user-attachments/assets/ade34b94-efa2-4195-aea1-0b37a5cf98f4" />

> 파란색: 분석 대상(Me) / 빨간색: Top Performer  

비교 분석: 일반 유저(Bonnie#0314)와 데이터셋 내 Top Performer를 비교한 결과, 

전체적인 육각형의 크기에서 차이가 발생한다. 특히 안정성 및 성장 지표 전반에서 Top Performer가 압도적인 우위를 점하고 있어,

이를 벤치마킹한 플레이 개선이 필요하다.

---
# 5. 결론 
---
본 프로젝트는 데이터 기반(Data-Driven) 의 PCA 방법론을 통해 LoL 의 기여도를 정의했다.  

이 모델은 **"승패 결과와 무관하게, 라인전 단계에서 통계적으로 얼마나 우월했는가"** 를  객관적으로 수치화함으로써, 
단순 KDA 를 넘어선 심층적인 피드백을 제공할 수 있었다.
