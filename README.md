# ⚔️ League of Legends Contribution Model (LoL 기여도 분석 모델)

## 1. 프로젝트 개요

이 프로젝트는 **리그 오브 레전드(LoL)** 경기 데이터를 분석하여, 단순한 KDA나 최종 스코어가 아닌 **포지션별·시간대별 종합 기여도**를 객관적으로 측정하기 위해 설계된 **기여도 모델**을 구현합니다.

### 라인 정의
가이드라인에 따라 **TOP, MID, JUNGLE, BOTTOM(ADC+SUP 묶음)** 4개 라인으로 제한합니다.  
단, 분석 목적의 유효성을 위해 ‘**선수 유형 군집 분석**’은 4개 라인(TOP, JUNGLE, MID, BOTTOM) 개별로 수행합니다.

### 핵심 목표

- **중앙값(Median) 기반 측정**  
  → 같은 라인의 평균적 성능 대비 상대적인 기여도를 산출하여 포지션별 역할 차이의 불공정성을 최소화합니다.
  
- **시간대별 분리 평가**  
  → 게임 초중반의 성장 기여도와 후반의 핵심 역할 수행 기여도를 분리하여 종합적으로 평가합니다.

---

## 2. 데이터 전처리

본 분석은 v4.2 파이썬 스크립트의 `parse_all_match_data` 및 `calculate_contribution` 함수를 통해 **2단계의 전처리 과정**을 거칩니다.

### 1단계: 원시 데이터 파싱 및 피처 추출

**데이터 구성:**  
1087개의 `match_X.json` (경기 결과)과 `timeline_X.json` (분당 데이터) 파일을 파싱합니다.

**플레이어 식별:**  
`riotIdGameName + riotIdTagline` 조합 (예: `플레이어#KR1`)을 고유 ID로 사용합니다.

**피처 추출:**

- **t_ (Timeline)** 피처:  
  `totalGold`, `xp`, `damageToChampions`, `minionsKilled`, `jungleMinionsKilled`
- **f_ (Final)** 피처:  
  `killParticipation`, `visionScore`, `soloKills` 등 약 10여 개의 최종 성과 지표

**→ 산출물:** `1_minute_stats_hybrid.csv`

---

### 2단계: 상대 기여도 피처 생성

**중앙값 계산:**  
모든 분당/최종 스탯의 라인별(TOP, MID, ADC, SUP, JUNGLE) **중앙값(Median)** 을 계산합니다.

**상대(Relative) 피처 변환:**  
각 플레이어의 스탯을 해당 라인의 중앙값으로 나눠 정규화합니다.  
예: `rel_t_gold = t_totalGold / t_totalGold_median`

**→ 산출물:**
- `2_per_minute_contrib.csv` (분당)
- `2_final_contributions_simple.csv` (최종)

---

## 3. 모델 방법론: 기여도 정의 (Hybrid Contribution Score)

전처리된 상대 피처를 바탕으로 **최종 기여도(Contribution)** 점수를 산출합니다.

\[
\text{Contribution} = (\text{Timeline Score} \times 0.7) + (\text{Final Stats Score} \times 0.3)
\]

### A. Timeline Score (분당 수행 점수 – 가중치 70%)

**목표:**  
게임 초중반의 성장 속도, 자원 획득, 라인전 기여도 평가

**측정 방식:**  
`calculate_contribution` 내 `get_timeline_score`가 분당 상대 피처(`rel_t_gold` 등)에 **라인별 가중치**를 적용하여 계산.

---

### B. Final Stats Score (최종 성과 점수 – 가중치 30%)

**목표:**  
게임 종료 시점의 주요 역할 수행 능력(시야, 오브젝트, 유틸리티 등) 평가

**측정 방식:**  
`calculate_contribution` 내 `get_final_stats_score`가 최종 상대 피처(`rel_f_visionScore` 등)에 **라인별 가중치**를 적용하여 계산.

---

## 4. 사용된 피처 정의

### 4.1. Timeline Features (`t_` 접두사)

| 피처명 | 데이터 유형 | 설명 | 주요 사용 라인 |
|--------|--------------|------|----------------|
| t_totalGold | 분당 누적 골드 | 성장 및 경제력 | All |
| t_xp | 분당 누적 경험치 | 레벨 우위 확보 | All |
| t_damageToChampions | 분당 챔피언 피해량 | 전투 참여 및 딜 기여 | All |
| t_minionsKilled | 분당 라인 CS | 라인 관리 및 파밍 효율 | TOP, MID, ADC |
| t_jungleMinionsKilled | 분당 정글 몬스터 처치 수 | 정글링 효율 및 동선 | JUNGLE |

---

### 4.2. Final Stats Features (`f_` 접두사)

| 피처명 | 데이터 유형 | 설명 | 핵심 기여 역할 |
|--------|--------------|------|----------------|
| f_killParticipation | 최종 킬 관여율 | 팀 전투 기여도 | All |
| f_visionScore | 최종 시야 점수 | 시야 장악 및 정보전 | JUNGLE, SUP |
| f_soloKills | 최종 솔로킬 횟수 | 라인 압박 및 개인 기량 | TOP, MID |
| f_damageDealtToTurrets | 최종 포탑 피해량 | 스플릿 및 오브젝트 압박 | TOP, ADC |
| f_totalHealOnTeammates | 최종 아군 치유량 | 서포트 유틸리티 | SUP |
| f_timeCCingOthers | 최종 CC 시간 | 군중 제어 능력 | SUP, TANK |
| f_objectivesStolen | 오브젝트 스틸 횟수 | 변수 창출 능력 | JUNGLE |

---

## 5. 라인별 가중치 로직

### 5.1. Timeline Score 가중치

| 라인 | Rel. Gold | Rel. XP | Rel. Damage | Rel. Lane CS | Rel. Jungle CS |
|------|------------|----------|--------------|----------------|----------------|
| TOP, MID, ADC | 0.3 | 0.2 | 0.3 | 0.2 | - |
| JUNGLE | 0.3 | 0.3 | 0.1 | - | 0.3 |
| SUP | 0.4 | 0.4 | 0.2 | - | - |

---

### 5.2. Final Stats Score 가중치

| 라인 | Solo Kills | KP | Vision | Turret DMG | Heal/Shield | CC Time | Obj. Stolen |
|------|-------------|----|---------|-------------|--------------|-----------|-------------|
| TOP | 0.4 | 0.1 | 0.1 | 0.4 | - | - | - |
| JUNGLE | - | 0.4 | 0.4 | - | - | - | 0.2 |
| MID | 0.3 | 0.5 | 0.1 | 0.1 | - | - | - |
| ADC | - | 0.5 | 0.1 | 0.4 | - | - | - |
| SUP | 0.2 | 0.2 | 0.4 | - | 0.2 | 0.2 | - |

---

## 6. 실험 및 분석

### 6.1. 한 경기 단위: 시간축 기여도 곡선

특정 경기(`Match ID 367`)의 시간(분)별 4개 라인 기여도 변화를 시각화했습니다.  
**승리팀(실선)** 과 **패배팀(점선)** 을 비교하여 흐름과 승리 요인을 분석합니다.

**예시 분석:**  
패배팀(점선)의 **바텀(빨강)**은 초반 높은 기여도를 보였으나, 10분경부터 승리팀의 **미드(초록)** 와 **정글(주황)** 이 역전하며 게임을 주도했습니다.

<img width="1500" height="800" alt="117" src="https://github.com/user-attachments/assets/be183b43-e0a5-46f0-a937-5c69b3c3ac89" />



---

### 6.2. 플레이어 유형 군집 분석

‘**평균 기여도(X축)**’와 ‘**기여도 기복(Y축, 표준편차)**’을 기준으로 **K-Means 군집 분석**을 수행했습니다.  
(가이드라인에 따라 **BOTTOM은 ADC/SUP 통합**)

#### 예시 분석

- **TOP (113.png)**  
  - 우측 상단(파랑): 높은 기여도·높은 기복 → ‘캐리형/공격형’  
  - 좌측 하단(노랑): 낮은 기여도·낮은 기복 → ‘안정형/수비형’

 <br>   
<img width="1200" height="900" alt="113" src="https://github.com/user-attachments/assets/304a8628-5129-4471-aae6-4afba9b86d00" />
<br />

- **JUNGLE (114.png)**  
  - ‘기복이 심한 캐리형’과 ‘안정적인 운영형’으로 분화됨.
  <br>    
<img width="1200" height="900" alt="114" src="https://github.com/user-attachments/assets/8db9b5d7-29d9-47f5-bea7-7659a67530da" />
<br />

- **MID (115.png)**  
  - 평균 기여도 1.25 기준으로 ‘안정/수비형’(보라, 노랑)과 ‘공격/캐리형’(파랑, 초록)으로 구분됨.

    
 <img width="1200" height="900" alt="115" src="https://github.com/user-attachments/assets/b2ee0649-04f0-4e2c-938a-abacc231ab64" />


- **BOTTOM (116.png)**  
  - 기여도 1.5 초과: ‘캐리형’(초록, 노랑)  
  - 1.0~1.25: ‘안정형’(보라)

 
<img width="1200" height="900" alt="116" src="https://github.com/user-attachments/assets/64ad45ce-7aa2-4902-ad23-232edd0f64b6" />


---

## 7. 최종 아웃풋 및 해석

### 7.1. 라인별 기여도 분포 (Violin Plot)

`matchId` 기준으로 4개 라인의 평균 기여도를 집계해 **승리팀과 패배팀 간 분포를 비교** 합니다.

**해석:**  
- **승리팀(파랑)** 의 기여도 중앙값이 **1.0 이상**,  
- **패배팀(빨강)** 은 **1.0 미만**으로 나타나,  
모델의 ‘기여도’ 정의가 **승패와 강한 양의 상관관계**를 가짐을 확인했습니다.

<img width="1200" height="700" alt="111" src="https://github.com/user-attachments/assets/e3387868-54b0-474c-9fcb-44b610ff0560" />


---

### 7.2. 개인별 일관성 플롯 (Scatter Plot)

특정 소환사(예: `화내지말자#0722`)의 경기별 기여도 추이를 시각화하고, **승/패 여부**를 색상으로 표시합니다.

**해석:**  
평균 기여도 **2.22**로 매우 높으며, 패배 경기에서도 대부분 **1.0 이상**으로 일관된 퍼포먼스를 보임 → **‘에이스형 플레이어’** 로 해석.

<img width="1500" height="600" alt="112" src="https://github.com/user-attachments/assets/f912eb3c-77c8-4172-b34d-f67f4faae207" />


---

## 8. 결론

본 프로젝트는 가이드라인에 따라  
**전처리(2장) → 모델 정의(3장) → 실험/분석(6장) → 최종 아웃풋(7장)** 의 전체 과정을 체계적으로 수행했습니다.

**실험/분석:**  
시간대별 기여도 곡선과 군집 분석을 통해 경기 흐름 및 선수 성향을 도출했습니다.

**최종 아웃풋:**  
라인별 분포와 개인별 일관성 플롯을 통해 모델의 기여도 점수가 **승패와 높은 상관성을 가지며** ,  
선수 퍼포먼스를 **객관적으로 해석 가능한 지표** 로 활용 가능함을 입증했습니다.


