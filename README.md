# ⚔️ League of Legends Contribution Model (LoL 기여도 분석 모델)



---

## 1. 프로젝트 개요

이 프로젝트는 리그 오브 레전드(LoL) 경기 데이터를 분석하여, 단순한 KDA나 최종 스코어가 아닌 포지션별·시간대별 종합 기여도를 객관적으로 측정하기 위해 설계된 기여도 모델을 구현합니다.

### 라인 정의

- 가이드라인에 따라 TOP, MID, JUNGLE, BOTTOM(ADC+SUP 묶음) 4개 라인으로 제한합니다.
- 단, 분석 목적의 유효성을 위해 ‘선수 유형 군집 분석’은 4개 라인(TOP, JUNGLE, MID, BOTTOM) 개별로 수행합니다.

### 핵심 목표

- **중앙값(Median) 기반 측정**  
  → 같은 라인의 평균적 성능 대비 상대적인 기여도를 산출하여 포지션별 역할 차이의 불공정성을 최소화합니다.

- **시간대별 평가**  
  → 게임 초중반의 성장 기여도를 중심으로 종합적으로 평가합니다.

---

## 2. 데이터 전처리

본 분석은 **v4.9 파이썬 스크립트 (`lol_analysis_final_v4.9_1min_data.py`)**의 `parse_all_match_data` 및 `calculate_contribution` 함수를 통해 2단계의 전처리 과정을 거칩니다.

### 1단계: 원시 데이터 파싱 및 피처 추출

- **데이터 구성:** 1087개의 `match_X.json` (경기 결과)과 `timeline_X.json` (분당 데이터) 파일을 파싱합니다.
- **플레이어 식별:** `riotIdGameName + riotIdTagline` 조합 (예: `플레이어#KR1`) 을 고유 ID로 사용합니다.
- **피처 추출:**
  - `t_` (Timeline) 피처: `totalGold`, `xp`, `damageToChampions`, `minionsKilled`, `jungleMinionsKilled`
  - `f_` (Final) 피처: `killParticipation`, `visionScore`, `soloKills` 등 약 10여 개의 최종 성과 지표
- **[v4.9] 데이터 포함 범위:** `minute == 0`을 제외한 1분부터의 모든 데이터를 분석에 포함합니다.

**→ 1차 산출물:** `1_minute_stats_hybrid_v4.9.csv`

### 2단계: 상대 기여도 피처 생성

- **중앙값 계산:** 모든 분당/최종 스탯의 라인별(TOP, MID, ADC, SUP, JUNGLE) 중앙값(Median) 을 계산합니다.
- **상대(Relative) 피처 변환:** 각 플레이어의 스탯을 해당 라인의 중앙값으로 나눠 정규화합니다.  
  (예: `rel_t_gold = t_totalGold / t_totalGold_median`)

**→ 2차 산출물 (v4.9):**

- `2_final_contributions_timeline_only_v4.9.csv` (5-1, 5-2, 5-3, 레이더 차트용)
- `2_per_minute_contrib_hybrid_v4.9.csv` (5-4 시간대별 곡선 전용)

---

## 3. 모델 방법론: 기여도 정의 

본 모델은 분석 목적에 따라 2가지 점수 체계를 분리하여 사용합니다.

### A. Timeline Score (기본 기여도)

- **목표:** 게임 초중반의 성장 속도, 자원 획득, 라인전 기여도 평가
- **측정 방식:** `get_timeline_score`가 분당 상대 피처(`rel_t_gold` 등)에 **라인별 가중치(표 5.1)**를 적용하여 계산.
- **[핵심] 적용 범위:**  
  5-1(분포), 5-2(개인), 5-3(군집), 선택(레이더) 분석에서는, 이 Timeline Score를 경기별로 평균 내어 **100% '최종 기여도'**로 사용합니다.

### B. Hybrid Score (5-4 시간대별 곡선 전용)

- **목표:** 게임의 분당 흐름(Timeline)과 그 경기의 최종 성과(Final Stats)를 동시에 반영.
- **[핵심] 측정 방식:**  
  오직 5-4(시간대별 곡선) 분석에서만, 분당 흐름과 최종 성과를 모두 반영하기 위해 아래의 하이브리드 공식을 예외적으로 사용합니다.

Contribution (5-4) = (Timeline Score * 0.7) + (Final Stats Score * 0.3)

- `Final Stats Score`는 `get_final_stats_score`가 최종 상대 피처(`rel_f_visionScore` 등)에 라인별 가중치를 적용하여 계산합니다.

---

## 4. 사용된 피처 정의

### 4.1. Timeline Features (t_ 접두사)

| 피처명 | 데이터 유형 | 설명 | 주요 사용 라인 |
|--------|------------|------|----------------|
| t_totalGold | 분당 누적 | 성장 및 경제력 | All |
| t_xp | 분당 누적 | 레벨 우위 확보 | All |
| t_damageToChampions | 분당 누적 | 전투 참여 및 딜 기여 | All |
| t_minionsKilled | 분당 누적 | 라인 관리 및 파밍 효율 | TOP, MID, ADC |
| t_jungleMinionsKilled | 분당 누적 | 정글링 효율 및 동선 | JUNGLE |

### 4.2. Final Stats Features (f_ 접두사)

| 피처명 | 데이터 유형 | 설명 | 핵심 기여 역할 |
|--------|------------|------|----------------|
| f_killParticipation | 최종 | 팀 전투 기여도 | All |
| f_visionScore | 최종 | 시야 장악 및 정보전 | JUNGLE, SUP |
| f_soloKills | 최종 | 라인 압박 및 개인 기량 | TOP, MID |
| f_damageDealtToTurrets | 최종 | 스플릿 및 오브젝트 압박 | TOP, ADC |
| f_totalHealOnTeammates | 최종 | 서포트 유틸리티 | SUP |
| f_timeCCingOthers | 최종 | 군중 제어 능력 | SUP, TANK |
| f_objectivesStolen | 최종 | 변수 창출 능력 | JUNGLE |

---

## 5. 라인별 가중치 로직 및 근거

### 5.1. Timeline Score 가중치

| 라인 | Rel. Gold | Rel. XP | Rel. Damage | Rel. Lane CS | Rel. Jungle CS |
|------|------------|---------|-------------|--------------|----------------|
| TOP, MID, ADC | 0.3 | 0.2 | 0.3 | 0.2 | - |
| JUNGLE | 0.3 | 0.3 | 0.1 | - | 0.3 |
| SUP | 0.4 | 0.4 | 0.2 | - | - |

### 5.2. Timeline 피처 간 상관성 검증 (가중치 근거)

Timeline 피처들을 하나의 '성장력' 점수로 묶는 것이 타당한지 검증하기 위해, '승패' 라벨을 사용하지 않고 **피처 간의 내적 일관성(평균 상관계수)**을 분석했습니다.

**분석 로그 요약 (마크다운 표):**

| 라인   | Timeline 근거 | 피처 내적 평균 상관계수 |
|--------|---------------|------------------------|
| TOP    | Timeline      | 0.465                  |
| JUNGLE | Timeline      | 0.431                  |
| MID    | Timeline      | 0.391                  |
| ADC    | Timeline      | 0.386                  |
| SUP    | Timeline      | 0.385                  |

### 5.3. Timeline 피처 간 상관성 해석

| 라인 | 평균 상관계수 | 해석 |
|------|---------------|------|
| TOP | 0.465 | 성장 및 전투 관련 피처 간 균형적 상관 구조 |
| JUNGLE | 0.431 | 경제, 경험, 정글CS 간 상호보완적 관계 |
| MID | 0.391 | 전투 및 성장 지표 간 유의미한 상관성 |
| ADC | 0.386 | 피해 중심 지표 간 중간 수준의 상관 구조 |
| SUP | 0.385 | 지원형 변수 간 적정 상관성 확보 |

> 피처 간 상관계수(0.38~0.46)는 과도한 중복 없이(0.9 아님) 상호 연관된(0.1 아님) 특성을 보여줍니다.  
> 이는 '골드', 'XP' 등이 "성장력"이라는 하나의 개념을 구성하는 관련 있으면서도 고유한 요소들임을 증명하며, 이 피처들을 가중 합산하여 'Timeline Score'로 정의한 방식의 정당성을 뒷받침합니다.

---

## 6. 실험 및 분석

### 6.1. 한 경기 단위: 시간축 기여도 곡선 (Match ID 367)

본 분석은 **하이브리드 점수(T0.7 + F0.3)**를 사용하여, 특정 경기(Match ID 367)의 시간(분)별 4개 라인 기여도 변화를 시각화했습니다.

<img width="1500" height="800" alt="시간대별 기여도" src="https://github.com/user-attachments/assets/f69af0b6-b926-4fe2-9faf-3c5b3e43eeab" />

- 승리팀(실선)과 패배팀(점선)을 비교하여 흐름과 승리 요인을 분석합니다.
- 예시 분석:  
  패배팀(점선)의 **바텀(빨강)**은 초반 높은 기여도를 보였으나, 10분경부터 승리팀의 미드(초록)와 정글(주황)이 역전하며 게임을 주도했습니다.

### 6.2. 플레이어 유형 군집 분석

본 분석은 Timeline-Only 점수를 사용하여, ‘평균 기여도(X축)’와 ‘기여도 기복(Y축, 표준편차)’을 기준으로 K-Means 군집 분석을 수행했습니다. (가이드라인에 따라 BOTTOM은 ADC/SUP 통합)

- 예시 분석 (TOP)  
  - 우측 상단(파랑): 높은 기여도·높은 기복 → ‘캐리형/공격형’  
  - 좌측 하단(노랑): 낮은 기여도·낮은 기복 → ‘안정형/수비형’
<img width="1200" height="900" alt="top 라인" src="https://github.com/user-attachments/assets/b612f95b-9a8b-47d5-a4cc-aca6c404811a" />

- 예시 분석 (JUNGLE)  
  - ‘기복이 심한 캐리형’과 ‘안정적인 운영형’으로 분화됨.
<img width="1200" height="900" alt="정글" src="https://github.com/user-attachments/assets/8b37622e-4459-453a-b2e6-6404222ab765" />

- 예시 분석 (MID)  
  - 평균 기여도 1.25 기준으로 ‘안정/수비형’(보라, 노랑)과 ‘공격/캐리형’(파랑, 초록)으로 구분됨.
<img width="1200" height="900" alt="미드" src="https://github.com/user-attachments/assets/718ebf14-1291-4167-b6ec-74356f6ff1cc" />

- 예시 분석 (BOTTOM)  
  - 기여도 1.5 초과: ‘캐리형’(초록, 노랑)  
  - 1.0~1.25: ‘안정형’(보라)
<img width="1200" height="900" alt="바텀" src="https://github.com/user-attachments/assets/3deb8025-8b5c-4a60-9bb7-8ae35794bf2a" />

---

## 7. 최종 아웃풋 및 해석

### 7.1. 라인별 기여도 분포 (Violin Plot)

본 분석은 Timeline-Only 점수를 사용하여, matchId 기준으로 4개 라인의 평균 기여도를 집계해 승리팀과 패배팀 간 분포를 비교합니다.
<img width="1200" height="700" alt="라인별 종합 기여도" src="https://github.com/user-attachments/assets/2a468f91-5265-4645-a922-4eb5bab6c7b1" />

- 해석:  
  - 승리팀(파랑)의 기여도 중앙값이 1.0 이상  
  - 패배팀(빨강)은 1.0 미만  
  → 모델의 ‘기여도’ 정의가 승패와 강한 양의 상관관계를 가짐을 확인했습니다.

### 7.2. 개인별 일관성 플롯 (Scatter Plot)

 본 분석은 Timeline-Only 점수를 사용하여, `Bonnie#0314` 소환사의 경기별 기여도 추이를 시각화하고, 승/패 여부를 색상으로 표시합니다.
<img width="1500" height="600" alt="소환사 종합 기여도" src="https://github.com/user-attachments/assets/c7d9ebda-9982-4c05-82e3-253b16dd6fd0" />

- 해석:  
  평균 기여도 2.22로 매우 높으며, 패배 경기에서도 대부분 1.0 이상으로 일관된 퍼포먼스를 보임 → ‘에이스형 플레이어’로 해석.

---

## 8. 선택 아웃풋 (Optional Outputs)

- 목적: Timeline-Only 점수를 사용하여, `Bonnie#0314` 소환사의 라인별 평균 기여도를 전체 평균과 비교하여, 포지션별 강점과 약점을 시각적으로 파악
- 시각화: 레이더(스파이더) 차트
  
  <img width="600" height="600" alt="봄니vs 전체평균" src="https://github.com/user-attachments/assets/4a43cb46-35ee-4e00-9815-f40b0b695140" />

- 해석:  
  - BOTTOM 포지션의 기여도가 전체 평균보다 높음  
  - JUNGLE, MID, TOP은 평균 이하로, 특정 라인 중심의 플레이 스타일  
  → 전반적으로 포지션 특화형 플레이어로 분류됨

---

## 9. 결론

본 프로젝트는 리그 오브 레전드(LoL) 경기 데이터를 기반으로, 포지션별·시간대별 종합 기여도를 정량적으로 평가하기 위한 분석 모델을 구축한 연구입니다.

- 본 모델은 분석 목적에 따라 **Timeline-Only (일반 분석용)**와 Hybrid (시간대별 분석용) 점수를 분리하여 적용했습니다.
- 중앙값 기반의 상대 기여도 계산을 통해 포지션별 역할 차이를 보정하고, 'Timeline Score'의 가중치 근거로 '피처 내적 일관성'을 검증하여 0.4 내외의 합당한 상관관계를 확인했습니다. 
- 승패에 따른 기여도 분포 비교 결과, 모델의 기여도 점수가 승패와 높은 상관관계를 보였습니다.
- 라인별 군집 분석을 통해 ‘공격/캐리형’, ‘운영형’ 등 플레이스타일 유형을 식별하였고, `Bonnie#0314`의 개인별 기여도 추이 분석을 통해 플레이어의 일관성을 확인했습니다.
- 결과적으로 본 모델은 단순 KDA 지표를 넘어, 시간대별 성장 및 최종 성과를 종합적으로 반영하는 객관적 성과 평가 체계로 기능할 수 있음을 입증했습니다.


