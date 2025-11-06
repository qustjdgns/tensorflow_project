⚔️ League of Legends Contribution Model (LoL 기여도 분석 모델)

1. 프로젝트 개요 (가이드라인 1)

이 프로젝트는 리그 오브 레전드(LoL) 경기 데이터를 분석하여, 단순한 KDA나 최종 스코어가 아닌 플레이어의 포지션별/시간대별 종합적인 기여도를 객관적으로 측정하기 위해 설계된 기여도 모델을 구현합니다.

라인 정의: 가이드라인에 따라 TOP, MID, JUNGGLE, BOTTOM (ADC+SUP 묶음) 4개 라인으로 제한합니다. (단, 분석 목적의 유효성을 위해 '선수 유형 군집 분석'은 4개 라인(TOP, JUNGLE, MID, BOTTOM) 개별로 수행합니다.)

핵심 목표

중앙값(Median) 기반 측정 (가이드라인 2)
같은 라인의 평균적 성능 대비 상대적인 기여도를 산출하여 포지션별 역할 차이의 불공정성을 최소화합니다.

시간대별 분리 평가 (가이드라인 2)
게임 초중반의 성장 기여도와 후반의 핵심 역할 수행 기여도를 분리하여 종합적으로 평가합니다.

2. 데이터 전처리 (가이드라인 2)

본 분석은 v4.2 파이썬 스크립트의 parse_all_match_data 및 calculate_contribution 함수를 통해 2단계의 전처리 과정을 거칩니다.

1단계: 원시 데이터 파싱 및 피처 추출

1087개의 match_X.json (경기 결과)과 timeline_X.json (분당 데이터) 파일을 파싱합니다.

플레이어 식별: riotIdGameName과 riotIdTagline을 조합하여 (예: 플레이어#KR1) 고유 ID로 사용합니다.

피처 추출:

t_ (Timeline) 피처: timeline.json에서 분당 totalGold, xp, damageToChampions, minionsKilled, jungleMinionsKilled를 추출합니다.

f_ (Final) 피처: match.json에서 최종 killParticipation, visionScore, soloKills 등 10여 개의 핵심 성과 지표를 추출합니다.

1_minute_stats_hybrid.csv 파일이 1차 산출물로 생성됩니다.

2단계: 상대 기여도 피처 생성

중앙값 계산: 1단계에서 추출한 모든 분당/최종 스탯의 라인별(TOP, MID, ADC, SUP, JUNGLE) 중앙값(Median)을 계산합니다.

상대(Relative) 피처 변환: 모든 플레이어의 스탯을 해당 라인의 중앙값으로 나눕니다.
(예: rel_t_gold = t_totalGold / t_totalGold_median)

이 과정을 통해 모든 스탯은 "평균(중앙값) 대비 몇 배를 수행했는가" (중앙값=1.0)라는 절대 스케일로 정규화됩니다.

이 상대 피처들은 2_per_minute_contrib.csv (분당)와 2_final_contributions_simple.csv (최종) 파일로 저장되어 3단계 모델의 입력값으로 사용됩니다.

3. 모델 방법론: 기여도 정의 (가이드라인 2)

전처리된 상대 피처를 바탕으로, 최종 기여도(Contribution) 점수를 산출합니다. 점수는 다음 두 하위 점수를 가중 평균하여 산출됩니다.

## 2. 모델 방법론 (Hybrid Contribution Score)

\text{Contribution} = (\text{Timeline Score} \times 0.7) + (\text{Final Stats Score} \times 0.3)
$$### A. Timeline Score (분당 수행 점수 – 가중치 70%)

**목표:** 게임 초반\~중반의 성장 속도, 자원 획득, 라인전 기여도를 평가합니다. (분당 데이터 전제)

**측정 방식:** `calculate_contribution` 함수 내 `get_timeline_score`가 전처리된 분당 상대 피처(`rel_t_gold` 등)에 라인별 가중치를 적용하여 계산합니다.

### B. Final Stats Score (핵심 성과 점수 – 가중치 30%)

**목표:** 게임 종료 시점의 주요 역할 수행 능력(시야, 오브젝트, 유틸리티 등)을 평가합니다.

**측정 방식:** `calculate_contribution` 함수 내 `get_final_stats_score`가 전처리된 최종 상대 피처(`rel_f_visionScore` 등)에 라인별 가중치를 적용하여 계산합니다.

-----

## 4\. 사용된 피처 정의 (가이드라인 2: 데이터 전처리)

*2단계 전처리 과정에서 추출 및 사용된 피처 목록입니다.*

### 4.1. Timeline Features (`t_` 접두사)

| 피처명 | 데이터 유형 | 설명 | 주요 사용 라인 |
|:---|:---|:---|:---|
| `t_totalGold` | 분당 누적 골드 | 성장 및 경제력 | All |
| `t_xp` | 분당 누적 경험치 | 레벨 우위 확보 | All |
| `t_damageToChampions` | 분당 챔피언 피해량 | 전투 참여 및 딜 기여 | All |
| `t_minionsKilled` | 분당 라인 CS | 라인 관리 및 파밍 효율 | TOP, MID, ADC |
| `t_jungleMinionsKilled` | 분당 정글 몬스터 처치 수 | 정글링 효율 및 동선 | JUNGLE |

-----

### 4.2. Final Stats Features (`f_` 접두사)

| 피처명 | 데이터 유형 | 설명 | 핵심 기여 역할 |
|:---|:---|:---|:---|
| `f_killParticipation` | 최종 킬 관여율 | 팀 전투 기여도 | All |
| `f_visionScore` | 최종 시야 점수 | 시야 장악 및 정보전 | JUNGLE, SUP |
| `f_soloKills` | 최종 솔로킬 횟수 | 라인 압박 및 개인 기량 | TOP, MID |
| `f_damageDealtToTurrets` | 최종 포탑 피해량 | 스플릿 및 오브젝트 압박 | TOP, ADC |
| `f_totalHealOnTeammates` | 최종 아군 치유량 | 서포트 유틸리티 | SUP |
| `f_timeCCingOthers` | 최종 CC 시간 | 군중 제어 능력 | SUP, TANK |
| `f_objectivesStolen` | 최종 오브젝트 스틸 | 정글러의 변수 창출 능력 | JUNGLE |

-----

## 5\. 라인별 가중치 상세 로직 (가이드라인 2: 팀별 설계)

*3단계 기여도 정의 모델에서 사용된 라인별 가중치입니다.*

### 5.1. Timeline Score 가중치 (Section 3A)

| 라인 | Rel. Gold | Rel. XP | Rel. Damage | Rel. Lane CS | Rel. Jungle CS |
|:---|:---:|:---:|:---:|:---:|:---:|
| TOP, MID, ADC | 0.3 | 0.2 | 0.3 | 0.2 | - |
| JUNGLE | 0.3 | 0.3 | 0.1 | - | 0.3 |
| SUP | 0.4 | 0.4 | 0.2 | - | - |

### 5.2. Final Stats Score 가중치 (Section 3B)

| 라인 | Solo Kills | KP | Vision | Turret DMG | Heal/Shield | CC Time | Obj. Stolen |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TOP | 0.4 | 0.1 | 0.1 | 0.4 | - | - | - |
| JUNGLE | - | 0.4 | 0.4 | - | - | - | 0.2 |
| MID | 0.3 | 0.5 | 0.1 | 0.1 | - | - | - |
| ADC | - | 0.5 | 0.1 | 0.4 | - | - | - |
| SUP | 0.2 | 0.2 | 0.4 | - | 0.2 | 0.2 | - |

-----

## 6\. 실험 및 분석 (가이드라인 2, 4)

최종 아웃풋을 해석하기 위한 상세 분석 과정입니다.

### 6.1. 한 경기 단위: 시간축 기여도 곡선 (가이드라인 2)

특정 경기(예: Match ID 367)를 선정하여 시간(분)에 따른 4개 라인(BOTTOM 통합)의 기여도 변화를 시각화했습니다. 승리팀(실선)과 패배팀(점선)의 기여도 곡선을 비교하여 게임의 흐름과 핵심 승리 요인을 분석합니다.

* **분석 (예시):** 이 경기(105.png)에서는 패배팀(점선)의 바텀(빨강)이 초반부터 압도적인 기여도를 보였으나, 10분경부터 승리팀(실선)의 미드(초록)와 정글(주황)이 역전하며 게임을 주도한 것을 볼 수 있습니다.

*(105.png - 5-4. Match ID 367 시간대별 기여도)*
![Figure\_5\_4](https://github.com/user-attachments/assets/db3f972b-2321-4f1b-85d1-7f897f2cd6f8)

### 6.2. 플레이어 유형 군집 분석 (가이드라인 2, 4)

선수들의 성향을 파악하기 위해 '평균 기여도'(X축)와 '기여도 기복'(Y축, 표준편차)을 기준으로 K-Means 군집 분석을 수행했습니다. (가이드라인 1번에 따라 `BOTTOM`은 ADC/SUP 통합)

* **분석 (예시):**
* **TOP (103.png):** 4개 유형으로 군집화되었습니다. 우측 상단(파랑)은 "높은 기여도, 높은 기복"을 가진 '캐리형/공격형' 선수 집단으로, 좌측 하단(노랑)은 "낮은 기여도, 낮은 기복"을 가진 '안정형/수비형' 선수 집단으로 해석할 수 있습니다.
* **JUNGLE (104.png):** 정글 또한 유사한 패턴을 보이며, 플레이어의 성향이 '기복이 심한 캐리형'과 '안정적인 운영형'으로 나뉘는 것을 확인할 수 있습니다.

*(103.png - 5-3. TOP 라인 군집 분석)*
![Figure\_5\_3\_TOP](https://github.com/user-attachments/assets/f0d85a11-e63d-4c3d-b43e-a6125139031c)

*(104.png - 5-3. JUNGLE 라인 군집 분석)*
![Figure\_5\_3\_JUNGLE](https://github.com/user-attachments/assets/401c450a-f0e7-4f6c-829d-47214a1a511e)

-----

## 7\. 최종 아웃풋 및 해석 (가이드라인 3)

### 7.1. 라인별 기여도 분포 (Violin Plot)

가이드라인 "3. 아웃풋 (고정)" 항목입니다. `matchId` 기준으로 4개 라인(BOTTOM 통합)의 평균 기여도를 집계하여, 승리팀과 패배팀 간의 라인별 기여도 분포 차이를 비교합니다.

* **해석:** (101.png) 모든 라인에서 승리팀(파랑)의 기여도 중앙값이 1.0을 상회하고, 패배팀(빨강)의 중앙값이 1.0 미만에 머무르는 것을 통해, 본 모델이 정의한 '기여도'가 승패와 강한 양의 상관관계가 있음을 알 수 있습니다.

*(101.png - 5-1. 라인별 종합 기여도 분포)*
&lt;img width=&quot;1200&quot; height=&quot;700&quot; alt=&quot;Figure_5_1&quot; src=&quot;https://github.com/user-attachments/assets/caec11ce-c3e4-469e-acae-85ccf2cf0410&quot; /&gt;

### 7.2. 개인별 일관성 플롯 (Scatter Plot)

가이드라인 "3. 아웃풋 (고정)" 항목입니다. 특정 소환사(예: '화내지말자\#0722')의 경기별 기여도 변화 추이를 시각화합니다. 승/패 여부를 색상으로 표시하여 기여도와 승리의 상관관계를 분석하고 플레이어의 기복(일관성)을 파악합니다.

* **해석:** (102.png) 이 소환사는 평균 2.22라는 압도적인 기여도를 보이며, 패배(빨강)한 경기조차 1.0 이상의 기여도를 기록하는 등(1경기 제외) 매우 높은 수준의 일관성을 보여주는 '에이스' 플레이어임을 알 수 있습니다.

*(102.png - 5-2. '화내지말자\#0722' 소환사 일관성)*
![Figure\_5\_2](https://github.com/user-attachments/assets/d3780517-1f48-43d0-8f68-7517c4fa4895)

-----

## 8\. 결론

본 프로젝트는 가이드라인에 따라 원시 JSON 데이터의 \*\*전처리(2장)\*\*부터 **모델 정의(3장)**, **실험/분석(6장)**, \*\*최종 아웃풋(7장)\*\*까지 체계적으로 수행했습니다.

* **(실험/분석)** 시간대별 곡선(6.1)과 군집 분석(6.2)을 통해 게임의 동적인 흐름과 선수 개개인의 성향을 분석할 수 있었습니다.
* **(최종 아웃풋)** 라인별 분포(7.1)와 개인별 일관성(7.2) 플롯을 통해 모델의 기여도 점수가 승패와 강한 연관성을 가지며, 선수의 퍼포먼스를 객관적으로 해석하는 지표로 활용 가능함을 입증했습니다.
