⚔️ League of Legends Hybrid Contribution Model (LoL 하이브리드 기여도 분석 모델)

1. 프로젝트 개요

이 프로젝트는 리그 오브 레전드(LoL) 경기 데이터를 분석하여, 단순한 KDA나 최종 스코어가 아닌 플레이어의 포지션별/시간대별 종합적인 기여도를 객관적으로 측정하기 위해 설계된 '하이브리드 기여도 모델'을 구현합니다.

핵심 목표:

중앙값(Median) 기반 측정: 같은 라인의 평균적인 성능 대비 상대적인 기여도를 산출하여, 포지션별 역할 차이에서 오는 불공정성을 최소화합니다.

시간대별 분리 평가: 게임 초중반의 성장 기여도와 후반 핵심 역할 수행 기여도를 분리하여 종합적으로 평가합니다.

2. 모델 방법론 (Hybrid Contribution Score)

최종 기여도(Contribution) 점수는 다음 두 가지 주요 점수를 가중 평균하여 산출됩니다. 기본 중앙값 대비 성능이 좋으면 1.0보다 높은 점수를, 낮으면 1.0보다 낮은 점수를 받게 됩니다.

$$\text{Contribution} = (\text{Timeline Score} \times 0.7) + (\text{Final Stats Score} \times 0.3)$$

A. Timeline Score (분당 수행 점수 - 가중치 70%)

목표: 게임 초반부터 중반까지의 성장 속도, 자원 획득, 라인전 기여도를 평가합니다.

측정 방식: 각 분당 스탯(Gold, XP, Damage, CS)을 해당 라인의 분당 중앙값으로 나누어 상대적 성능을 측정합니다.

B. Final Stats Score (핵심 성과 점수 - 가중치 30%)

목표: 게임 종료 시점의 **주요 역할 수행 능력 (시야 장악, 오브젝트 압박, 유틸리티 등)**을 평가합니다.

측정 방식: 최종 스탯(KP, Vision Score, Turret Damage 등)을 해당 라인의 전체 중앙값으로 나누어 상대적 성능을 측정합니다.

3. 사용된 피처 정의 (Feature Definitions)

3.1. Timeline Features (t_ 접두사)

피처명

데이터 유형

설명

주요 사용 라인

t_totalGold

분당 누적 골드

성장 및 경제력

All

t_xp

분당 누적 경험치

레벨 우위 확보

All

t_damageToChampions

분당 챔피언 피해량

전투 참여 및 딜 기여

All

t_minionsKilled

분당 라인 CS

라인 관리 및 파밍 효율

TOP, MID, ADC

t_jungleMinionsKilled

분당 정글 몬스터 처치 수

정글링 효율 및 동선

JUNGLE

3.2. Final Stats Features (f_ 접두사)

피처명

데이터 유형

설명

핵심 기여 역할

f_killParticipation

최종 킬 관여율

팀 전투 기여도

All

f_visionScore

최종 시야 점수

시야 장악 및 정보전

JUNGLE, SUP

f_soloKills

최종 솔로킬 횟수

라인 압박 및 개인 기량

TOP, MID

f_damageDealtToTurrets

최종 포탑 피해량

스플릿 및 오브젝트 압박

TOP, ADC

f_totalHealOnTeammates

최종 아군 치유량

서포트 유틸리티

SUP

f_timeCCingOthers

최종 CC 시간

군중 제어 능력

SUP, TANK

f_objectivesStolen

최종 오브젝트 스틸

정글러의 변수 창출 능력

JUNGLE

4. 라인별 가중치 상세 로직

각 라인(포지션)의 특성에 맞춰 기여도를 계산하기 위해, 각 점수 산출 시 핵심 지표에 높은 가중치를 부여합니다.

4.1. Timeline Score 가중치 (Section 2A)

라인

Rel. Gold

Rel. XP

Rel. Damage

Rel. Lane CS

Rel. Jungle CS

TOP, MID, ADC

0.3

0.2

0.3

0.2

-

JUNGLE

0.3

0.3

0.1

-

0.3

SUP

0.4

0.4

0.2

-

-

4.2. Final Stats Score 가중치 (Section 2B)

라인

Solo Kills

KP

Vision Score

Turret DMG

Heal/Shield

CC Time

Obj. Stolen

TOP

0.4

0.1

0.1

0.4

-

-

-

JUNGLE

-

0.4

0.4

-

-

-

0.2

MID

0.3

0.5

0.1

0.1

-

-

-

ADC

-

0.5

0.1

0.4

-

-

-

SUP

0.2

0.2

0.4

-

0.2

0.2

-

5. 분석 결과 시각화 (Outputs)

이 모델을 통해 산출된 contribution 점수는 다음과 같은 시각화에 사용됩니다.

라인별 기여도 분포 (Violin Plot):

matchId 기준으로 라인별 평균 기여도를 집계하여, 승리팀과 패배팀 간의 라인별 기여도 분포 차이를 한눈에 비교할 수 있습니다.

(ADC/SUP 라인은 BOTTOM으로 통합하여 분석)

개인별 일관성 플롯 (Scatter Plot):

특정 소환사의 경기별 기여도 변화를 시각화하여, 해당 플레이어의 **기복(일관성)**을 파악하는 데 사용됩니다.

승/패 여부를 색깔로 표시하여 기여도와 승리 간의 관계를 분석합니다.
