"""
======================================================================
[최종 완성본 v8.4 - LoL 기여도 분석 (PCA Statistical Model)]
======================================================================
[수정 내역 (v8.4)]
1. [군집 분석] TOP, JUNGLE, MID, BOTTOM(ADC+SUP) 4개 그룹으로 분리하여 시각화
2. [레이더 차트] matplotlib 구버전 호환성 문제(AttributeError) 완벽 해결
3. [가중치 시각화] PCA 가중치 그래프 포함

[포함된 시각화 (총 7종)]
1. PCA 가중치 비교 (Bar Chart)
2. 5-1. 라인별 기여도 분포 (Violin Plot)
3. 5-2. 개인별 일관성 플롯 (Scatter Plot)
4. 5-3. [수정됨] 라인별 플레이어 유형 군집 분석 (2x2 Grid)
5. 5-4. 시간대별 기여도 곡선 (Timeline Line Chart)
6. 6-1, 6-2. 모델 검증 (ROC Curve & CM)
7. 7. (확장) Top Performer 벤치마킹 (Radar Chart)
======================================================================
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from math import pi

# 머신러닝 & 통계 라이브러리
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


#######################################################################
# 1. 데이터 전처리 (Raw Data -> DataFrame)
#######################################################################
def parse_all_match_data(base_path, num_files):
    all_frames_data = []
    print(f"[1단계] 총 {num_files}개의 매치 데이터 파싱 시작...")

    for i in tqdm(range(1, num_files + 1), desc="[1단계] 매치 파일 처리 중"):
        match_file = os.path.join(base_path, f'match_{i}.json')
        timeline_file = os.path.join(base_path, f'timeline_{i}.json')

        try:
            if not os.path.exists(match_file) or not os.path.exists(timeline_file):
                continue

            with open(match_file, 'r', encoding='utf-8') as f:
                match_data = json.load(f)

            if isinstance(match_data, list) and len(match_data) > 0:
                match_data_dict = match_data[0]
            else:
                match_data_dict = match_data

            participant_map = {}
            if 'info' not in match_data_dict or 'participants' not in match_data_dict['info']:
                continue

            for p in match_data_dict['info']['participants']:
                raw_pos = p.get('teamPosition', p.get('individualPosition', 'UNKNOWN'))
                if raw_pos == 'BOTTOM':
                    lane = 'ADC'
                elif raw_pos == 'UTILITY':
                    lane = 'SUP'
                elif raw_pos in ['TOP', 'JUNGLE', 'MIDDLE']:
                    lane = raw_pos.replace('MIDDLE', 'MID')
                else:
                    lane = 'UNKNOWN'

                game_name = p.get('riotIdGameName')
                tag_line = p.get('riotIdTagline')
                summoner_name = f"{game_name}#{tag_line}" if (game_name and tag_line) else p.get('summonerName',
                                                                                                 'UNKNOWN')

                participant_map[p['participantId']] = {
                    'summonerName': summoner_name,
                    'lane': lane,
                    'win': p['win'],
                    'deaths': p.get('deaths', 0),
                    'f_killParticipation': p.get('challenges', {}).get('killParticipation', 0),
                    'f_visionScore': p.get('visionScore', 0),
                    'f_damageDealtToTurrets': p.get('damageDealtToTurrets', 0),
                }

            with open(timeline_file, 'r', encoding='utf-8') as f:
                timeline_data = json.load(f)

            if isinstance(timeline_data, list) and len(timeline_data) > 0:
                timeline_data_dict = timeline_data[0]
            else:
                timeline_data_dict = timeline_data

            if 'info' not in timeline_data_dict or 'frames' not in timeline_data_dict['info']:
                continue

            for frame in timeline_data_dict['info']['frames']:
                minute = frame['timestamp'] // 60000
                if minute == 0: continue
                if 'participantFrames' not in frame: continue

                for pid_str, p_frame in frame['participantFrames'].items():
                    pid = int(pid_str)
                    p_info = participant_map.get(pid)
                    if not p_info or p_info['lane'] == 'UNKNOWN': continue

                    stats = {
                        'matchId': i,
                        'minute': minute,
                        'participantId': pid,
                        't_totalGold': p_frame['totalGold'],
                        't_xp': p_frame['xp'],
                        't_minionsKilled': p_frame['minionsKilled'],
                        't_damageToChampions': p_frame['damageStats']['totalDamageDoneToChampions'],
                        **p_info
                    }
                    all_frames_data.append(stats)
        except Exception:
            pass

    return pd.DataFrame(all_frames_data)


#######################################################################
# 2. 기여도 정의 (핵심: PCA를 통한 통계적 가중치 산출)
#######################################################################
def calculate_contribution_pca(df_minute_stats):
    print("[2단계] 기여도 계산 시작 (PCA 기반 가중치 자동 산출)...")

    df_opp = df_minute_stats[['matchId', 'minute', 'lane', 'win',
                              't_totalGold', 't_xp', 't_minionsKilled', 't_damageToChampions']].copy()

    df_merged = pd.merge(df_minute_stats, df_opp,
                         on=['matchId', 'minute', 'lane'], suffixes=('', '_opp'))

    df_merged = df_merged[df_merged['win'] != df_merged['win_opp']].copy()

    df_merged['diff_gold'] = (df_merged['t_totalGold'] - df_merged['t_totalGold_opp']) / (df_merged['minute'] * 100 + 1)
    df_merged['diff_xp'] = (df_merged['t_xp'] - df_merged['t_xp_opp']) / (df_merged['minute'] * 100 + 1)
    df_merged['diff_cs'] = (df_merged['t_minionsKilled'] - df_merged['t_minionsKilled_opp']) / (df_merged['minute'] + 1)

    df_medians = df_merged.groupby(['lane', 'minute'])['t_damageToChampions'].median().reset_index()
    df_medians.rename(columns={'t_damageToChampions': 'median_dmg'}, inplace=True)
    df_final = pd.merge(df_merged, df_medians, on=['lane', 'minute'])
    df_final['rel_dmg'] = df_final['t_damageToChampions'] / (df_final['median_dmg'] + 1)

    feature_cols = ['diff_gold', 'diff_xp', 'diff_cs', 'rel_dmg']
    df_final['pca_score'] = 0.0

    pca_weights = {}

    print("\n[PCA 학습 결과 - 라인별 Feature 중요도(가중치)]")
    for lane in df_final['lane'].unique():
        mask = df_final['lane'] == lane
        X = df_final.loc[mask, feature_cols]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=1)
        pca_score = pca.fit_transform(X_scaled)

        comp = pca.components_[0]
        if comp[0] < 0:
            comp = -comp
            pca_score = -pca_score

        pca_weights[lane] = dict(zip(['Gold Diff', 'XP Diff', 'CS Diff', 'Rel Dmg'], comp))
        print(f" > {lane} 가중치: Gold({comp[0]:.2f}), XP({comp[1]:.2f}), CS({comp[2]:.2f}), Dmg({comp[3]:.2f})")

        df_final.loc[mask, 'pca_score'] = pca_score.flatten()

    def normalize_to_one(x):
        return (x - x.median()) / (x.std() + 1e-5) * 0.5 + 1.0

    df_final['norm_score'] = df_final.groupby('lane')['pca_score'].transform(normalize_to_one)

    penalty_factor = df_final['lane'].apply(lambda x: 0.1 if x == 'SUP' else 0.15)
    df_final['minute_timeline_contrib'] = df_final['norm_score'] - (df_final['deaths'] * penalty_factor)

    df_final['minute_timeline_contrib'] = df_final['minute_timeline_contrib'].clip(lower=0.1)

    snapshot_minutes = [8, 10, 12, 15, 20]
    df_snapshot = df_final[df_final['minute'].isin(snapshot_minutes)].copy()

    df_agg = df_snapshot.groupby(['matchId', 'participantId', 'summonerName', 'lane', 'win'])[
        'minute_timeline_contrib'].mean().reset_index()
    df_agg.rename(columns={'minute_timeline_contrib': 'contribution'}, inplace=True)

    df_weights = pd.DataFrame(pca_weights).T.reset_index().rename(columns={'index': 'lane'})

    return df_agg, df_final, df_weights


#######################################################################
# 시각화 함수 모음
#######################################################################
def plot_pca_weights(df_weights):
    print("\n[추가 시각화] PCA 가중치 분포 생성...")
    df_plot = df_weights.melt(id_vars='lane', var_name='Feature', value_name='Weight')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x='Feature', y='Weight', hue='lane', palette='tab10')
    plt.axhline(0.0, color='black', linestyle='--', linewidth=0.8)
    plt.title('[3-2] Feature별 PCA 가중치 비교 (모델 해석)', fontsize=16)
    plt.ylabel('PCA Component Weight', fontsize=12)
    plt.legend(title='Lane')
    plt.show()


def plot_lane_distribution(df):
    plt.figure(figsize=(12, 6))
    df['plot_lane'] = df['lane'].replace({'ADC': 'BOTTOM', 'SUP': 'BOTTOM'})
    sns.violinplot(data=df, x='plot_lane', y='contribution', hue='win', split=True,
                   palette={True: 'cornflowerblue', False: 'tomato'}, order=['TOP', 'JUNGLE', 'MID', 'BOTTOM'])
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.title('[5-1] 라인별 기여도 분포 (PCA 기반)', fontsize=16)
    plt.show()


def plot_consistency(df, target_summoner):
    user_df = df[df['summonerName'] == target_summoner].reset_index()
    if user_df.empty:
        print(f"경고: {target_summoner} 소환사의 데이터가 없습니다.")
        return
    plt.figure(figsize=(12, 5))
    colors = user_df['win'].map({True: 'blue', False: 'red'})
    plt.scatter(user_df.index, user_df['contribution'], c=colors, s=80, alpha=0.7)
    plt.axhline(1.0, color='gray', linestyle='--', label='Baseline (1.0)')
    plt.axhline(user_df['contribution'].mean(), color='green', label=f'My Avg ({user_df["contribution"].mean():.2f})')
    plt.title(f"[5-2] '{target_summoner}' 기여도 일관성", fontsize=16)
    plt.legend()
    plt.show()


# [수정] 라인별(4개 그룹) 군집 분석 시각화
def plot_clustering_all(df):
    print("\n[5-3] 라인별(TOP, JUNGLE, MID, BOTTOM) 군집 분석 생성 중...")

    # 분석할 그룹 정의: BOTTOM은 ADC+SUP 통합
    groups = ['TOP', 'JUNGLE', 'MID', 'BOTTOM']

    # 2행 2열의 subplot 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, group in enumerate(groups):
        ax = axes[i]

        # 데이터 필터링 로직
        if group == 'BOTTOM':
            target_df = df[df['lane'].isin(['ADC', 'SUP'])].copy()  # ADC와 SUP 합침
        else:
            target_df = df[df['lane'] == group].copy()

        # 소환사별 집계
        stats = target_df.groupby('summonerName')['contribution'].agg(['mean', 'std', 'count']).reset_index()
        stats = stats[stats['count'] >= 5].fillna(0)  # 최소 5판 이상 데이터만 사용

        if len(stats) < 4:
            ax.text(0.5, 0.5, '데이터 부족 (Min 4 users)', ha='center', va='center')
            ax.set_title(f"{group} (Data insufficient)")
            continue

        # K-Means 군집화
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(stats[['mean', 'std']])
        stats['cluster'] = kmeans.labels_

        # Scatter Plot 그리기
        sns.scatterplot(data=stats, x='mean', y='std', hue='cluster',
                        palette='viridis', s=80, ax=ax, legend=False)

        ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_title(f"{group} Cluster (n={len(stats)})", fontsize=14)
        ax.set_xlabel('Avg Contribution')
        ax.set_ylabel('Consistency (Std)')

    plt.suptitle('[5-3] 라인별 플레이어 유형 군집화 (4 Groups)', fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_timeline(df_raw, match_id):
    match_df = df_raw[df_raw['matchId'] == match_id].copy()
    if match_df.empty: return
    match_df['plot_lane'] = match_df['lane'].replace({'ADC': 'BOTTOM', 'SUP': 'BOTTOM'})
    agg = match_df.groupby(['minute', 'plot_lane', 'win'])['minute_timeline_contrib'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=agg, x='minute', y='minute_timeline_contrib', hue='plot_lane', style='win', markers=True, lw=2)
    plt.axhline(1.0, color='gray', linestyle=':')
    plt.title(f"[5-4] Match {match_id} 시간대별 기여도 변화", fontsize=16)
    plt.show()


def plot_validation(df):
    fpr, tpr, _ = roc_curve(df['win'], df['contribution'])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('[6-1] 모델 변별력 (ROC Curve)')
    plt.legend()
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(df['win'], (df['contribution'] >= 1.0).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('[6-2] 승패 분류 Confusion Matrix')
    plt.show()


# [수정] 에러 방지를 위해 명시적 subplot 사용
def plot_radar(df, user1, user2):
    print(f"\n[7] {user1} vs {user2} 레이더 차트 생성 중...")
    stats1 = df[df['summonerName'] == user1].groupby('lane')['contribution'].mean()
    stats2 = df[df['summonerName'] == user2].groupby('lane')['contribution'].mean()

    if stats1.empty or stats2.empty:
        print("경고: 비교 대상의 데이터가 부족하여 레이더 차트를 생성할 수 없습니다.")
        return

    labels = ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUP']
    # 없는 라인은 중앙값 1.0으로 채움
    vals1 = [stats1.get(l, 1.0) for l in labels]
    vals2 = [stats2.get(l, 1.0) for l in labels]

    angles = [n / float(5) * 2 * pi for n in range(5)]
    angles += angles[:1]
    vals1 += vals1[:1]
    vals2 += vals2[:1]

    # AttributeError 해결을 위해 plt.figure() 호출 후 add_subplot 사용
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, vals1, label=user1, color='blue')
    ax.fill(angles, vals1, alpha=0.1, color='blue')
    ax.plot(angles, vals2, label=f"{user2} (Top)", color='red', linestyle='--')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('[7] Top Performer 벤치마킹 분석', y=1.08)
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
    plt.show()


#######################################################################
# 메인 실행
#######################################################################
if __name__ == "__main__":
    BASE_PATH = './match_data/'
    NUM_FILES = 1087  # 파일 개수 설정
    MY_ID = "Bonnie#0314"

    # 1. 데이터 로드
    df_raw = parse_all_match_data(BASE_PATH, NUM_FILES)

    if not df_raw.empty:
        # 2. PCA 기여도 계산
        df_contrib, df_full, df_weights = calculate_contribution_pca(df_raw)

        # 3. 1등 유저(프로급) 자동 탐지
        top_player = df_contrib.groupby('summonerName')['contribution'].mean().idxmax()

        # 내 아이디가 데이터셋에 있으면 사용, 없으면 1등과 비교를 위해 1등 유저를 그대로 Target으로 (예외 처리)
        if MY_ID in df_contrib['summonerName'].values:
            target_user = MY_ID
        else:
            print(f"알림: '{MY_ID}'가 데이터에 없습니다. 분석 대상을 임의의 유저로 대체합니다.")
            target_user = df_contrib['summonerName'].iloc[0]

        print(f"\n분석 대상: {target_user} vs 벤치마크(Top): {top_player}")

        # 4. 시각화 (총 7종)
        plot_pca_weights(df_weights)  # [3-2] 가중치
        plot_lane_distribution(df_contrib)  # [5-1] 분포
        plot_consistency(df_contrib, target_user)  # [5-2] 일관성
        plot_clustering_all(df_contrib)  # [5-3] 군집화 (수정됨: 4개 그룹)

        # 타임라인 시각화를 위한 매치 ID 추출
        if not df_full.empty:
            match_id = df_full['matchId'].iloc[0]
            plot_timeline(df_full, match_id)  # [5-4] 타임라인

        plot_validation(df_contrib)  # [6-1, 6-2] 검증
        plot_radar(df_contrib, target_user, top_player)  # [7] 벤치마킹

        print("\n=== 모든 분석 완료 ===")
    else:
        print("경고: 데이터를 찾을 수 없습니다. ./match_data/ 폴더에 파일이 있는지 확인하세요.")
