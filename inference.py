"""
NFL BIG DATA BOWL 2026 - GEOMETRIC GRU INFERENCE
API Submission Script

This script loads pre-trained models and artifacts to perform inference.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import polars as pl
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import warnings

import kaggle_evaluation.nfl_inference_server

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # Path to your saved model artifacts (uploaded to Kaggle dataset)
    MODEL_DIR = Path("/kaggle/input/enhanced-gru-inference-v1/")  # Change this to your Kaggle dataset path

    SEED = 42
    N_FOLDS = 5
    WINDOW_SIZE = 10
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 94
    BATCH_SIZE = 256

    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3

    K_NEIGH = 6
    RADIUS = 30.0
    TAU = 8.0
    N_ROUTE_CLUSTERS = 7

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(Config.SEED)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_velocity(speed, direction_deg):
    theta = np.deg2rad(direction_deg)
    return speed * np.sin(theta), speed * np.cos(theta)

def height_to_feet(height_str):
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches/12
    except:
        return 6.0

# ============================================================================
# GEOMETRIC BASELINE
# ============================================================================

def compute_geometric_endpoint(df):
    df = df.copy()

    if 'num_frames_output' in df.columns:
        t_total = df['num_frames_output'] / 10.0
    else:
        t_total = 3.0

    df['time_to_endpoint'] = t_total

    df['geo_endpoint_x'] = df['x'] + df['velocity_x'] * t_total
    df['geo_endpoint_y'] = df['y'] + df['velocity_y'] * t_total

    if 'ball_land_x' in df.columns:
        receiver_mask = df['player_role'] == 'Targeted Receiver'
        df.loc[receiver_mask, 'geo_endpoint_x'] = df.loc[receiver_mask, 'ball_land_x']
        df.loc[receiver_mask, 'geo_endpoint_y'] = df.loc[receiver_mask, 'ball_land_y']

        defender_mask = df['player_role'] == 'Defensive Coverage'
        has_mirror = df.get('mirror_offset_x', 0).notna() & (df.get('mirror_wr_dist', 50) < 15)
        coverage_mask = defender_mask & has_mirror

        df.loc[coverage_mask, 'geo_endpoint_x'] = (
            df.loc[coverage_mask, 'ball_land_x'] +
            df.loc[coverage_mask, 'mirror_offset_x'].fillna(0)
        )
        df.loc[coverage_mask, 'geo_endpoint_y'] = (
            df.loc[coverage_mask, 'ball_land_y'] +
            df.loc[coverage_mask, 'mirror_offset_y'].fillna(0)
        )

    df['geo_endpoint_x'] = df['geo_endpoint_x'].clip(Config.FIELD_X_MIN, Config.FIELD_X_MAX)
    df['geo_endpoint_y'] = df['geo_endpoint_y'].clip(Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)

    return df

def add_geometric_features(df):
    df = compute_geometric_endpoint(df)

    df['geo_vector_x'] = df['geo_endpoint_x'] - df['x']
    df['geo_vector_y'] = df['geo_endpoint_y'] - df['y']
    df['geo_distance'] = np.sqrt(df['geo_vector_x']**2 + df['geo_vector_y']**2)

    t = df['time_to_endpoint'] + 0.1
    df['geo_required_vx'] = df['geo_vector_x'] / t
    df['geo_required_vy'] = df['geo_vector_y'] / t

    df['geo_velocity_error_x'] = df['geo_required_vx'] - df['velocity_x']
    df['geo_velocity_error_y'] = df['geo_required_vy'] - df['velocity_y']
    df['geo_velocity_error'] = np.sqrt(
        df['geo_velocity_error_x']**2 + df['geo_velocity_error_y']**2
    )

    t_sq = t * t
    df['geo_required_ax'] = 2 * df['geo_vector_x'] / t_sq
    df['geo_required_ay'] = 2 * df['geo_vector_y'] / t_sq
    df['geo_required_ax'] = df['geo_required_ax'].clip(-10, 10)
    df['geo_required_ay'] = df['geo_required_ay'].clip(-10, 10)

    velocity_mag = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    geo_unit_x = df['geo_vector_x'] / (df['geo_distance'] + 0.1)
    geo_unit_y = df['geo_vector_y'] / (df['geo_distance'] + 0.1)
    df['geo_alignment'] = (
        df['velocity_x'] * geo_unit_x + df['velocity_y'] * geo_unit_y
    ) / (velocity_mag + 0.1)

    df['geo_receiver_urgency'] = df['is_receiver'] * df['geo_distance'] / (t + 0.1)
    df['geo_defender_coupling'] = df['is_coverage'] * (1.0 / (df.get('mirror_wr_dist', 50) + 1.0))

    return df

def add_ball_flight_features(df):
    df = df.copy()

    if 'ball_land_x' not in df.columns or 'distance_to_ball' not in df.columns:
        return df

    distance_to_ball = df['distance_to_ball']
    df['estimated_ball_land_time'] = 1.0 + 0.07 * np.sqrt(distance_to_ball)
    df['estimated_ball_land_time'] = df['estimated_ball_land_time'].clip(0.8, 4.0)

    current_speed = df['s'] + 0.1
    df['time_to_reach_ball'] = distance_to_ball / current_speed

    df['catch_window'] = df['estimated_ball_land_time'] - df['time_to_reach_ball']

    df['can_catch'] = (
        (df['catch_window'] > -0.5) &
        (df['catch_window'] < 2.0) &
        (df['is_receiver'] == 1)
    ).astype(float)

    df['receiver_urgency'] = np.where(
        df['is_receiver'] == 1,
        np.clip(-df['catch_window'], 0, 3),
        0
    )

    velocity_mag = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2) + 0.1
    ball_dir_x = (df['ball_land_x'] - df['x']) / (distance_to_ball + 0.1)
    ball_dir_y = (df['ball_land_y'] - df['y']) / (distance_to_ball + 0.1)
    df['route_alignment'] = (
        df['velocity_x'] * ball_dir_x + df['velocity_y'] * ball_dir_y
    ) / velocity_mag

    if 'is_coverage' in df.columns:
        df['defender_can_contest'] = (
            (df['catch_window'] > 0) &
            (df['is_coverage'] == 1)
        ).astype(float)
    else:
        df['defender_can_contest'] = 0.0

    return df

# ============================================================================
# OPPONENT FEATURES
# ============================================================================

def get_opponent_features(input_df):
    features = []

    for (gid, pid), group in input_df.groupby(['game_id', 'play_id']):
        last = group.sort_values('frame_id').groupby('nfl_id').last()

        if len(last) < 2:
            continue

        positions = last[['x', 'y']].values
        sides = last['player_side'].values
        speeds = last['s'].values
        directions = last['dir'].values
        roles = last['player_role'].values

        receiver_mask = np.isin(roles, ['Targeted Receiver', 'Other Route Runner'])

        for i, (nid, side, role) in enumerate(zip(last.index, sides, roles)):
            opp_mask = sides != side

            feat = {
                'game_id': gid, 'play_id': pid, 'nfl_id': nid,
                'nearest_opp_dist': 50.0, 'closing_speed': 0.0,
                'num_nearby_opp_3': 0, 'num_nearby_opp_5': 0,
                'mirror_wr_vx': 0.0, 'mirror_wr_vy': 0.0,
                'mirror_offset_x': 0.0, 'mirror_offset_y': 0.0,
                'mirror_wr_dist': 50.0,
            }

            if not opp_mask.any():
                features.append(feat)
                continue

            opp_positions = positions[opp_mask]
            distances = np.sqrt(((positions[i] - opp_positions)**2).sum(axis=1))

            if len(distances) == 0:
                features.append(feat)
                continue

            nearest_idx = distances.argmin()
            feat['nearest_opp_dist'] = distances[nearest_idx]
            feat['num_nearby_opp_3'] = (distances < 3.0).sum()
            feat['num_nearby_opp_5'] = (distances < 5.0).sum()

            my_vx, my_vy = get_velocity(speeds[i], directions[i])
            opp_speeds = speeds[opp_mask]
            opp_dirs = directions[opp_mask]
            opp_vx, opp_vy = get_velocity(opp_speeds[nearest_idx], opp_dirs[nearest_idx])

            rel_vx = my_vx - opp_vx
            rel_vy = my_vy - opp_vy
            to_me = positions[i] - opp_positions[nearest_idx]
            to_me_norm = to_me / (np.linalg.norm(to_me) + 0.1)
            feat['closing_speed'] = -(rel_vx * to_me_norm[0] + rel_vy * to_me_norm[1])

            if role == 'Defensive Coverage' and receiver_mask.any():
                rec_positions = positions[receiver_mask]
                rec_distances = np.sqrt(((positions[i] - rec_positions)**2).sum(axis=1))

                if len(rec_distances) > 0:
                    closest_rec_idx = rec_distances.argmin()
                    rec_indices = np.where(receiver_mask)[0]
                    actual_rec_idx = rec_indices[closest_rec_idx]

                    rec_vx, rec_vy = get_velocity(speeds[actual_rec_idx], directions[actual_rec_idx])

                    feat['mirror_wr_vx'] = rec_vx
                    feat['mirror_wr_vy'] = rec_vy
                    feat['mirror_wr_dist'] = rec_distances[closest_rec_idx]
                    feat['mirror_offset_x'] = positions[i][0] - rec_positions[closest_rec_idx][0]
                    feat['mirror_offset_y'] = positions[i][1] - rec_positions[closest_rec_idx][1]

            features.append(feat)

    return pd.DataFrame(features)

# ============================================================================
# ROUTE PATTERNS
# ============================================================================

def extract_route_patterns(input_df, kmeans=None, scaler=None, fit=True):
    route_features = []

    for (gid, pid, nid), group in input_df.groupby(['game_id', 'play_id', 'nfl_id']):
        traj = group.sort_values('frame_id').tail(5)

        if len(traj) < 3:
            continue

        positions = traj[['x', 'y']].values
        speeds = traj['s'].values

        total_dist = np.sum(np.sqrt(np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2))
        displacement = np.sqrt((positions[-1, 0] - positions[0, 0])**2 +
                              (positions[-1, 1] - positions[0, 1])**2)
        straightness = displacement / (total_dist + 0.1)

        angles = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
        if len(angles) > 1:
            angle_changes = np.abs(np.diff(angles))
            max_turn = np.max(angle_changes)
            mean_turn = np.mean(angle_changes)
        else:
            max_turn = mean_turn = 0

        speed_mean = speeds.mean()
        speed_change = speeds[-1] - speeds[0] if len(speeds) > 1 else 0
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]

        route_features.append({
            'game_id': gid, 'play_id': pid, 'nfl_id': nid,
            'traj_straightness': straightness,
            'traj_max_turn': max_turn,
            'traj_mean_turn': mean_turn,
            'traj_depth': abs(dx),
            'traj_width': abs(dy),
            'speed_mean': speed_mean,
            'speed_change': speed_change,
        })

    route_df = pd.DataFrame(route_features)
    feat_cols = ['traj_straightness', 'traj_max_turn', 'traj_mean_turn',
                 'traj_depth', 'traj_width', 'speed_mean', 'speed_change']
    X = route_df[feat_cols].fillna(0)

    if not fit:
        X_scaled = scaler.transform(X)
        route_df['route_pattern'] = kmeans.predict(X_scaled)

    return route_df

# ============================================================================
# GNN EMBEDDINGS
# ============================================================================

def compute_neighbor_embeddings(input_df, k_neigh=Config.K_NEIGH,
                                radius=Config.RADIUS, tau=Config.TAU):
    cols_needed = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y",
                   "velocity_x", "velocity_y", "player_side"]
    src = input_df[cols_needed].copy()

    last = (src.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
               .groupby(["game_id", "play_id", "nfl_id"], as_index=False)
               .tail(1)
               .rename(columns={"frame_id": "last_frame_id"})
               .reset_index(drop=True))

    tmp = last.merge(
        src.rename(columns={
            "frame_id": "nb_frame_id", "nfl_id": "nfl_id_nb",
            "x": "x_nb", "y": "y_nb",
            "velocity_x": "vx_nb", "velocity_y": "vy_nb",
            "player_side": "player_side_nb"
        }),
        left_on=["game_id", "play_id", "last_frame_id"],
        right_on=["game_id", "play_id", "nb_frame_id"],
        how="left"
    )

    tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]
    tmp["dx"] = tmp["x_nb"] - tmp["x"]
    tmp["dy"] = tmp["y_nb"] - tmp["y"]
    tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
    tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
    tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)

    tmp = tmp[np.isfinite(tmp["dist"]) & (tmp["dist"] > 1e-6)]
    if radius is not None:
        tmp = tmp[tmp["dist"] <= radius]

    tmp["is_ally"] = (tmp["player_side_nb"] == tmp["player_side"]).astype(np.float32)

    keys = ["game_id", "play_id", "nfl_id"]
    tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
    if k_neigh is not None:
        tmp = tmp[tmp["rnk"] <= float(k_neigh)]

    tmp["w"] = np.exp(-tmp["dist"] / float(tau))
    sum_w = tmp.groupby(keys)["w"].transform("sum")
    tmp["wn"] = np.where(sum_w > 0, tmp["w"] / sum_w, 0.0)

    tmp["wn_ally"] = tmp["wn"] * tmp["is_ally"]
    tmp["wn_opp"] = tmp["wn"] * (1.0 - tmp["is_ally"])

    for col in ["dx", "dy", "dvx", "dvy"]:
        tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
        tmp[f"{col}_opp_w"] = tmp[col] * tmp["wn_opp"]

    tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
    tmp["dist_opp"] = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)

    ag = tmp.groupby(keys).agg(
        gnn_ally_dx_mean=("dx_ally_w", "sum"),
        gnn_ally_dy_mean=("dy_ally_w", "sum"),
        gnn_ally_dvx_mean=("dvx_ally_w", "sum"),
        gnn_ally_dvy_mean=("dvy_ally_w", "sum"),
        gnn_opp_dx_mean=("dx_opp_w", "sum"),
        gnn_opp_dy_mean=("dy_opp_w", "sum"),
        gnn_opp_dvx_mean=("dvx_opp_w", "sum"),
        gnn_opp_dvy_mean=("dvy_opp_w", "sum"),
        gnn_ally_cnt=("is_ally", "sum"),
        gnn_opp_cnt=("is_ally", lambda s: float(len(s) - s.sum())),
        gnn_ally_dmin=("dist_ally", "min"),
        gnn_ally_dmean=("dist_ally", "mean"),
        gnn_opp_dmin=("dist_opp", "min"),
        gnn_opp_dmean=("dist_opp", "mean"),
    ).reset_index()

    near = tmp.loc[tmp["rnk"] <= 3, keys + ["rnk", "dist"]].copy()
    if len(near) > 0:
        near["rnk"] = near["rnk"].astype(int)
        dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
        dwide = dwide.rename(columns={1: "gnn_d1", 2: "gnn_d2", 3: "gnn_d3"}).reset_index()
        ag = ag.merge(dwide, on=keys, how="left")

    for c in ["gnn_ally_dx_mean", "gnn_ally_dy_mean", "gnn_ally_dvx_mean", "gnn_ally_dvy_mean",
              "gnn_opp_dx_mean", "gnn_opp_dy_mean", "gnn_opp_dvx_mean", "gnn_opp_dvy_mean"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_cnt", "gnn_opp_cnt"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_dmin", "gnn_opp_dmin", "gnn_ally_dmean", "gnn_opp_dmean",
              "gnn_d1", "gnn_d2", "gnn_d3"]:
        ag[c] = ag[c].fillna(radius if radius is not None else 30.0)

    return ag

# ============================================================================
# SEQUENCE PREPARATION
# ============================================================================

def prepare_sequences_geometric(input_df, test_template, window_size,
                                route_kmeans, route_scaler):
    """Inference version - no training logic"""

    input_df = input_df.copy()
    input_df = input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])

    # Base features
    input_df['player_height_feet'] = input_df['player_height'].apply(height_to_feet)
    height_parts = input_df['player_height'].str.split('-', expand=True)
    input_df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    input_df['bmi'] = (input_df['player_weight'] / (input_df['height_inches']**2)) * 703

    dir_rad = np.deg2rad(input_df['dir'].fillna(0))
    input_df['velocity_x'] = input_df['s'] * np.sin(dir_rad)
    input_df['velocity_y'] = input_df['s'] * np.cos(dir_rad)
    input_df['acceleration_x'] = input_df['a'] * np.cos(dir_rad)
    input_df['acceleration_y'] = input_df['a'] * np.sin(dir_rad)

    input_df['speed_squared'] = input_df['s'] ** 2
    input_df['accel_magnitude'] = np.sqrt(input_df['acceleration_x']**2 + input_df['acceleration_y']**2)
    input_df['momentum_x'] = input_df['velocity_x'] * input_df['player_weight']
    input_df['momentum_y'] = input_df['velocity_y'] * input_df['player_weight']
    input_df['kinetic_energy'] = 0.5 * input_df['player_weight'] * input_df['speed_squared']

    input_df['orientation_diff'] = np.abs(input_df['o'] - input_df['dir'])
    input_df['orientation_diff'] = np.minimum(input_df['orientation_diff'], 360 - input_df['orientation_diff'])

    input_df['is_offense'] = (input_df['player_side'] == 'Offense').astype(int)
    input_df['is_defense'] = (input_df['player_side'] == 'Defense').astype(int)
    input_df['is_receiver'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
    input_df['is_coverage'] = (input_df['player_role'] == 'Defensive Coverage').astype(int)
    input_df['is_passer'] = (input_df['player_role'] == 'Passer').astype(int)
    input_df['role_targeted_receiver'] = input_df['is_receiver']
    input_df['role_defensive_coverage'] = input_df['is_coverage']
    input_df['role_passer'] = input_df['is_passer']
    input_df['side_offense'] = input_df['is_offense']

    if 'ball_land_x' in input_df.columns:
        ball_dx = input_df['ball_land_x'] - input_df['x']
        ball_dy = input_df['ball_land_y'] - input_df['y']
        input_df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
        input_df['dist_to_ball'] = input_df['distance_to_ball']
        input_df['dist_squared'] = input_df['distance_to_ball'] ** 2
        input_df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)
        input_df['ball_direction_x'] = ball_dx / (input_df['distance_to_ball'] + 1e-6)
        input_df['ball_direction_y'] = ball_dy / (input_df['distance_to_ball'] + 1e-6)
        input_df['closing_speed_ball'] = (
            input_df['velocity_x'] * input_df['ball_direction_x'] +
            input_df['velocity_y'] * input_df['ball_direction_y']
        )
        input_df['velocity_toward_ball'] = (
            input_df['velocity_x'] * np.cos(input_df['angle_to_ball']) +
            input_df['velocity_y'] * np.sin(input_df['angle_to_ball'])
        )
        input_df['velocity_alignment'] = np.cos(input_df['angle_to_ball'] - dir_rad)
        input_df['angle_diff'] = np.abs(input_df['o'] - np.degrees(input_df['angle_to_ball']))
        input_df['angle_diff'] = np.minimum(input_df['angle_diff'], 360 - input_df['angle_diff'])

    # Advanced features
    opp_features = get_opponent_features(input_df)
    input_df = input_df.merge(opp_features, on=['game_id', 'play_id', 'nfl_id'], how='left')

    route_features = extract_route_patterns(input_df, route_kmeans, route_scaler, fit=False)
    input_df = input_df.merge(route_features, on=['game_id', 'play_id', 'nfl_id'], how='left')

    gnn_features = compute_neighbor_embeddings(input_df)
    input_df = input_df.merge(gnn_features, on=['game_id', 'play_id', 'nfl_id'], how='left')

    if 'nearest_opp_dist' in input_df.columns:
        input_df['pressure'] = 1 / np.maximum(input_df['nearest_opp_dist'], 0.5)
        input_df['under_pressure'] = (input_df['nearest_opp_dist'] < 3).astype(int)
        input_df['pressure_x_speed'] = input_df['pressure'] * input_df['s']

    if 'mirror_wr_vx' in input_df.columns:
        s_safe = np.maximum(input_df['s'], 0.1)
        input_df['mirror_similarity'] = (
            input_df['velocity_x'] * input_df['mirror_wr_vx'] +
            input_df['velocity_y'] * input_df['mirror_wr_vy']
        ) / s_safe
        input_df['mirror_offset_dist'] = np.sqrt(
            input_df['mirror_offset_x']**2 + input_df['mirror_offset_y']**2
        )
        input_df['mirror_alignment'] = input_df['mirror_similarity'] * input_df['role_defensive_coverage']

    # Temporal features
    gcols = ['game_id', 'play_id', 'nfl_id']

    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in input_df.columns:
                input_df[f'{col}_lag{lag}'] = input_df.groupby(gcols)[col].shift(lag)

    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in input_df.columns:
                input_df[f'{col}_rolling_mean_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).mean()
                      .reset_index(level=[0,1,2], drop=True)
                )
                input_df[f'{col}_rolling_std_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).std()
                      .reset_index(level=[0,1,2], drop=True)
                )

    for col in ['velocity_x', 'velocity_y']:
        if col in input_df.columns:
            input_df[f'{col}_delta'] = input_df.groupby(gcols)[col].diff()

    input_df['velocity_x_ema'] = input_df.groupby(gcols)['velocity_x'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['velocity_y_ema'] = input_df.groupby(gcols)['velocity_y'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['speed_ema'] = input_df.groupby(gcols)['s'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )

    # Time features
    if 'num_frames_output' in input_df.columns:
        max_frames = input_df['num_frames_output']

        input_df['max_play_duration'] = max_frames / 10.0
        input_df['frame_time'] = input_df['frame_id'] / 10.0
        input_df['progress_ratio'] = input_df['frame_id'] / np.maximum(max_frames, 1)
        input_df['time_remaining'] = (max_frames - input_df['frame_id']) / 10.0
        input_df['frames_remaining'] = max_frames - input_df['frame_id']

        input_df['expected_x_at_ball'] = input_df['x'] + input_df['velocity_x'] * input_df['frame_time']
        input_df['expected_y_at_ball'] = input_df['y'] + input_df['velocity_y'] * input_df['frame_time']

        if 'ball_land_x' in input_df.columns:
            input_df['error_from_ball_x'] = input_df['expected_x_at_ball'] - input_df['ball_land_x']
            input_df['error_from_ball_y'] = input_df['expected_y_at_ball'] - input_df['ball_land_y']
            input_df['error_from_ball'] = np.sqrt(
                input_df['error_from_ball_x']**2 + input_df['error_from_ball_y']**2
            )

            input_df['weighted_dist_by_time'] = input_df['dist_to_ball'] / (input_df['frame_time'] + 0.1)
            input_df['dist_scaled_by_progress'] = input_df['dist_to_ball'] * (1 - input_df['progress_ratio'])

        input_df['time_squared'] = input_df['frame_time'] ** 2
        input_df['velocity_x_progress'] = input_df['velocity_x'] * input_df['progress_ratio']
        input_df['velocity_y_progress'] = input_df['velocity_y'] * input_df['progress_ratio']
        input_df['speed_scaled_by_time_left'] = input_df['s'] * input_df['time_remaining']

        input_df['actual_play_length'] = max_frames
        input_df['length_ratio'] = max_frames / 30.0

    # Geometric features
    input_df = add_geometric_features(input_df)

    # Ball-in-air features
    input_df = add_ball_flight_features(input_df)

    # Feature list (174 features total)
    feature_cols = [
        'x', 'y', 's', 'a', 'o', 'dir', 'frame_id', 'ball_land_x', 'ball_land_y',
        'player_height_feet', 'player_weight', 'height_inches', 'bmi',
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
        'momentum_x', 'momentum_y', 'kinetic_energy',
        'speed_squared', 'accel_magnitude', 'orientation_diff',
        'is_offense', 'is_defense', 'is_receiver', 'is_coverage', 'is_passer',
        'role_targeted_receiver', 'role_defensive_coverage', 'role_passer', 'side_offense',
        'distance_to_ball', 'dist_to_ball', 'dist_squared', 'angle_to_ball',
        'ball_direction_x', 'ball_direction_y', 'closing_speed_ball',
        'velocity_toward_ball', 'velocity_alignment', 'angle_diff',
        'nearest_opp_dist', 'closing_speed', 'num_nearby_opp_3', 'num_nearby_opp_5',
        'mirror_wr_vx', 'mirror_wr_vy', 'mirror_offset_x', 'mirror_offset_y',
        'pressure', 'under_pressure', 'pressure_x_speed',
        'mirror_similarity', 'mirror_offset_dist', 'mirror_alignment',
        'route_pattern', 'traj_straightness', 'traj_max_turn', 'traj_mean_turn',
        'traj_depth', 'traj_width', 'speed_mean', 'speed_change',
        'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean',
        'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
        'gnn_ally_cnt', 'gnn_opp_cnt',
        'gnn_ally_dmin', 'gnn_ally_dmean', 'gnn_opp_dmin', 'gnn_opp_dmean',
        'gnn_d1', 'gnn_d2', 'gnn_d3',
    ]

    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            feature_cols.append(f'{col}_lag{lag}')

    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            feature_cols.append(f'{col}_rolling_mean_{window}')
            feature_cols.append(f'{col}_rolling_std_{window}')

    feature_cols.extend(['velocity_x_delta', 'velocity_y_delta'])
    feature_cols.extend(['velocity_x_ema', 'velocity_y_ema', 'speed_ema'])

    feature_cols.extend([
        'max_play_duration', 'frame_time', 'progress_ratio', 'time_remaining', 'frames_remaining',
        'expected_x_at_ball', 'expected_y_at_ball',
        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball',
        'time_squared', 'weighted_dist_by_time',
        'velocity_x_progress', 'velocity_y_progress', 'dist_scaled_by_progress',
        'speed_scaled_by_time_left', 'actual_play_length', 'length_ratio',
    ])

    # Geometric features (13)
    feature_cols.extend([
        'geo_endpoint_x', 'geo_endpoint_y',
        'geo_vector_x', 'geo_vector_y', 'geo_distance',
        'geo_required_vx', 'geo_required_vy',
        'geo_velocity_error_x', 'geo_velocity_error_y', 'geo_velocity_error',
        'geo_required_ax', 'geo_required_ay',
        'geo_alignment',
    ])

    # Ball-in-air features (7)
    feature_cols.extend([
        'estimated_ball_land_time',
        'time_to_reach_ball',
        'catch_window',
        'can_catch',
        'receiver_urgency',
        'route_alignment',
        'defender_can_contest',
    ])

    feature_cols = [c for c in feature_cols if c in input_df.columns]

    # Create sequences
    input_df.set_index(['game_id', 'play_id', 'nfl_id'], inplace=True)
    grouped = input_df.groupby(level=['game_id', 'play_id', 'nfl_id'])

    target_groups = test_template[['game_id', 'play_id', 'nfl_id']].drop_duplicates()

    sequences, sequence_ids = [], []
    geo_endpoints_x, geo_endpoints_y = [], []

    for _, row in target_groups.iterrows():
        key = (row['game_id'], row['play_id'], row['nfl_id'])

        try:
            group_df = grouped.get_group(key)
        except KeyError:
            continue

        input_window = group_df.tail(window_size)

        if len(input_window) < window_size:
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)

        input_window = input_window.fillna(group_df.mean(numeric_only=True))
        seq = input_window[feature_cols].values

        if np.isnan(seq).any():
            seq = np.nan_to_num(seq, nan=0.0)

        sequences.append(seq)

        geo_x = input_window.iloc[-1]['geo_endpoint_x']
        geo_y = input_window.iloc[-1]['geo_endpoint_y']
        geo_endpoints_x.append(geo_x)
        geo_endpoints_y.append(geo_y)

        sequence_ids.append({
            'game_id': key[0],
            'play_id': key[1],
            'nfl_id': key[2],
            'frame_id': input_window.iloc[-1]['frame_id']
        })

    return sequences, sequence_ids, geo_endpoints_x, geo_endpoints_y

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class JointSeqModel(nn.Module):
    def __init__(self, input_dim, horizon):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.pool_ln = nn.LayerNorm(128)
        self.pool_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, 128))

        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, horizon * 2)
        )

    def forward(self, x):
        h, _ = self.gru(x)
        B = h.size(0)
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        out = self.head(ctx.squeeze(1))
        out = out.view(B, -1, 2)
        return torch.cumsum(out, dim=1)

# ============================================================================
# PREDICTOR CLASS
# ============================================================================

class NFLPredictor:
    def __init__(self):
        self.config = Config
        print("Loading trained artifacts...")

        # Load route clustering objects
        print("  Loading route artifacts...")
        with open(self.config.MODEL_DIR / "route_kmeans.pkl", "rb") as f:
            self.route_kmeans = pickle.load(f)
        with open(self.config.MODEL_DIR / "route_scaler.pkl", "rb") as f:
            self.route_scaler = pickle.load(f)
        print("  ✓ Loaded route_kmeans and route_scaler")

        # Infer input dimension from scaler
        with open(self.config.MODEL_DIR / "scaler_fold1.pkl", "rb") as f:
            scaler1 = pickle.load(f)
        input_dim = scaler1.mean_.shape[0]
        print(f"  ✓ Detected input_dim = {input_dim} features")

        # Load all fold models and scalers
        self.models = []
        self.scalers = []

        print(f"  Loading {self.config.N_FOLDS} fold models...")
        for fold in range(1, self.config.N_FOLDS + 1):
            model_path = self.config.MODEL_DIR / f"model_fold{fold}.pt"
            scaler_path = self.config.MODEL_DIR / f"scaler_fold{fold}.pkl"

            # Load scaler
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            # Load model
            model = JointSeqModel(input_dim, self.config.MAX_FUTURE_HORIZON)
            state = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state)
            model.to(self.config.DEVICE)
            model.eval()

            self.models.append(model)
            self.scalers.append(scaler)

            print(f"    ✓ Fold {fold}: loaded")

        print(f"✓ Successfully loaded all artifacts! Ready for inference.")

    def predict(self, test: pl.DataFrame, test_input: pl.DataFrame) -> pd.DataFrame:
        """Inference function called by the API"""

        # Convert Polars to Pandas
        test_input_pd = test_input.to_pandas()
        test_template_pd = test.to_pandas()

        # Prepare sequences
        test_seq, test_ids, test_geo_x, test_geo_y = prepare_sequences_geometric(
            test_input_pd, test_template_pd, self.config.WINDOW_SIZE,
            self.route_kmeans, self.route_scaler
        )

        X_test = list(test_seq)
        x_last = np.array([s[-1, 0] for s in X_test])
        y_last = np.array([s[-1, 1] for s in X_test])

        # Ensemble prediction
        all_preds = []
        H = self.config.MAX_FUTURE_HORIZON

        for model, sc in zip(self.models, self.scalers):
            X_sc = [sc.transform(s) for s in X_test]
            X_t = torch.tensor(np.stack(X_sc).astype(np.float32)).to(self.config.DEVICE)

            with torch.no_grad():
                preds = model(X_t).cpu().numpy()

            all_preds.append(preds)

        ens_preds = np.mean(all_preds, axis=0)

        # Format submission
        rows = []
        for i, sid in enumerate(test_ids):
            fids = test_template_pd[
                (test_template_pd['game_id'] == sid['game_id']) &
                (test_template_pd['play_id'] == sid['play_id']) &
                (test_template_pd['nfl_id'] == sid['nfl_id'])
            ]['frame_id'].sort_values().tolist()

            for t, fid in enumerate(fids):
                tt = min(t, H - 1)
                px = np.clip(x_last[i] + ens_preds[i, tt, 0], 0, self.config.FIELD_X_MAX)
                py = np.clip(y_last[i] + ens_preds[i, tt, 1], 0, self.config.FIELD_Y_MAX)

                rows.append({'x': px, 'y': py})

        return pd.DataFrame(rows)

# ============================================================================
# API SETUP
# ============================================================================

predictor = NFLPredictor()

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """API wrapper function"""
    return predictor.predict(test, test_input)

# API server
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))
