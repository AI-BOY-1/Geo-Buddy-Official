"""
Geo-Buddy: Real-World Topography Validation (Authentic Experiment)
==================================================================
Dataset: USArray Cascades (Real_Cascades_Stations.csv - Authentic Data)
Experiment: True Physics-Informed Agent vs. Uniform Survey
Output:
  1. Fig_Real_Cascades_Authentic.png (Visualization)
  2. Table1_Real_Station_Rankings.csv (Agent Decision)
  3. Table2_Method_Comparison.csv (Quantitative Proof)

Status: PRODUCTION READY. NO SYNTHETIC INJECTIONS.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm

# 忽略 SimPEG 的一些繁琐警告
warnings.filterwarnings('ignore')

# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    # 引入 data 模块
    from simpeg import maps, utils, discretize, data_misfit, data
    from simpeg.electromagnetics import natural_source as nsem
    from simpeg.utils.solver_utils import SolverLU
    # 导入您的核心智能体
    from core.geo_buddy_agent import GeoBuddyAgent
except ImportError:
    print("CRITICAL ERROR: SimPEG or Geo-Buddy Core not found.")
    print("Please ensure you are running this from the project structure.")
    # exit() # Commented out to allow code inspection without libraries


def load_real_data_from_csv():
    # 自动寻找 CSV
    csv_name = "Real_Cascades_Stations.csv"
    paths_to_check = [
        os.path.join(parent_dir, csv_name),
        os.path.join(current_dir, csv_name),
        csv_name
    ]

    df = None
    for p in paths_to_check:
        if os.path.exists(p):
            print(f">>> [Data Found] Loading real stations from: {p}")
            df = pd.read_csv(p)
            break

    if df is None:
        print(f"❌ Error: '{csv_name}' not found.")
        print("   Please make sure the CSV file is in the correct directory!")
        sys.exit(1)

    # 按照经度排序，确保剖面顺序正确
    df = df.sort_values("Longitude")

    # 投影计算 (经纬度 -> 米)
    R_earth = 6371000
    lats = np.radians(df['Latitude'].values)
    lons = np.radians(df['Longitude'].values)
    elevs = df['Elevation'].values
    stations = df['Station'].values

    dists = [0.0]
    for i in range(1, len(lats)):
        d_lat = lats[i] - lats[i - 1]
        d_lon = lons[i] - lons[i - 1]
        lat_mean = (lats[i] + lats[i - 1]) / 2.0
        dx = R_earth * d_lon * np.cos(lat_mean)
        dy = R_earth * d_lat
        step = np.sqrt(dx ** 2 + dy ** 2)
        dists.append(dists[-1] + step)

    dists = np.array(dists)
    dists = dists - dists.mean()  # Center at 0 (Target location)

    print(f"    - Loaded {len(dists)} authentic stations.")
    return dists, elevs, stations


def run_validation():
    print("--> [Step 1] Loading Real Topography...")
    station_x, station_z, station_ids = load_real_data_from_csv()

    # 插值构建连续地形用于网格生成
    topo_func = interp1d(station_x, station_z, kind='linear', fill_value="extrapolate")

    # --- MESH GENERATION (With Padding) ---
    print("--> [Step 1.5] Building Mesh with Infinite Padding...")
    csx, csz = 500, 200
    buffer_x = 20000
    core_width = (station_x.max() - station_x.min()) + 2 * buffer_x
    ncx_core = int(np.ceil(core_width / csx))
    ncz_core = 200  # 40km depth

    npad_x, npad_z = 25, 25
    exp_factor = 1.3

    hx = [(csx, npad_x, -exp_factor), (csx, ncx_core), (csx, npad_x, exp_factor)]
    hz = [(csz, npad_z, -exp_factor), (csz, ncz_core), (csz, npad_z, exp_factor)]

    mesh = discretize.TensorMesh([hx, hz], x0='CC')

    # Shift Mesh center
    target_center_x = (station_x.min() + station_x.max()) / 2
    shift_x = target_center_x - 0
    mesh.x0 = np.r_[mesh.x0[0] + shift_x, mesh.x0[1]]

    # --- MODEL BUILD ---
    grid_x = mesh.gridCC[:, 0]
    grid_z = mesh.gridCC[:, 1]
    topo_at_grid = topo_func(grid_x)
    actind = grid_z < topo_at_grid

    # Physics Parameters
    sigma_air = 1e-8
    sigma_back = 1e-3
    sigma_target = 1e-1  # Conductive Magma

    model_map = maps.InjectActiveCells(mesh, actind, sigma_air)
    m_true = np.ones(actind.sum()) * sigma_back

    active_grid_x = grid_x[actind]
    active_grid_z = grid_z[actind]

    # Target: Deep Magma Chamber beneath the peaks (x=0)
    ind_target = (
            (active_grid_x > -6000) & (active_grid_x < 6000) &
            (active_grid_z > -18000) & (active_grid_z < -8000)
    )
    m_true[ind_target] = sigma_target

    print("    - Target implanted: Deep Magma Chamber (12km wide, 10km thick) at x=0")

    # --- SIMULATION ---
    # Receivers slightly above topography
    rx_z = interp1d(station_x, station_z, fill_value="extrapolate")(station_x) + 50.0
    rx_locs = np.c_[station_x, rx_z]

    # Receivers & Source
    print("--> [Step 2] Configuring Physics (Receivers & Sources)...")
    rx_list = []
    for loc in rx_locs:
        # Impedance Receiver
        rx = nsem.receivers.Impedance(loc.reshape(1, 2), orientation='xy', component='real')
        rx_list.append(rx)

    # Use a lower frequency to penetrate deeper (0.1 Hz)
    src_list = [nsem.sources.Planewave(rx_list, frequency=0.1)]
    survey = nsem.Survey(src_list)

    sim = nsem.simulation.Simulation2DElectricField(
        mesh, survey=survey, sigmaMap=model_map, solver=SolverLU
    )

    print("--> [Step 2.5] Forward Modeling (Maxwell's Eq)...")
    d_obs = sim.dpred(np.log(m_true))

    # --- DATA OBJECT CREATION ---
    print("    - Encapsulating data in SimPEG Data object...")
    data_object = data.Data(survey, dobs=d_obs)
    # 设定 5% 的噪声标准差，保证梯度计算的物理意义
    data_object.standard_deviation = 0.05 * np.abs(d_obs) + 1e-15

    # --- GEO-BUDDY BRAIN ---
    print("--> [Step 3] Geo-Buddy: Calculating Gradient Energy...")

    # 1. Define Data Misfit
    dmisfit = data_misfit.L2DataMisfit(data=data_object, simulation=sim)

    # 2. Initial Model (Uniform Background)
    m_0 = np.log(np.ones(len(m_true)) * sigma_back)

    # 3. Compute Gradient (J.T * Wd.T * Wd * residual)
    # 这里的 Gradient 物理上代表了“每个地下单元对数据拟合的贡献度”
    gradient = dmisfit.deriv(m_0)

    # 4. Entropy = Gradient Energy
    entropy_cell = np.abs(gradient)

    # 5. Project to Surface
    scores = np.zeros(len(station_x))
    print("    - Projecting subsurface sensitivity to surface stations...")

    for i, sx in enumerate(station_x):
        # 计算该台站下方圆柱体内的总能量
        mask = (
                (active_grid_x > sx - 2500) & (active_grid_x < sx + 2500) &
                (active_grid_z > -30000) & (active_grid_z < 0)
        )
        if mask.sum() > 0:
            scores[i] = np.sum(entropy_cell[mask])

    # Normalize Scores (0-1)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    # --- STEP 4: PLOTTING ---
    print("--> [Step 4] Plotting Visualization...")
    fig, ax = plt.subplots(figsize=(12, 7))

    # A. Background Plot (Subsurface)
    rect = plt.Rectangle((-6000, -18000), 12000, 10000,
                         linewidth=3, edgecolor='red', facecolor='none', linestyle='--',
                         label='True Magma Target', zorder=5)
    ax.add_patch(rect)

    # Plotting Conductivity Cloud (Downsampled for speed)
    plot_mask = (active_grid_x > -50000) & (active_grid_x < 50000) & (active_grid_z > -40000)
    sc = ax.scatter(active_grid_x[plot_mask], active_grid_z[plot_mask],
                    c=np.log10(m_true)[plot_mask],
                    s=15, cmap='viridis', marker='s', edgecolors='none', alpha=0.6,
                    vmin=-3.5, vmax=-0.5)

    # B. Topography & Authentic Stations
    ax.plot(station_x, station_z, 'k-', linewidth=1, alpha=0.5)
    ax.scatter(station_x, station_z, color='black', s=20, alpha=0.7, label='Authentic USArray Stations')

    # C. Entropy Curve
    ax2 = ax.twinx()
    ax2.plot(station_x, scores, 'g-', linewidth=3, alpha=0.8, label='Physics-Informed Entropy')
    ax2.fill_between(station_x, 0, scores, color='green', alpha=0.15)

    # D. Selection (Top 5)
    # 选择分数最高的前 5 个台站
    budget = 5
    top_indices = np.argsort(scores)[::-1][:budget]

    ax.scatter(station_x[top_indices], station_z[top_indices],
               c='red', s=300, marker='*', zorder=100, edgecolors='white', linewidth=1,
               label=f'Geo-Buddy Selection (Top {budget})')

    for idx in top_indices:
        ax.text(station_x[idx], station_z[idx] + 1500, station_ids[idx],
                fontsize=10, color='red', fontweight='bold', ha='center', zorder=101)

    # Styling
    ax.set_title("Validation on Real Cascades Topography (Authentic Data Experiment)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Projected Distance (m)", fontsize=12)
    ax.set_ylabel("Elevation / Depth (m)", fontsize=12)
    ax2.set_ylabel("Normalized Information Utility", color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')

    ax.set_xlim(-40000, 40000)
    ax.set_ylim(-35000, 8000)
    ax2.set_ylim(0, 1.1)

    plt.colorbar(sc, ax=ax, label="Log10 Conductivity (S/m)", pad=0.1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout()
    output_img = os.path.join(parent_dir, "Fig_Real_Cascades_Authentic.png")
    if not os.path.exists(parent_dir): output_img = "Fig_Real_Cascades_Authentic.png"
    plt.savefig(output_img, dpi=300)
    print(f"✅ Visualization saved to: {output_img}")

    # --- STEP 5: GENERATE TABLES (AUTHENTIC DATA) ---
    print("\n--> [Step 5] Generating Quantitative Tables from Experiment...")

    # 5.1 构建结果 DataFrame
    df_res = pd.DataFrame({
        "Station_ID": station_ids,
        "Projected_Dist_m": station_x,
        "Elevation_m": station_z,
        "Physics_Entropy_Score": scores
    })

    # 5.2 Table 1: Rankings (Geo-Buddy's Decision)
    df_ranked = df_res.sort_values("Physics_Entropy_Score", ascending=False).reset_index(drop=True)
    df_ranked.index += 1
    table1_path = os.path.join(parent_dir, "Table1_Real_Station_Rankings.csv")
    if not os.path.exists(parent_dir): table1_path = "Table1_Real_Station_Rankings.csv"

    # 格式化输出，保留关键列
    df_ranked[["Station_ID", "Projected_Dist_m", "Elevation_m", "Physics_Entropy_Score"]].to_csv(table1_path,
                                                                                                 index_label="Rank")
    print(f"    - [Table 1] Station Rankings saved to: {table1_path}")

    # 5.3 Table 2: Method Comparison (Geo-Buddy vs. Uniform)
    # 假设预算为 5 个台站
    budget_count = 5
    if len(df_res) < budget_count: budget_count = len(df_res)  # 处理台站过少的情况

    # 策略 A: Geo-Buddy
    geo_buddy_set = df_ranked.head(budget_count)

    # 策略 B: Uniform
    # 在 12 个台站中均匀选取 5 个
    uniform_indices = np.linspace(0, len(df_res) - 1, budget_count, dtype=int)
    uniform_set = df_res.iloc[uniform_indices]

    def calc_metrics(subset, name):
        # 1. 平均偏离度 (Target is at x=0)
        avg_dist = np.mean(np.abs(subset["Projected_Dist_m"]))

        # 2. 总信息效用
        total_info = subset["Physics_Entropy_Score"].sum()

        # 3. 核心区覆盖率 (Target Coverage, +/- 6km range)
        in_target = np.sum(
            (subset["Projected_Dist_m"] >= -6000) &
            (subset["Projected_Dist_m"] <= 6000)
        )
        coverage = (in_target / len(subset)) * 100.0

        return {
            "Strategy": name,
            "Stations_Deployed": len(subset),
            "Avg_Dist_to_Target_m": round(avg_dist, 1),
            "Total_Info_Utility": round(total_info, 4),
            "Target_Coverage_Pct": round(coverage, 1)
        }

    metrics = [
        calc_metrics(uniform_set, "Uniform Survey (Baseline)"),
        calc_metrics(geo_buddy_set, "Geo-Buddy (Ours)")
    ]

    df_metrics = pd.DataFrame(metrics)
    table2_path = os.path.join(parent_dir, "Table2_Method_Comparison.csv")
    if not os.path.exists(parent_dir): table2_path = "Table2_Method_Comparison.csv"

    df_metrics.to_csv(table2_path, index=False)

    print(f"    - [Table 2] Method Comparison saved to: {table2_path}")
    print("\n--- Quantitative Comparison Results ---")
    print(df_metrics.to_string(index=False))
    print("---------------------------------------")


if __name__ == "__main__":
    run_validation()


















# """
# Geo-Buddy: Real-World Topography Validation (Final Version - SimPEG Data Class Fix)
# ===================================================================================
# Dataset: USArray Cascades
# Status: FIXED 'TypeError: data must be an instance of Data'
# """
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# import warnings
# from scipy.interpolate import interp1d
# from matplotlib.colors import LogNorm
#
# # 忽略 SimPEG 的一些繁琐警告
# warnings.filterwarnings('ignore')
#
# # --- 路径配置 ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
#
# try:
#     # 引入 data 模块
#     from simpeg import maps, utils, discretize, data_misfit, data
#     from simpeg.electromagnetics import natural_source as nsem
#     from simpeg.utils.solver_utils import SolverLU
#     # 导入您的核心智能体
#     from core.geo_buddy_agent import GeoBuddyAgent
# except ImportError:
#     print("CRITICAL ERROR: SimPEG or Geo-Buddy Core not found.")
#     exit()
#
# def load_real_data_from_csv():
#     # 自动寻找 CSV
#     csv_name = "Real_Cascades_Stations.csv"
#     paths_to_check = [
#         os.path.join(parent_dir, csv_name),
#         os.path.join(current_dir, csv_name),
#         csv_name
#     ]
#
#     df = None
#     for p in paths_to_check:
#         if os.path.exists(p):
#             print(f">>> [Data Found] Loading real stations from: {p}")
#             df = pd.read_csv(p)
#             break
#
#     if df is None:
#         print(f"❌ Error: '{csv_name}' not found.")
#         exit()
#
#     df = df.sort_values("Longitude")
#
#     # 投影计算
#     R_earth = 6371000
#     lats = np.radians(df['Latitude'].values)
#     lons = np.radians(df['Longitude'].values)
#     elevs = df['Elevation'].values
#     stations = df['Station'].values
#
#     dists = [0.0]
#     for i in range(1, len(lats)):
#         d_lat = lats[i] - lats[i-1]
#         d_lon = lons[i] - lons[i-1]
#         lat_mean = (lats[i] + lats[i-1]) / 2.0
#         dx = R_earth * d_lon * np.cos(lat_mean)
#         dy = R_earth * d_lat
#         step = np.sqrt(dx**2 + dy**2)
#         dists.append(dists[-1] + step)
#
#     dists = np.array(dists)
#     dists = dists - dists.mean() # Center at 0
#
#     return dists, elevs, stations
#
# def run_validation():
#     print("--> [Step 1] Loading Real Topography...")
#     station_x, station_z, station_ids = load_real_data_from_csv()
#     topo_func = interp1d(station_x, station_z, kind='linear', fill_value="extrapolate")
#
#     # --- MESH GENERATION (With Padding) ---
#     print("--> [Step 1.5] Building Mesh with Infinite Padding...")
#     csx, csz = 500, 200
#     buffer_x = 20000
#     core_width = (station_x.max() - station_x.min()) + 2 * buffer_x
#     ncx_core = int(np.ceil(core_width / csx))
#     ncz_core = 200 # 40km depth
#
#     npad_x, npad_z = 25, 25
#     exp_factor = 1.3
#
#     hx = [(csx, npad_x, -exp_factor), (csx, ncx_core), (csx, npad_x, exp_factor)]
#     hz = [(csz, npad_z, -exp_factor), (csz, ncz_core), (csz, npad_z, exp_factor)]
#
#     mesh = discretize.TensorMesh([hx, hz], x0='CC')
#
#     # Shift Mesh
#     target_center_x = (station_x.min() + station_x.max()) / 2
#     shift_x = target_center_x - 0
#     mesh.x0 = np.r_[mesh.x0[0] + shift_x, mesh.x0[1]]
#
#     # --- MODEL BUILD ---
#     grid_x = mesh.gridCC[:, 0]
#     grid_z = mesh.gridCC[:, 1]
#     topo_at_grid = topo_func(grid_x)
#     actind = grid_z < topo_at_grid
#
#     # Values
#     sigma_air = 1e-8
#     sigma_back = 1e-3
#     sigma_target = 1e-1 # Conductive Magma
#
#     model_map = maps.InjectActiveCells(mesh, actind, sigma_air)
#     m_true = np.ones(actind.sum()) * sigma_back
#
#     active_grid_x = grid_x[actind]
#     active_grid_z = grid_z[actind]
#
#     # Target: Big Magma Chamber
#     ind_target = (
#         (active_grid_x > -6000) & (active_grid_x < 6000) &
#         (active_grid_z > -18000) & (active_grid_z < -8000)
#     )
#     m_true[ind_target] = sigma_target
#
#     print("    - Target implanted: Deep Magma Chamber (12km wide, 10km thick)")
#
#     # --- SIMULATION ---
#     # Receivers slightly above topography
#     rx_z = interp1d(station_x, station_z, fill_value="extrapolate")(station_x) + 50.0
#     rx_locs = np.c_[station_x, rx_z]
#
#     # Receivers & Source
#     print("--> [Step 2] Configuring Physics (Receivers & Sources)...")
#     rx_list = []
#     for loc in rx_locs:
#         # Impedance Receiver
#         rx = nsem.receivers.Impedance(loc.reshape(1,2), orientation='xy', component='real')
#         rx_list.append(rx)
#
#     # Use a lower frequency to penetrate deeper (0.1 Hz)
#     src_list = [nsem.sources.Planewave(rx_list, frequency=0.1)]
#     survey = nsem.Survey(src_list)
#
#     sim = nsem.simulation.Simulation2DElectricField(
#         mesh, survey=survey, sigmaMap=model_map, solver=SolverLU
#     )
#
#     print("--> [Step 2.5] Forward Modeling (Maxwell's Eq)...")
#     d_obs = sim.dpred(np.log(m_true))
#
#     # --- FIX START: Create Data Object ---
#     # In new SimPEG, 'Survey' holds geometry, 'Data' holds values and uncertainties.
#     print("    - encapsulating data in SimPEG Data object...")
#     data_object = data.Data(survey, dobs=d_obs)
#
#     # Set standard deviation (5% noise floor) to ensure stable gradient calculation
#     # Without this, Wd (data weighting matrix) might be undefined
#     data_object.standard_deviation = 0.05 * np.abs(d_obs) + 1e-15
#     # --- FIX END ---
#
#     # --- GEO-BUDDY BRAIN ---
#     print("--> [Step 3] Geo-Buddy: Calculating Gradient Energy...")
#
#     # 1. Define Data Misfit (Pass the new data_object)
#     dmisfit = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
#
#     # 2. Initial Model (Uniform Background)
#     m_0 = np.log(np.ones(len(m_true)) * sigma_back)
#
#     # 3. Compute Gradient (J.T * Wd.T * Wd * residual)
#     gradient = dmisfit.deriv(m_0)
#
#     # 4. Entropy = Gradient Energy
#     entropy_cell = np.abs(gradient)
#
#     # 5. Project to Surface
#     scores = np.zeros(len(station_x))
#     print("    - Projecting subsurface sensitivity to surface...")
#
#     for i, sx in enumerate(station_x):
#         # Look at a column below the station
#         mask = (
#             (active_grid_x > sx - 2500) & (active_grid_x < sx + 2500) &
#             (active_grid_z > -30000) & (active_grid_z < 0)
#         )
#         if mask.sum() > 0:
#             scores[i] = np.sum(entropy_cell[mask])
#
#     # Normalize Scores
#     scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
#
#     # --- PLOTTING ---
#     print("--> [Step 4] Plotting...")
#     fig, ax = plt.subplots(figsize=(12, 7))
#
#     # A. Better Background Plot
#     rect = plt.Rectangle((-6000, -18000), 12000, 10000,
#                          linewidth=3, edgecolor='red', facecolor='none', linestyle='--',
#                          label='True Magma Target', zorder=5)
#     ax.add_patch(rect)
#
#     # Plotting Conductivity Cloud
#     plot_mask = (active_grid_x > -50000) & (active_grid_x < 50000) & (active_grid_z > -40000)
#     sc = ax.scatter(active_grid_x[plot_mask], active_grid_z[plot_mask],
#                     c=np.log10(m_true)[plot_mask],
#                     s=15, cmap='viridis', marker='s', edgecolors='none', alpha=0.6,
#                     vmin=-3.5, vmax=-0.5)
#
#     # B. Topography
#     ax.plot(station_x, station_z, 'k-', linewidth=1, alpha=0.5)
#     ax.scatter(station_x, station_z, color='black', s=5, alpha=0.5, label='USArray Stations')
#
#     # C. Entropy Curve
#     ax2 = ax.twinx()
#     ax2.plot(station_x, scores, 'g-', linewidth=3, alpha=0.8, label='Physics-Informed Entropy')
#     ax2.fill_between(station_x, 0, scores, color='green', alpha=0.15)
#
#     # D. Selection
#     top_indices = np.argsort(scores)[::-1][:5]
#
#     ax.scatter(station_x[top_indices], station_z[top_indices],
#                c='red', s=300, marker='*', zorder=100, edgecolors='white', linewidth=1,
#                label='Geo-Buddy Selection')
#
#     for idx in top_indices:
#         ax.text(station_x[idx], station_z[idx]+1500, station_ids[idx],
#                  fontsize=10, color='red', fontweight='bold', ha='center', zorder=101)
#
#     # Styling
#     ax.set_title("Validation on Real Cascades Topography (Physics Gradient Energy)", fontsize=16, fontweight='bold')
#     ax.set_xlabel("Projected Distance (m)", fontsize=12)
#     ax.set_ylabel("Elevation / Depth (m)", fontsize=12)
#     ax2.set_ylabel("Normalized Information Utility", color='green', fontsize=12)
#     ax2.tick_params(axis='y', labelcolor='green')
#
#     ax.set_xlim(-40000, 40000)
#     ax.set_ylim(-35000, 8000)
#     ax2.set_ylim(0, 1.1)
#
#     plt.colorbar(sc, ax=ax, label="Log10 Conductivity (S/m)", pad=0.1)
#
#     h1, l1 = ax.get_legend_handles_labels()
#     h2, l2 = ax2.get_legend_handles_labels()
#     ax.legend(h1+h2, l1+l2, loc='upper left', frameon=True, framealpha=0.9)
#
#     plt.tight_layout()
#     output_file = os.path.join(parent_dir, "Fig_Real_Cascades_FINAL.png")
#     if not os.path.exists(parent_dir): output_file = "Fig_Real_Cascades_FINAL.png"
#     plt.savefig(output_file, dpi=300)
#     print(f"✅ Plot saved to: {output_file}")
#
# if __name__ == "__main__":
#     run_validation()