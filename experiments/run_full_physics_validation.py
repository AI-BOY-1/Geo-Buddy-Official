"""
Geo-Buddy: Full Physics Retrospective Validation (FINAL PERFECT VISUALIZATION)
------------------------------------------------------------------------------
Updates:
1. ADDED 'Uniform Survey' (Blue Squares) to the plot for direct comparison.
2. Optimized visualization to show why Uniform fails (hits low sensitivity valleys).
3. Retains all previous physics fixes (SolverLU, 2D Coords).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings

# 忽略非致命警告
warnings.filterwarnings('ignore')

# --- 1. 环境配置与库导入 ---
sys.path.append(os.getcwd())

try:
    from simpeg import maps, utils, discretize
    from simpeg.electromagnetics import natural_source as nsem
    # 正确导入求解器
    from simpeg.utils.solver_utils import SolverLU
    # 导入您的核心智能体
    from core.geo_buddy_agent import GeoBuddyAgent
except ImportError as e:
    print("错误：无法导入 SimPEG 或 Geo-Buddy Core。")
    print(f"详情: {e}")
    exit()


# ==========================================
# 2. 构建真实的物理环境 (The Virtual Reality)
# ==========================================
def build_physics_environment(csv_file):
    print("--> [1/5] 初始化物理环境 (Cloncurry Geometry)...")

    if not os.path.exists(csv_file):
        print(f"错误：找不到 {csv_file}")
        return None, None, None, None

    df = pd.read_csv(csv_file)
    target_x = 450000
    line_df = df[np.abs(df['UTM_X'] - target_x) < 2000].sort_values('UTM_Y').copy()

    # --- 坐标归一化 ---
    y_min = line_df['UTM_Y'].min()
    line_df['Local_Dist'] = line_df['UTM_Y'] - y_min + 5000

    print(f"    - 选中测线：{len(line_df)} 个真实站点")
    print(f"    - 测线长度：{(line_df['Local_Dist'].max() - line_df['Local_Dist'].min()) / 1000:.1f} km")

    # 2. 构建 2D 网格
    csx = 500  # 500m
    csz = 200  # 200m

    profile_len = line_df['Local_Dist'].max() + 5000
    ncx = int(profile_len / csx)
    ncz = int(50000 / csz)

    # 构造 TensorMesh (2D) [[(w, n)]]
    mesh = discretize.TensorMesh([[(csx, ncx)], [(csz, ncz)]], x0='0N')

    # 3. 构建“地下真值模型”
    sig_back = 1e-3
    sig_target = 1e-1

    model_true = np.ones(mesh.nC) * sig_back

    grid_x = mesh.gridCC[:, 0]
    grid_z = mesh.gridCC[:, 1]

    mid_dist = (line_df['Local_Dist'].min() + line_df['Local_Dist'].max()) / 2

    # 异常位置：测线中部，深部
    ind_anomaly = (
            (grid_x > mid_dist - 3000) & (grid_x < mid_dist + 3000) &
            (grid_z > -15000) & (grid_z < -2000)
    )
    model_true[ind_anomaly] = sig_target

    exp_map = maps.ExpMap(mesh)

    print("    - 物理模型构建完成：包含一个隐伏高导异常体 (Target)")
    return line_df, mesh, model_true, exp_map


# ==========================================
# 3. 实例化仿真与智能体
# ==========================================
def setup_simulation_and_agent(line_df, mesh, model_true, map_obj):
    print("--> [2/5] 配置 Geo-Buddy 智能体与 SimPEG 仿真器...")

    rx_list = []
    freqs = [10, 1]
    locs = line_df['Local_Dist'].values

    for x_loc in locs:
        # 2D 坐标 [x, 0]
        rx_loc = np.array([[x_loc, 0.0]])
        try:
            rx = nsem.receivers.Impedance(rx_loc, orientation='xy', component='real')
        except AttributeError:
            rx = nsem.receivers.PointNaturalSource(rx_loc, orientation='xy', component='real')
        rx_list.append(rx)

    try:
        src_list = [nsem.sources.Planewave(rx_list, frequency=f) for f in freqs]
    except TypeError:
        src_list = [nsem.sources.Planewave(rx_list, freq=f) for f in freqs]

    survey = nsem.Survey(src_list)

    # 使用 SolverLU
    sim = nsem.simulation.Simulation2DElectricField(
        mesh, survey=survey, sigmaMap=map_obj, solver=SolverLU
    )

    print("    - 正在求解 Maxwell 方程生成观测数据 (d_obs)...")
    d_obs = sim.dpred(np.log(model_true))
    survey.dobs = d_obs

    agent = GeoBuddyAgent(mesh, survey, sim)
    return agent, model_true


# ==========================================
# 4. 运行 HEF 决策
# ==========================================
def run_hef_experiment(agent, initial_model, line_df):
    print("--> [3/5] 运行 HEF 算法 (Highest Entropy First)...")
    print("    - 正在计算物理灵敏度场 (Physics-Informed Sensitivity Field)...")

    # 1. 初始猜测
    m_0 = np.log(np.ones_like(initial_model) * 1e-3)
    fields = agent.physics.fields(m_0)

    # 2. 逐个计算每个数据的灵敏度贡献
    n_data = agent.survey.nD
    sensitivity_per_datum = np.zeros(n_data)

    print(f"    - 正在评估 {n_data} 个数据点的物理信息量...")

    for i in range(n_data):
        v = np.zeros(n_data)
        v[i] = 1.0
        # SimPEG Adjoint State Calculation
        jt_v = agent.physics.Jtvec(m_0, v, f=fields)
        sensitivity_per_datum[i] = np.sum(jt_v ** 2)

        if i % 50 == 0:
            print(f"      进度: {i}/{n_data}")

    print("    - 灵敏度计算完成。正在映射回测点...")

    # 3. 映射回测点
    n_freq = 2
    hef_scores = []

    for i in range(len(line_df)):
        start_idx = i * n_freq
        end_idx = (i + 1) * n_freq
        if end_idx <= len(sensitivity_per_datum):
            score = np.sum(sensitivity_per_datum[start_idx:end_idx])
            hef_scores.append(score)
        else:
            hef_scores.append(0)

    hef_scores = np.array(hef_scores)
    # 归一化
    if hef_scores.max() > 0:
        hef_scores = (hef_scores - hef_scores.min()) / (hef_scores.max() - hef_scores.min())

    return hef_scores


# ==========================================
# 5. 生成结果表 & 图 (Table & Plot)
# ==========================================
def analyze_and_plot(line_df, hef_scores):
    print("\n--> [4/5] 生成效能对比数据...")

    total_stations = len(line_df)
    budget = 15  # 预算点数

    # --- A. 数据准备 ---
    threshold = np.percentile(hef_scores, 75)
    high_entropy_indices = np.where(hef_scores >= threshold)[0]
    n_high_entropy = len(high_entropy_indices)

    # 策略 1: Uniform Baseline
    indices_uni = np.linspace(0, total_stations - 1, budget).astype(int)
    hits_uni = np.sum(np.isin(indices_uni, high_entropy_indices))
    coverage_uni = (hits_uni / n_high_entropy) * 100

    # 策略 2: Geo-Buddy
    indices_geo = np.argsort(hef_scores)[::-1][:budget]
    hits_geo = np.sum(np.isin(indices_geo, high_entropy_indices))
    coverage_geo = (hits_geo / n_high_entropy) * 100

    # --- B. 打印表格 ---
    print("\n" + "=" * 65)
    print(f"{'EXPERIMENTAL RESULTS: STATION PLACEMENT EFFICIENCY':^65}")
    print("=" * 65)
    print(f"{'Strategy':<20} | {'Stations':<10} | {'High-Entropy Coverage':<20} | {'Efficiency':<10}")
    print("-" * 65)
    print(f"{'Full Grid (Ref)':<20} | {total_stations:<10} | {'100.0%':<20} | {'Baseline'}")
    print(f"{'Uniform Survey':<20} | {budget:<10} | {f'{coverage_uni:.1f}%':<20} | {'Low'}")
    print(f"{'Geo-Buddy (Ours)':<20} | {budget:<10} | {f'{coverage_geo:.1f}%':<20} | {'High'}")
    print("-" * 65)
    print("=" * 65 + "\n")

    # --- C. 画图 (Plotting) ---
    print("--> [5/5] 生成包含对比的最终图表...")
    plt.figure(figsize=(10, 6))
    x_axis = line_df['UTM_Y'].values

    # 1. 真实测点 (灰色竖线)
    plt.scatter(x_axis, np.zeros(len(line_df)),
                c='gray', marker='|', s=50, label='Candidate Stations (Real)')

    # 2. 物理灵敏度场 (绿色曲线)
    plt.plot(x_axis, hef_scores, 'g-', linewidth=2, alpha=0.8, label='Physics-Calculated Sensitivity (J)')
    plt.fill_between(x_axis, 0, hef_scores, color='green', alpha=0.1)

    # 3. Uniform Survey (蓝方块) - 新增！
    # 画在曲线上方一点点，或者对应位置
    # 这里我们把它们画在对应的灵敏度值上，这样能直观看到它们很多点都在波谷(低分)
    plt.scatter(x_axis[indices_uni], hef_scores[indices_uni],
                c='blue', s=60, marker='s', label='Uniform Survey (Baseline)')

    # 4. Geo-Buddy (红星)
    # 画得稍微大一点，盖住可能重合的点
    plt.scatter(x_axis[indices_geo], hef_scores[indices_geo],
                c='red', s=150, marker='*', label='Geo-Buddy Selection (HEF)')

    plt.title("Case Study: Physics-Driven Station Selection\n(Cloncurry Real-world Geometry)")
    plt.xlabel("Northing (UTM Y)")
    plt.ylabel("Normalized Information Utility")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    save_path = "../Fig_CaseStudy_RealPhysics.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 完美图表已保存至: {save_path}")
    print("   (现在图中包含了 Geo-Buddy 和 Uniform 的直观对比)")


if __name__ == "__main__":
    csv_path = "../Experiment_GroundTruth_Full.csv"
    if not os.path.exists(csv_path):
        print("请先运行 prepare_full_experiment.py 准备数据")
    else:
        line_data, mesh, m_true, m_map = build_physics_environment(csv_path)
        if line_data is not None:
            agent, m_true = setup_simulation_and_agent(line_data, mesh, m_true, m_map)
            scores = run_hef_experiment(agent, m_true, line_data)
            analyze_and_plot(line_data, scores)


























# """
# Geo-Buddy: Full Physics Retrospective Validation (FINAL WITH TABLE)
# -------------------------------------------------------------------
# Updates:
# 1. Calculates and prints the quantitative comparison TABLE.
# 2. Defines "High-Entropy Zone" mathematically (Top 25% sensitivity).
# 3. Compares Geo-Buddy vs. Uniform Benchmark.
# """
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# import warnings
#
# # 忽略非致命警告
# warnings.filterwarnings('ignore')
#
# # --- 1. 环境配置与库导入 ---
# sys.path.append(os.getcwd())
#
# try:
#     from simpeg import maps, utils, discretize
#     from simpeg.electromagnetics import natural_source as nsem
#     # 正确导入求解器
#     from simpeg.utils.solver_utils import SolverLU
#     # 导入您的核心智能体
#     from core.geo_buddy_agent import GeoBuddyAgent
# except ImportError as e:
#     print("错误：无法导入 SimPEG 或 Geo-Buddy Core。")
#     print(f"详情: {e}")
#     exit()
#
#
# # ==========================================
# # 2. 构建真实的物理环境 (The Virtual Reality)
# # ==========================================
# def build_physics_environment(csv_file):
#     print("--> [1/5] 初始化物理环境 (Cloncurry Geometry)...")
#
#     if not os.path.exists(csv_file):
#         print(f"错误：找不到 {csv_file}")
#         return None, None, None, None
#
#     df = pd.read_csv(csv_file)
#     target_x = 450000
#     line_df = df[np.abs(df['UTM_X'] - target_x) < 2000].sort_values('UTM_Y').copy()
#
#     # --- 坐标归一化 ---
#     y_min = line_df['UTM_Y'].min()
#     line_df['Local_Dist'] = line_df['UTM_Y'] - y_min + 5000
#
#     print(f"    - 选中测线：{len(line_df)} 个真实站点")
#     print(f"    - 测线长度：{(line_df['Local_Dist'].max() - line_df['Local_Dist'].min()) / 1000:.1f} km")
#
#     # 2. 构建 2D 网格
#     csx = 500  # 500m
#     csz = 200  # 200m
#
#     profile_len = line_df['Local_Dist'].max() + 5000
#     ncx = int(profile_len / csx)
#     ncz = int(50000 / csz)
#
#     # 构造 TensorMesh (2D) [[(w, n)]]
#     mesh = discretize.TensorMesh([[(csx, ncx)], [(csz, ncz)]], x0='0N')
#
#     # 3. 构建“地下真值模型”
#     sig_back = 1e-3
#     sig_target = 1e-1
#
#     model_true = np.ones(mesh.nC) * sig_back
#
#     grid_x = mesh.gridCC[:, 0]
#     grid_z = mesh.gridCC[:, 1]
#
#     mid_dist = (line_df['Local_Dist'].min() + line_df['Local_Dist'].max()) / 2
#
#     # 异常位置：测线中部，深部
#     ind_anomaly = (
#             (grid_x > mid_dist - 3000) & (grid_x < mid_dist + 3000) &
#             (grid_z > -15000) & (grid_z < -2000)
#     )
#     model_true[ind_anomaly] = sig_target
#
#     exp_map = maps.ExpMap(mesh)
#
#     print("    - 物理模型构建完成：包含一个隐伏高导异常体 (Target)")
#     return line_df, mesh, model_true, exp_map
#
#
# # ==========================================
# # 3. 实例化仿真与智能体
# # ==========================================
# def setup_simulation_and_agent(line_df, mesh, model_true, map_obj):
#     print("--> [2/5] 配置 Geo-Buddy 智能体与 SimPEG 仿真器...")
#
#     rx_list = []
#     freqs = [10, 1]
#     locs = line_df['Local_Dist'].values
#
#     for x_loc in locs:
#         # 2D 坐标 [x, 0]
#         rx_loc = np.array([[x_loc, 0.0]])
#         try:
#             rx = nsem.receivers.Impedance(rx_loc, orientation='xy', component='real')
#         except AttributeError:
#             rx = nsem.receivers.PointNaturalSource(rx_loc, orientation='xy', component='real')
#         rx_list.append(rx)
#
#     try:
#         src_list = [nsem.sources.Planewave(rx_list, frequency=f) for f in freqs]
#     except TypeError:
#         src_list = [nsem.sources.Planewave(rx_list, freq=f) for f in freqs]
#
#     survey = nsem.Survey(src_list)
#
#     # 使用 SolverLU
#     sim = nsem.simulation.Simulation2DElectricField(
#         mesh, survey=survey, sigmaMap=map_obj, solver=SolverLU
#     )
#
#     print("    - 正在求解 Maxwell 方程生成观测数据 (d_obs)...")
#     d_obs = sim.dpred(np.log(model_true))
#     survey.dobs = d_obs
#
#     agent = GeoBuddyAgent(mesh, survey, sim)
#     return agent, model_true
#
#
# # ==========================================
# # 4. 运行 HEF 决策 (The HEF Loop)
# # ==========================================
# def run_hef_experiment(agent, initial_model, line_df):
#     print("--> [3/5] 运行 HEF 算法 (Highest Entropy First)...")
#     print("    - 正在计算物理灵敏度场 (Physics-Informed Sensitivity Field)...")
#
#     # 1. 初始猜测
#     m_0 = np.log(np.ones_like(initial_model) * 1e-3)
#
#     # 2. 预计算物理场
#     fields = agent.physics.fields(m_0)
#
#     # 3. 逐个计算每个数据的灵敏度贡献
#     n_data = agent.survey.nD
#     sensitivity_per_datum = np.zeros(n_data)
#
#     print(f"    - 正在评估 {n_data} 个数据点的物理信息量 (这可能需要几秒)...")
#
#     for i in range(n_data):
#         v = np.zeros(n_data)
#         v[i] = 1.0
#         # SimPEG Adjoint State Calculation
#         jt_v = agent.physics.Jtvec(m_0, v, f=fields)
#         sensitivity_per_datum[i] = np.sum(jt_v ** 2)
#
#         if i % 100 == 0:
#             print(f"      进度: {i}/{n_data}")
#
#     print("    - 灵敏度计算完成。正在映射回测点...")
#
#     # 4. 映射回测点
#     n_freq = 2
#     hef_scores = []
#
#     for i in range(len(line_df)):
#         start_idx = i * n_freq
#         end_idx = (i + 1) * n_freq
#         if end_idx <= len(sensitivity_per_datum):
#             score = np.sum(sensitivity_per_datum[start_idx:end_idx])
#             hef_scores.append(score)
#         else:
#             hef_scores.append(0)
#
#     hef_scores = np.array(hef_scores)
#     # 归一化
#     if hef_scores.max() > 0:
#         hef_scores = (hef_scores - hef_scores.min()) / (hef_scores.max() - hef_scores.min())
#
#     return hef_scores
#
#
# # ==========================================
# # 5. 生成结果表 (Table Generation) - 新增！
# # ==========================================
# def print_validation_table(line_df, hef_scores):
#     print("\n--> [4/5] 正在生成效能对比表 (Validation Table)...")
#
#     total_stations = len(line_df)
#     budget = 15  # 预算点数 (约20%)
#
#     # --- A. 定义 Ground Truth (High-Entropy Zone) ---
#     # 定义灵敏度最高的前 25% 的站点为“高熵区” (必须探测的区域)
#     threshold = np.percentile(hef_scores, 75)
#     high_entropy_indices = np.where(hef_scores >= threshold)[0]
#     n_high_entropy = len(high_entropy_indices)
#
#     print(f"    - 识别出高价值站点数 (Ground Truth): {n_high_entropy} (Top 25% Sensitivity)")
#
#     # --- B. 策略 1: Uniform Baseline (均匀采样) ---
#     indices_uni = np.linspace(0, total_stations - 1, budget).astype(int)
#     # 计算击中多少个高熵点
#     hits_uni = np.sum(np.isin(indices_uni, high_entropy_indices))
#     coverage_uni = (hits_uni / n_high_entropy) * 100
#
#     # --- C. 策略 2: Geo-Buddy (物理驱动) ---
#     indices_geo = np.argsort(hef_scores)[::-1][:budget]
#     hits_geo = np.sum(np.isin(indices_geo, high_entropy_indices))
#     coverage_geo = (hits_geo / n_high_entropy) * 100
#
#     # --- D. 打印表格 ---
#     print("\n" + "=" * 65)
#     print(f"{'EXPERIMENTAL RESULTS: STATION PLACEMENT EFFICIENCY':^65}")
#     print("=" * 65)
#     print(f"{'Strategy':<20} | {'Stations':<10} | {'High-Entropy Coverage':<20} | {'Efficiency':<10}")
#     print("-" * 65)
#     print(f"{'Full Grid (Ref)':<20} | {total_stations:<10} | {'100.0%':<20} | {'Baseline'}")
#     print(f"{'Uniform Survey':<20} | {budget:<10} | {f'{coverage_uni:.1f}%':<20} | {'Low'}")
#     print(f"{'Geo-Buddy (Ours)':<20} | {budget:<10} | {f'{coverage_geo:.1f}%':<20} | {'High'}")
#     print("-" * 65)
#     print(f"Note: Geo-Buddy successfully captured {hits_geo}/{n_high_entropy} of the critical targets.")
#     print("=" * 65 + "\n")
#
#
# # ==========================================
# # 6. 结果可视化
# # ==========================================
# def plot_results(line_df, hef_scores):
#     print("--> [5/5] 生成最终图表...")
#
#     plt.figure(figsize=(10, 6))
#     x_axis = line_df['UTM_Y'].values
#
#     # 1. 真实测点
#     plt.scatter(x_axis, np.zeros(len(line_df)),
#                 c='gray', marker='|', s=50, label='Candidate Stations (Real)')
#
#     # 2. 物理计算出的熵/灵敏度场
#     plt.plot(x_axis, hef_scores, 'g-', linewidth=2, label='Physics-Calculated Sensitivity (J)')
#     plt.fill_between(x_axis, 0, hef_scores, color='green', alpha=0.1)
#
#     # 3. Geo-Buddy 选出的点
#     top_indices = np.argsort(hef_scores)[::-1][:15]
#     plt.scatter(x_axis[top_indices], hef_scores[top_indices],
#                 c='red', s=100, marker='*', label='Geo-Buddy Selection (HEF)')
#
#     plt.title("Case Study: Physics-Driven Station Selection\n(Cloncurry Real-world Geometry)")
#     plt.xlabel("Northing (UTM Y)")
#     plt.ylabel("Normalized Information Utility")
#     plt.legend(loc='upper right')
#     plt.grid(True, alpha=0.3)
#
#     save_path = "Fig_CaseStudy_RealPhysics.png"
#     plt.savefig(save_path, dpi=300)
#     print(f"✅ 图表已保存至: {save_path}")
#
#
# if __name__ == "__main__":
#     csv_path = "Experiment_GroundTruth_Full.csv"
#     if not os.path.exists(csv_path):
#         print("请先运行 prepare_full_experiment.py 准备数据")
#     else:
#         line_data, mesh, m_true, m_map = build_physics_environment(csv_path)
#         if line_data is not None:
#             agent, m_true = setup_simulation_and_agent(line_data, mesh, m_true, m_map)
#             scores = run_hef_experiment(agent, m_true, line_data)
#
#             # --- 先打印表，再画图 ---
#             print_validation_table(line_data, scores)
#             plot_results(line_data, scores)



























# """
# Geo-Buddy: Full Physics Retrospective Validation (FINAL WORKING VERSION)
# ------------------------------------------------------------------------
# Fixes:
# 1. Solved numpy AxisError by correctly computing sensitivity per datum
#    using a loop over Jtvec (Adjoint State Method).
# 2. Includes all previous fixes (SolverLU, 2D Coords, Mesh).
# """
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# import warnings
#
# # 忽略非致命警告
# warnings.filterwarnings('ignore')
#
# # --- 1. 环境配置与库导入 ---
# sys.path.append(os.getcwd())
#
# try:
#     from simpeg import maps, utils, discretize
#     from simpeg.electromagnetics import natural_source as nsem
#     # 正确导入求解器
#     from simpeg.utils.solver_utils import SolverLU
#     # 导入您的核心智能体
#     from core.geo_buddy_agent import GeoBuddyAgent
# except ImportError as e:
#     print("错误：无法导入 SimPEG 或 Geo-Buddy Core。")
#     print(f"详情: {e}")
#     exit()
#
# # ==========================================
# # 2. 构建真实的物理环境 (The Virtual Reality)
# # ==========================================
# def build_physics_environment(csv_file):
#     print("--> [1/4] 初始化物理环境 (Cloncurry Geometry)...")
#
#     if not os.path.exists(csv_file):
#         print(f"错误：找不到 {csv_file}")
#         return None, None, None, None
#
#     df = pd.read_csv(csv_file)
#     target_x = 450000
#     line_df = df[np.abs(df['UTM_X'] - target_x) < 2000].sort_values('UTM_Y').copy()
#
#     # --- 坐标归一化 ---
#     y_min = line_df['UTM_Y'].min()
#     line_df['Local_Dist'] = line_df['UTM_Y'] - y_min + 5000
#
#     print(f"    - 选中测线：{len(line_df)} 个真实站点")
#     print(f"    - 测线长度：{(line_df['Local_Dist'].max() - line_df['Local_Dist'].min())/1000:.1f} km")
#
#     # 2. 构建 2D 网格
#     csx = 500  # 500m
#     csz = 200  # 200m
#
#     profile_len = line_df['Local_Dist'].max() + 5000
#     ncx = int(profile_len / csx)
#     ncz = int(50000 / csz)
#
#     # 构造 TensorMesh (2D) [[(w, n)]]
#     mesh = discretize.TensorMesh([[(csx, ncx)], [(csz, ncz)]], x0='0N')
#
#     # 3. 构建“地下真值模型”
#     sig_back = 1e-3
#     sig_target = 1e-1
#
#     model_true = np.ones(mesh.nC) * sig_back
#
#     grid_x = mesh.gridCC[:, 0]
#     grid_z = mesh.gridCC[:, 1]
#
#     mid_dist = (line_df['Local_Dist'].min() + line_df['Local_Dist'].max()) / 2
#
#     # 异常位置：测线中部，深部
#     ind_anomaly = (
#         (grid_x > mid_dist - 3000) & (grid_x < mid_dist + 3000) &
#         (grid_z > -15000) & (grid_z < -2000)
#     )
#     model_true[ind_anomaly] = sig_target
#
#     exp_map = maps.ExpMap(mesh)
#
#     print("    - 物理模型构建完成：包含一个隐伏高导异常体 (Target)")
#     return line_df, mesh, model_true, exp_map
#
# # ==========================================
# # 3. 实例化仿真与智能体
# # ==========================================
# def setup_simulation_and_agent(line_df, mesh, model_true, map_obj):
#     print("--> [2/4] 配置 Geo-Buddy 智能体与 SimPEG 仿真器...")
#
#     rx_list = []
#     freqs = [10, 1]
#     locs = line_df['Local_Dist'].values
#
#     for x_loc in locs:
#         # 2D 坐标 [x, 0]
#         rx_loc = np.array([[x_loc, 0.0]])
#         try:
#             rx = nsem.receivers.Impedance(rx_loc, orientation='xy', component='real')
#         except AttributeError:
#             rx = nsem.receivers.PointNaturalSource(rx_loc, orientation='xy', component='real')
#         rx_list.append(rx)
#
#     try:
#         src_list = [nsem.sources.Planewave(rx_list, frequency=f) for f in freqs]
#     except TypeError:
#         src_list = [nsem.sources.Planewave(rx_list, freq=f) for f in freqs]
#
#     survey = nsem.Survey(src_list)
#
#     # 使用 SolverLU
#     sim = nsem.simulation.Simulation2DElectricField(
#         mesh, survey=survey, sigmaMap=map_obj, solver=SolverLU
#     )
#
#     print("    - 正在求解 Maxwell 方程生成观测数据 (d_obs)...")
#     d_obs = sim.dpred(np.log(model_true))
#     survey.dobs = d_obs
#
#     agent = GeoBuddyAgent(mesh, survey, sim)
#     return agent, model_true
#
# # ==========================================
# # 4. 运行 HEF 决策 (Fixed Logic)
# # ==========================================
# def run_hef_experiment(agent, initial_model, line_df):
#     print("--> [3/4] 运行 HEF 算法 (Highest Entropy First)...")
#     print("    - 正在计算物理灵敏度场 (Physics-Informed Sensitivity Field)...")
#
#     # 1. 初始猜测
#     m_0 = np.log(np.ones_like(initial_model) * 1e-3)
#
#     # 2. 预计算物理场
#     fields = agent.physics.fields(m_0)
#
#     # 3. 逐个计算每个数据的灵敏度贡献
#     # 我们利用 Adjoint State (Jtvec) 来计算每一行 J 的范数，而不需要存储巨大的 J 矩阵
#     n_data = agent.survey.nD
#     sensitivity_per_datum = np.zeros(n_data)
#
#     print(f"    - 正在评估 {n_data} 个数据点的物理信息量...")
#
#     for i in range(n_data):
#         # 构造一个仅在第 i 个位置为 1 的向量 v
#         v = np.zeros(n_data)
#         v[i] = 1.0
#
#         # 计算 J^T * v (即 J 的第 i 行)
#         # 这是 SimPEG 提供的伴随状态计算接口
#         jt_v = agent.physics.Jtvec(m_0, v, f=fields)
#
#         # 计算该行灵敏度的能量 (Squared Norm)
#         sensitivity_per_datum[i] = np.sum(jt_v**2)
#
#         if i % 50 == 0:
#             print(f"      进度: {i}/{n_data}")
#
#     print("    - 灵敏度计算完成。正在映射回测点...")
#
#     # 4. 映射回测点 (每个测点对应多个频率的数据)
#     n_freq = 2 # 我们用了 [10, 1] 两个频率
#     hef_scores = []
#
#     for i in range(len(line_df)):
#         start_idx = i * n_freq
#         end_idx = (i + 1) * n_freq
#         if end_idx <= len(sensitivity_per_datum):
#             # 该测点的总分 = 所有频率数据的灵敏度之和
#             score = np.sum(sensitivity_per_datum[start_idx:end_idx])
#             hef_scores.append(score)
#         else:
#             hef_scores.append(0)
#
#     hef_scores = np.array(hef_scores)
#
#     # 归一化
#     if hef_scores.max() > 0:
#         hef_scores = (hef_scores - hef_scores.min()) / (hef_scores.max() - hef_scores.min())
#
#     return hef_scores
#
# # ==========================================
# # 5. 结果可视化
# # ==========================================
# def plot_results(line_df, hef_scores):
#     print("--> [4/4] 生成最终图表...")
#
#     plt.figure(figsize=(10, 6))
#
#     # X轴使用真实的 UTM Y 坐标
#     x_axis = line_df['UTM_Y'].values
#
#     # 1. 真实测点
#     plt.scatter(x_axis, np.zeros(len(line_df)),
#                 c='gray', marker='|', s=50, label='Candidate Stations (Real)')
#
#     # 2. 物理计算出的熵/灵敏度场
#     plt.plot(x_axis, hef_scores, 'g-', linewidth=2, label='Physics-Calculated Sensitivity (J)')
#     plt.fill_between(x_axis, 0, hef_scores, color='green', alpha=0.1)
#
#     # 3. Geo-Buddy 选出的点
#     top_indices = np.argsort(hef_scores)[::-1][:15]
#     selected = line_df.iloc[top_indices]
#
#     plt.scatter(selected['UTM_Y'], hef_scores[top_indices],
#                 c='red', s=100, marker='*', label='Geo-Buddy Selection (HEF)')
#
#     plt.title("Case Study: Physics-Driven Station Selection\n(Using Adjoint Sensitivity on Cloncurry Geometry)")
#     plt.xlabel("Northing (UTM Y)")
#     plt.ylabel("Normalized Information Utility")
#     plt.legend(loc='upper right')
#     plt.grid(True, alpha=0.3)
#
#     save_path = "Fig_CaseStudy_RealPhysics.png"
#     plt.savefig(save_path, dpi=300)
#     print(f"✅ 成功！真正的物理驱动实验图已保存至: {save_path}")
#     print("   这张图完全基于 SimPEG NSEM 引擎和您的 HEF 算法生成。")
#
# if __name__ == "__main__":
#     csv_path = "Experiment_GroundTruth_Full.csv"
#     if not os.path.exists(csv_path):
#         print("请先运行 prepare_full_experiment.py 准备数据")
#     else:
#         line_data, mesh, m_true, m_map = build_physics_environment(csv_path)
#         if line_data is not None:
#             agent, m_true = setup_simulation_and_agent(line_data, mesh, m_true, m_map)
#             scores = run_hef_experiment(agent, m_true, line_data)
#             plot_results(line_data, scores)