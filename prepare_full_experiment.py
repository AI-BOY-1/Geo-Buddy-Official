import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

# ==========================================
# 脚本名称: prepare_full_experiment.py
# 功能: 准备全量实验环境 (521个测点)
# 位置: 请放在 Geo-Buddy-Official 根目录下运行
# ==========================================

# 1. 读取之前清洗生成的总表
csv_path = "Cloncurry_GroundTruth.csv"
if not os.path.exists(csv_path):
    print("错误：找不到 Cloncurry_GroundTruth.csv，请先运行 data clean.py。")
    exit()

df = pd.read_csv(csv_path)
print(f"--> 读取成功！全量测点数: {len(df)}")

# 2. 可视化确认（全景图）
plt.figure(figsize=(8, 12))  # 调整为长方形，适应测区形状
plt.scatter(df["UTM_X"], df["UTM_Y"], c='k', s=10, marker='.', label='Candidate Stations')
plt.title(f"Full Experimental Environment\n({len(df)} Real-world Stations)")
plt.xlabel("UTM X (m)")
plt.ylabel("UTM Y (m)")
plt.legend()
plt.grid(True, alpha=0.5)
plt.axis('equal')  # 保持真实比例
plt.tight_layout()
plt.savefig("Full_Survey_Area.png")
print("--> 全景分布图已保存为 Full_Survey_Area.png")
# plt.show() # 如果在远程运行可注释掉

# 3. 准备实验数据文件夹
# 我们创建一个新文件夹，专门存放这 521 个实验用 EDI 文件
experiment_dir = "Experiment_Data_Full"
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
    print(f"--> 创建文件夹: {experiment_dir}")
else:
    print(f"--> 文件夹已存在: {experiment_dir}")

print(f"\n--> 正在将 521 个 EDI 文件复制到实验目录... (请稍候)")

# 4. 复制文件并生成实验索引表
df_experiment = df.copy()
success_count = 0

for idx, row in df_experiment.iterrows():
    src_path = row["File"]
    fname = os.path.basename(src_path)
    dst_path = os.path.join(experiment_dir, fname)

    try:
        shutil.copy2(src_path, dst_path)
        # 更新表格中的路径为新位置（使用绝对路径确保 Agent 读取不出错）
        df_experiment.at[idx, "File"] = os.path.abspath(dst_path)
        success_count += 1
    except Exception as e:
        print(f"   [!] 复制失败 {fname}: {e}")

# 5. 保存最终的“环境配置文件”
output_csv = "Experiment_GroundTruth_Full.csv"
df_experiment.to_csv(output_csv, index=False)

print("-" * 40)
print(f"✅ 全量实验环境部署完成！")
print(f"   - 成功迁移文件: {success_count} / {len(df)}")
print(f"   - 实验数据库: {output_csv}")
print(f"   - 数据文件夹: {experiment_dir}")
print("-" * 40)
print("💡 接下来，您的 Geo-Buddy Agent 将读取这个 CSV 作为'地图'。")