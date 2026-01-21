import os
import glob
import pandas as pd
import numpy as np
from pyproj import Proj


def parse_edi_coordinates(file_path):
    """
    解析单个 EDI 文件，提取 LAT, LONG, ELEV。
    支持格式：-20:26:45.96 (DMS) 或 -20.4461 (DD)
    """
    lat_dd = None
    lon_dd = None
    elev = 0.0

    with open(file_path, 'r', encoding='latin-1') as f:
        # 读取所有行（EDI通常不大，直接读内存没问题）
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # 1. 优先匹配 >HEAD 块中的 LAT/LONG
        if line.startswith("LAT="):
            lat_str = line.split("=")[1]
            lat_dd = dms_to_dd(lat_str)
        elif line.startswith("LONG="):
            lon_str = line.split("=")[1]
            lon_dd = dms_to_dd(lon_str)
        elif line.startswith("ELEV="):
            try:
                elev = float(line.split("=")[1])
            except:
                pass

        # 2. 如果 >HEAD 没找到，尝试找 REFLAT (通常在 >=DEFINEMEAS)
        if lat_dd is None and "REFLAT=" in line:
            parts = line.split("REFLAT=")
            if len(parts) > 1:
                lat_dd = dms_to_dd(parts[1].split()[0])  # 这种格式可能带其他字符

        if lon_dd is None and "REFLONG=" in line:
            parts = line.split("REFLONG=")
            if len(parts) > 1:
                lon_dd = dms_to_dd(parts[1].split()[0])

        # 如果都找到了，可以提前退出循环（优化速度）
        if lat_dd is not None and lon_dd is not None:
            break

    return lat_dd, lon_dd, elev


def dms_to_dd(dms_str):
    """
    将 -20:26:45.96 转换为 -20.4461
    """
    try:
        dms_str = dms_str.replace('"', '').strip()  # 去掉可能存在的引号
        if ":" in dms_str:
            parts = dms_str.split(":")
            d = float(parts[0])
            m = float(parts[1]) if len(parts) > 1 else 0.0
            s = float(parts[2]) if len(parts) > 2 else 0.0

            # 处理符号：如果度数是负的，后面分秒也要按负处理
            sign = -1 if (d < 0 or dms_str.startswith('-')) else 1
            dd = abs(d) + (abs(m) / 60.0) + (abs(s) / 3600.0)
            return dd * sign
        else:
            # 纯数字格式，直接转
            return float(dms_str)
    except Exception as e:
        return None


def main(root_folder):
    # 路径拼接
    target_folder = os.path.join(root_folder, "edis", "Renamed impedance")
    if not os.path.exists(target_folder):
        # 如果找不到，尝试直接搜索当前目录下的edi
        target_folder = root_folder
        print(f"提示: 未找到标准目录结构，正在扫描: {target_folder}")

    edi_files = glob.glob(os.path.join(target_folder, "*.edi"))
    print(f"找到 {len(edi_files)} 个 EDI 文件，开始解析...")

    # 定义投影 (Cloncurry -> UTM Zone 54S)
    myProj = Proj("+proj=utm +zone=54 +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    data_list = []

    for file_path in edi_files:
        fname = os.path.basename(file_path)
        lat, lon, elev = parse_edi_coordinates(file_path)

        if lat is not None and lon is not None:
            # 转 UTM
            utm_x, utm_y = myProj(lon, lat)

            data_list.append({
                "Station": fname.replace(".edi", ""),
                "Lat": lat,
                "Lon": lon,
                "UTM_X": utm_x,
                "UTM_Y": utm_y,
                "Elev": elev,
                "File": file_path
            })
        else:
            print(f"警告: 无法解析坐标 -> {fname}")

    # 保存
    if len(data_list) > 0:
        df = pd.DataFrame(data_list)
        df.to_csv("Cloncurry_GroundTruth.csv", index=False)
        print("-" * 30)
        print(f"成功！已生成 'Cloncurry_GroundTruth.csv'，共 {len(df)} 个有效站点。")
        print(df.head())

        # 简单画个图确认分布
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 8))
            plt.scatter(df["UTM_X"], df["UTM_Y"], s=5, c='k', marker='.')
            plt.title("Cloncurry MT Station Distribution")
            plt.xlabel("UTM X (m)")
            plt.ylabel("UTM Y (m)")
            plt.axis('equal')
            plt.grid(True)
            plt.show()
            print("分布图已显示，请确认网格形状。")
        except ImportError:
            print("提示: 未安装 matplotlib，跳过绘图。")
    else:
        print("错误: 没有提取到任何数据。")


# --- 运行 ---
# 请将下面的路径修改为您解压后的文件夹路径
# 例如: G:/博士所有材料/.../
folder_path = r"G:\博士所有材料\电磁ai\第十六篇开创性工作天马行空\Geo-Buddy-Official"
main(folder_path)