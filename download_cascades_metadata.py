"""
Script: Download USArray Station Metadata (Cascades Region)
Source: IRIS FDSN Station Service (http://service.iris.edu)
Purpose: Validate real-world topography for Geo-Buddy stress test.
"""

import requests
import pandas as pd
import io


def fetch_real_station_data():
    print(">>> 连接 IRIS 地球物理数据中心 (DMC)...")

    # IRIS FDSN Station API URL
    url = "http://service.iris.edu/fdsnws/station/1/query"

    # 查询参数：
    # net=TA (Transportable Array, 即 USArray)
    # cha=LHE,LHN,LHZ (长周期电磁/地震通道，确保是MT站)
    # minlat/maxlat/minlon/maxlon: 框选 Cascades 火山带 (华盛顿州/俄勒冈州)
    params = {
        "net": "TA",
        "level": "station",
        "format": "text",  # 下载为文本格式
        "minlat": 45.0,  # 俄勒冈北部
        "maxlat": 47.0,  # 华盛顿南部 (Mt St Helens 附近)
        "minlon": -123.0,
        "maxlon": -120.0,
        "includecomments": "false"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # 解析数据
        # 数据格式: Network | Station | Latitude | Longitude | Elevation | ...
        col_names = ["Network", "Station", "Latitude", "Longitude", "Elevation", "SiteName", "StartTime", "EndTime"]
        df = pd.read_csv(io.StringIO(response.text), sep="|", header=None, names=col_names, skiprows=1)  # 跳过第一行header

        print(f"✅ 成功获取 {len(df)} 个真实测站数据！")

        # 筛选并排序 (按经度或纬度排序形成一条测线)
        # 我们取一条横穿山脉的剖面 (按经度排序)
        line_profile = df.sort_values("Longitude")

        # 保存为 CSV 证据
        output_file = "Real_Cascades_Stations.csv"
        line_profile.to_csv(output_file, index=False)
        print(f"✅ 数据已保存至: {output_file}")

        # 打印部分数据供查看
        print("\n--- 真实地形数据预览 (Top 5) ---")
        print(line_profile[["Station", "Latitude", "Longitude", "Elevation"]].head())
        print(f"    最大高程: {line_profile['Elevation'].max()} m")
        print(f"    最小高程: {line_profile['Elevation'].min()} m")
        print(
            f"    地势高差: {line_profile['Elevation'].max() - line_profile['Elevation'].min()} m (这就是我们需要的崎岖地形！)")

        return line_profile

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


if __name__ == "__main__":
    fetch_real_station_data()