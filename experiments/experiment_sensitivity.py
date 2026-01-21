"""
Geo-Buddy: Autonomous Discovery of Subsurface Structures
Experiment 3: Sensitivity Analysis (REAL DATA GENERATION)

This script runs the ACTUAL mesh generation loop multiple times with
varying sensitivity thresholds to produce a physically grounded
sensitivity curve, consistent with Figure 2.
"""

import numpy as np
import matplotlib.pyplot as plt
from discretize import TreeMesh

class SensitivityExperiment:
    def __init__(self):
        # 基础配置 (必须与 Fig 2 保持一致)
        self.TOTAL_EXTENT = 32000.0
        self.MIN_CELL_SIZE = 31.25
        self.FAULT_X = 0.0

        # 基础规则 (Fig 2 的配置)
        self.BASE_RULES = [
            {'dist': 200,  'level_offset': 0},
            {'dist': 600,  'level_offset': 1},
            {'dist': 1500, 'level_offset': 2},
            {'dist': 3000, 'level_offset': 3},
            {'dist': 6000, 'level_offset': 4}
        ]

    def build_mesh_with_factor(self, factor):
        """
        根据 sensitivity factor 动态调整探测半径并生成网格。
        Factor = 1.0 对应 Figure 2 的配置。
        """
        n_cells = int(self.TOTAL_EXTENT / self.MIN_CELL_SIZE)
        h0 = self.MIN_CELL_SIZE
        mesh = TreeMesh([np.ones(n_cells)*h0, np.ones(n_cells)*h0], x0='CC', diagonal_balance=False)
        max_level = mesh.max_level

        # 动态调整规则：半径 * factor
        current_rules = []
        for r in self.BASE_RULES:
            current_rules.append({
                'dist': r['dist'] * factor, # 关键：缩放探测距离
                'level_offset': r['level_offset']
            })

        def refine_function(cell):
            dist = np.abs(cell.center[0] - self.FAULT_X)
            target_level = max_level - 5 # Background

            for rule in current_rules:
                if dist < rule['dist']:
                    prop_level = max_level - rule['level_offset']
                    if prop_level > target_level:
                        target_level = prop_level
            return target_level

        mesh.refine(refine_function)
        return mesh.nC

    def run_analysis(self):
        print(">>> [Step 3] Running REAL Sensitivity Analysis...")

        # 扫描 Sensitivity Factor (从 0.1倍 到 10倍)
        factors = np.logspace(np.log10(0.1), np.log10(10.0), 25)
        cell_counts = []

        print(f"    Simulating {len(factors)} different mesh configurations...")
        for i, f in enumerate(factors):
            nc = self.build_mesh_with_factor(f)
            cell_counts.append(nc)
            # 简单的进度条
            if i % 5 == 0: print(f"    Iter {i}: Factor={f:.2f} -> Cells={nc}")

        self.factors = factors
        self.cell_counts = np.array(cell_counts)

    def plot_figure_4(self):
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        # 1. 绘制真实数据曲线
        ax.semilogx(self.factors, self.cell_counts, 'o-', color='#2ca02c',
                   linewidth=2.5, markersize=6, label='Active Mesh Size')

        # 2. 标记 Figure 2 的选择 (Factor = 1.0)
        # 找到 Factor=1.0 对应的点
        idx_1 = np.abs(self.factors - 1.0).argmin()
        val_1 = self.cell_counts[idx_1]

        ax.plot(1.0, val_1, marker='*', color='gold', markersize=20,
               markeredgecolor='k', zorder=10, label='Our Setting (Fig.2)')

        # 3. 划分区域 (语义化)
        # Robust Zone: 比如 Factor 0.5 到 2.0 之间，网格数变化是可以接受的
        ax.axvspan(0.5, 2.5, color='green', alpha=0.1, label='Robust Operating Zone')

        # 添加文字注释
        ax.text(0.15, val_1 * 0.5, "Under-Refinement\n(Risk of Missing Fault)",
               color='red', fontsize=10, fontstyle='italic')
        ax.text(4.0, val_1 * 1.5, "Over-Refinement\n(Wasted Compute)",
               color='gray', fontsize=10, fontstyle='italic')

        # 装饰
        ax.set_xlabel(r"Sensitivity Parameter $\sigma$ (Normalized)", fontsize=13)
        ax.set_ylabel("Total Active Cells", fontsize=13)
        ax.set_title("Sensitivity Analysis: Robustness of Parameter Selection", fontsize=15, pad=15, weight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(loc='upper left', frameon=True, shadow=True)

        # 调整 Y 轴为对数坐标可能更好看，取决于数据跨度
        # 如果变化很大 (几千到几十万)，打开这一行：
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig("Fig4_Sensitivity_Real.png", dpi=300)
        print(f"[Output] Saved Fig4. Optimal Cells: {val_1}")
        plt.show()

if __name__ == "__main__":
    exp = SensitivityExperiment()
    exp.run_analysis()
    exp.plot_figure_4()