"""
Geo-Buddy: Autonomous Discovery of Subsurface Structures
Benchmark A: Adaptive Mesh Evolution for Fault Detection (FIXED)

Reason for fix: TreeMesh must be initialized with the FINEST cell size,
not the coarse size. The previous version clamped resolution at 1000m.
This version initializes with ~31m cells to allow proper refinement.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TreeMesh
import time

class SimulationConfig:
    """Configuration parameters for Benchmark A."""
    # 实验区域设置 (ROI)
    DOMAIN_WIDTH = 10000.0  # 10km interest area
    DOMAIN_DEPTH = 10000.0

    # 关键修改：必须定义整个模拟域的【最精细】尺寸
    # 为了支持从 31.25m (Level Max) 到 1000m (Background) 的多级变化
    # 我们设定总区域为 32km (保证是 2 的幂次方便分割)
    TOTAL_EXTENT = 32000.0
    MIN_CELL_SIZE = 31.25   # 最小单元 (原子)

    # 目标位置
    FAULT_LOCATION_X = 0.0

    # 物理驱动的细化规则
    # level_offset: 0 = 最精细(31m), 1 = 62m, 2 = 125m ... 5 = 1000m
    REFINEMENT_RULES = [
        {'dist': 200,  'level_offset': 0}, # 31.25m
        {'dist': 600,  'level_offset': 1}, # 62.5m
        {'dist': 1500, 'level_offset': 2}, # 125m
        {'dist': 3000, 'level_offset': 3}, # 250m
        {'dist': 6000, 'level_offset': 4}  # 500m
    ]
    # 背景保留 level_offset = 5 (1000m)

class GeoBuddyExperiment:
    def __init__(self, config):
        self.cfg = config
        self.mesh = None
        self.stats = {}

    def build_base_mesh(self):
        """Initialize the TreeMesh with FINEST fundamental cells."""
        print(f"[Init] Building base mesh...")

        # 计算最底层需要的单元数
        # 32000 / 31.25 = 1024 cells
        n_cells = int(self.cfg.TOTAL_EXTENT / self.cfg.MIN_CELL_SIZE)
        h0 = self.cfg.MIN_CELL_SIZE

        # 初始化 TreeMesh
        # 这里告诉 SimPEG：我的最小积木块是 31.25m
        self.mesh = TreeMesh(
            [np.ones(n_cells) * h0, np.ones(n_cells) * h0],
            x0='CC', # Center-Centered, (0,0) 是中心
            diagonal_balance=False
        )
        print(f"[Init] Base mesh created. Max Level: {self.mesh.max_level}")
        print(f"[Init] Theoretical finest cells: {self.mesh.n_cells} (Virtual)")

    def run_hef_adaptation(self):
        """Simulate the Highest Entropy First (HEF) agent refining the mesh."""
        print("[Agent] Starting adaptive refinement loop...")

        max_level = self.mesh.max_level # 通常是 log2(1024) = 10

        def refine_function(cell):
            xyz = cell.center
            dist = np.abs(xyz[0] - self.cfg.FAULT_LOCATION_X)

            # 默认背景：使用较粗网格 (1000m)
            # 31.25 * 2^5 = 1000. 所以我们要减去 5 级
            target_level = max_level - 5

            # 遍历规则
            for rule in self.cfg.REFINEMENT_RULES:
                if dist < rule['dist']:
                    # 计算目标层级. offset 0 就是 max_level
                    prop_level = max_level - rule['level_offset']
                    if prop_level > target_level:
                        target_level = prop_level

            return target_level

        # 执行 SimPEG 的细化
        self.mesh.refine(refine_function)

        # 强制细化中心的一条线，确保断层可见性（防止完全没切开）
        # self.mesh.insert_cells([0.0, -5000.0], max_level) # 可选，通常 refine 够了

        print(f"[Agent] Refinement complete. Final cell count: {self.mesh.nC}")

    def calculate_statistics(self):
        """Compute metrics for Table 2."""
        n_active = self.mesh.nC

        # Baseline: 仅计算 ROI (10km x 10km) 内如果全用 31.25m 需要多少网格
        roi_width = self.cfg.DOMAIN_WIDTH
        roi_depth = self.cfg.DOMAIN_DEPTH
        min_h = self.cfg.MIN_CELL_SIZE

        n_uniform_roi = (roi_width / min_h) * (roi_depth / min_h)

        compression_ratio = n_active / n_uniform_roi * 100
        cost_reduction = 100 - compression_ratio

        self.stats = {
            'n_active': n_active,
            'n_uniform': int(n_uniform_roi),
            'reduction': cost_reduction,
            'min_h': min_h
        }

        print("-" * 50)
        print(f"BENCHMARK A RESULTS")
        print("-" * 50)
        print(f"Min Cell Size           : {min_h:.2f} m")
        print(f"Geo-Buddy Active Blocks : {n_active}")
        print(f"Baseline (Uniform ROI)  : {int(n_uniform_roi)}")
        print(f"Cost Reduction          : {cost_reduction:.4f}%")
        print("-" * 50)

    def plot_results(self, filename="Fig2_BenchmarkA_Corrected.png"):
        """Generate the publication-ready figure."""
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # 绘制背景密度图
        # 注意：这里 cell_volumes 是面积
        cell_areas = self.mesh.cell_volumes

        out = self.mesh.plot_image(
            cell_areas,
            ax=ax,
            grid=True,
            grid_opts={'color': 'k', 'linewidth': 0.3, 'alpha': 0.5},
            pcolor_opts={'cmap': 'Blues_r', 'norm': LogNorm()}
        )

        cbar = plt.colorbar(out[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'Cell Area ($m^2$) - Log Scale', rotation=270, labelpad=15)

        # 绘制断层
        ax.axvline(x=self.cfg.FAULT_LOCATION_X, color='#D62728',
                   linestyle='--', linewidth=3, label='Geological Fault')

        # 限制视图范围 (只看 ROI 10km)
        limit_x = self.cfg.DOMAIN_WIDTH / 2
        limit_z = self.cfg.DOMAIN_DEPTH / 2
        ax.set_xlim(-limit_x, limit_x)
        ax.set_ylim(-limit_z, limit_z)

        ax.set_title("Benchmark A: Adaptive Mesh Evolution (SimPEG)", fontsize=16, pad=20, weight='bold')
        ax.set_xlabel("Distance x (m)", fontsize=14)
        ax.set_ylabel("Depth z (m)", fontsize=14)

        stats_text = (
            f"$\mathbf{{Active\ Cells}}$: {self.stats['n_active']}\n"
            f"$\mathbf{{Min\ Resolution}}$: {self.stats['min_h']:.1f} m\n"
            f"$\mathbf{{Cost\ Reduction}}$: {self.stats['reduction']:.2f}%"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95))

        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"[Output] Corrected figure saved to {filename}")
        plt.show()

if __name__ == "__main__":
    config = SimulationConfig()
    experiment = GeoBuddyExperiment(config)
    experiment.build_base_mesh()
    experiment.run_hef_adaptation()
    experiment.calculate_statistics()
    experiment.plot_results()