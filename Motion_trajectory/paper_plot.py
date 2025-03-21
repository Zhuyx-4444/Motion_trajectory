import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.interpolate import griddata

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用系统已安装的Arial字体
plt.rcParams['axes.unicode_minus'] = False  # 保留负号显示

class EnhancedTrajectoryComparator:
    def __init__(self,
                 human_data_folder: str,
                 human_score_path: str,
                 reference_path: str,
                 simulated_folder: str,
                 simulated_score_path: str):
        """
        Trajectory comparison analyzer with multiple data sources

        :param human_data_folder: Path to human trajectory folder (Human_trajectory)
        :param human_score_path: Path to human trajectory scores CSV
        :param reference_path: Path to reference trajectory CSV
        :param simulated_folder: Path to simulated trajectories folder
        :param simulated_score_path: Path to simulated trajectory scores CSV
        """
        # Initialize path configurations
        self.human_folder = human_data_folder
        self.human_scores = pd.read_csv(human_score_path)
        self.reference_path = reference_path
        self.simulated_folder = simulated_folder
        self.simulated_scores = pd.read_csv(simulated_score_path)

        # Data containers
        self.human_trajectories = []
        self.simulated_trajectories = []
        self.reference_trajectory = None

        # Visualization settings
        self.cmap = LinearSegmentedColormap.from_list('score_color', ['red', 'green'])
        self.color_config = {
            'human': {'color': 'blue', 'label': 'Human Trajectories'},
            'simulated': {'color': 'orange', 'label': 'Generated Trajectories'},
            'reference': {'color': 'green', 'label': 'Reference Trajectory'},
            'pedestrian': {'color': 'black', 'label': 'Stationary Vehicles'},
            'ahead': {'color': 'purple', 'label': 'Leading Vehicles'}
        }

    def load_all_data(self):
        """Load all data sources"""
        self._load_reference_trajectory()
        self._load_human_trajectories()
        self._load_simulated_trajectories()

    def _load_reference_trajectory(self):
        """Load reference trajectory data"""
        if not os.path.exists(self.reference_path):
            raise FileNotFoundError(f"Reference trajectory file missing: {self.reference_path}")

        try:
            df = pd.read_csv(self.reference_path)
            self.reference_trajectory = {
                'source': 'reference',
                'ego': {
                    'x': df['position_x'],
                    'y': df['position_y']
                }
            }
        except Exception as e:
            print(f"Error loading reference trajectory from {self.reference_path}: {str(e)}")
            print(f"Columns in CSV: {df.columns.tolist()}")
            raise

    def _load_human_trajectories(self):
        """Process human trajectory files"""
        scored_files = self.human_scores['filename'].tolist()
        for filename in os.listdir(self.human_folder):
            if filename.startswith('processed_') and filename.endswith('.csv'):
                base_name = self._extract_base_name(filename)
                if base_name in scored_files:
                    self._add_trajectory(os.path.join(self.human_folder, filename), 
                                       self.human_scores[self.human_scores['filename'] == base_name].iloc[0],
                                       'human')
                else:
                    print(f"警告: 文件 {filename} (标识:{base_name}) 无对应评分，建议检查评分生成逻辑")

    def _load_simulated_trajectories(self):
        """Process simulated trajectory files"""
        scored_files = self.simulated_scores['file'].tolist()
        for filename in os.listdir(self.simulated_folder):
            if filename.startswith('processed_simulated_trajectory_') and filename.endswith('.csv'):
                base_name = filename.replace('processed_', '', 1).replace('.csv', '')
                if base_name in scored_files:
                    score_record = self.simulated_scores[self.simulated_scores['file'] == base_name].iloc[0]
                    self._add_trajectory(
                        os.path.join(self.simulated_folder, filename),
                        score_record,
                        'simulated'
                    )
                else:
                    print(f"Warning: Unscored simulated trajectory {filename}")

    def _add_trajectory(self, file_path: str, score_record: pd.Series, source_type: str):
        """Add trajectory data to container"""
        try:
            data = pd.read_csv(file_path)
            if data.empty:
                print(f"文件为空: {file_path}")
                return

            # 仅对 Human 数据计算旋转角度
            rotation_angle = self._calculate_rotation_angle(data) if source_type == 'human' else 0.0

            trajectory = {
                'source': source_type,
                'score': score_record['average'],
                'ego': self._extract_ego_data(data, rotation_angle, source_type),
                'ahead': self._extract_vehicle_data(data, 1, rotation_angle, source_type),
                'pedestrian1': self._extract_vehicle_data(data, 2, rotation_angle, source_type),
                'pedestrian2': self._extract_vehicle_data(data, 3, rotation_angle, source_type)
            }

            if source_type == 'human':
                self.human_trajectories.append(trajectory)
            else:
                self.simulated_trajectories.append(trajectory)

        except Exception as e:
            print(f"Failed to load {file_path}: {str(e)}")

    @staticmethod
    def _extract_base_name(filename: str, prefix: str = 'processed_') -> str:
        """
        Extract base filename
        :param filename: Original filename (e.g. processed_human_001.csv)
        :param prefix: Prefix to remove (e.g. 'processed_')
        :return: Base name (e.g. human_001)
        """
        base = filename.replace(prefix, '').replace('csv', '')
        if '_' in base and base.count('_') > 1:
            parts = base.split('_')
            return '_'.join(parts[1:])  # 保留编号部分
        return base

    def _calculate_rotation_angle(self, data: pd.DataFrame) -> float:
        """Calculate rotation angle using leading vehicle trajectory (仅用于 Human 数据)"""
        if 'vehicle_1_x' in data.columns and 'vehicle_1_y' in data.columns:
            vehicle_data = data[['vehicle_1_x', 'vehicle_1_y']].head(50).dropna()
            if len(vehicle_data) >= 2:
                n_points = max(2, int(len(vehicle_data) * 0.05))
                start_points = vehicle_data.iloc[:n_points].mean().values
                end_points = vehicle_data.iloc[-n_points:].mean().values
                direction = end_points - start_points
                return -np.arctan2(direction[1], direction[0])
        return 0.0

    def _extract_ego_data(self, data: pd.DataFrame, angle: float, source_type: str) -> dict:
        """Extract ego vehicle trajectory"""
        if source_type == 'human':
            x_col, y_col = 'ego_x', 'ego_y'
        else:
            x_col, y_col = 'position_x', 'position_y'

        if x_col not in data.columns or y_col not in data.columns:
            return {'x': [], 'y': []}

        x = data[x_col] - data[x_col].iloc[0]
        y = data[y_col] - data[y_col].iloc[0]

        # 仅对 Human 数据应用旋转
        if source_type == 'human':
            rotated_x, rotated_y = self._rotate_coordinates(x, y, angle)
            return {'x': rotated_x, 'y': -rotated_y}
        else:
            # 对模拟轨迹进行平移和调整
            x = x - 20  # 横坐标整体左移20米
            y = y - y.iloc[0]  # 确保y坐标在原点处为0
            return {'x': x, 'y': y}
            # return {'x': x, 'y': y}

    def _extract_vehicle_data(self, data: pd.DataFrame, vehicle_id: int, angle: float, source_type: str) -> dict:
        """Extract surrounding vehicle data (仅用于 Human 数据)"""
        if source_type != 'human':
            return {'x': [], 'y': []}  # 仅处理 human 数据

        x_col = f'vehicle_{vehicle_id}_x'
        y_col = f'vehicle_{vehicle_id}_y'

        if x_col not in data.columns or y_col not in data.columns:
            return {'x': [], 'y': []}

        x = data[x_col] - data['ego_x'].iloc[0]
        y = data[y_col] - data['ego_y'].iloc[0]

        # 仅对 Human 数据应用旋转
        rotated_x, rotated_y = self._rotate_coordinates(x, y, angle)
        return {'x': rotated_x, 'y': -rotated_y}

    @staticmethod
    def _rotate_coordinates(x: pd.Series, y: pd.Series, angle: float) -> tuple:
        """Coordinate rotation function"""
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        rotated = np.dot(rotation_matrix, np.vstack([x, y]))
        return rotated[0, :], rotated[1, :]

    def generate_comparison_figure(self, comparison_type: str = 'human_vs_reference') -> Figure:
        """
        Generate comparison visualization
        :param comparison_type: Comparison mode
            - 'human_vs_reference': Human vs reference trajectories
            - 'simulated_vs_reference': Simulated vs reference trajectories
        """
        fig = Figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

        # Plot reference trajectory
        self._plot_reference(ax)

        # Add trajectories based on comparison type
        if comparison_type == 'human_vs_reference':
            self._plot_trajectories(ax, self.human_trajectories, 'human')
        elif comparison_type == 'simulated_vs_reference':
            self._plot_trajectories(ax, self.simulated_trajectories, 'simulated')

        # Add surrounding vehicles
        self._plot_surrounding_vehicles(ax)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        # Configure figure elements
        self._configure_axes(ax)
        self._add_colorbar(ax)
        return fig

    def _plot_reference(self, ax):
        """Plot reference trajectory"""
        if self.reference_trajectory is None:
            raise ValueError("参考轨迹数据未加载，请检查数据源！")
        ref = self.reference_trajectory
        try:
            ax.plot(ref['ego']['x'], ref['ego']['y'],
                    linestyle='--', linewidth=2,
                    color=self.color_config['reference']['color'],
                    label=self.color_config['reference']['label'])
        except KeyError as e:
            raise KeyError(f"参考轨迹数据结构错误，缺少关键键: {e}。请检查数据格式。")
            
    def _plot_trajectories(self, ax, trajectories: list, source_type: str):
        """Plot specific type of trajectories"""
        config = self.color_config[source_type]
        for traj in trajectories:
            # 过滤 x > 0 的部分
            x = np.asarray(traj['ego']['x'])
            y = np.asarray(traj['ego']['y'])
            mask = x > 0  # 只保留 x > 0 的部分
            x_filtered = x[mask]
            y_filtered = y[mask]
    
            if len(x_filtered) == 0:
                continue  # 如果没有数据，跳过绘制
    
            norm_score = traj['score'] / 100  # 假设分数范围0-100
            color = self.cmap(norm_score)
            line = ax.plot(x_filtered, y_filtered,
                          color=color,
                          alpha=0.6,
                          linewidth=1,
                          label=None)  # 不设置标签
    
            # Add score annotation
            if len(x_filtered) > 0:  # 确保有数据点
                ax.annotate(
                    f"{traj['score']:.1f}",
                    xy=(x_filtered[-1], y_filtered[-1]),  # 使用过滤后的最后一个点
                    xytext=(15, 10),
                    textcoords='offset points',
                    color=line[0].get_color(),
                    fontsize=8,
                    arrowprops=dict(
                        arrowstyle="->",
                        color=line[0].get_color(),
                        connectionstyle="arc3,rad=0.3"
                    )
                )
        # 添加代理句柄
        ax.plot([], [],
                color=config['color'],
                label=config['label'],
                alpha=0.6,
                linewidth=1)
        
    def _plot_surrounding_vehicles(self, ax):
        """Plot average surrounding vehicle trajectories"""
        # 计算 human 数据的平均轨迹
        ahead_avg = self._calculate_average_trajectory(self.human_trajectories, 'ahead')
        pedestrian1_avg = self._calculate_average_trajectory(self.human_trajectories, 'pedestrian1')
        pedestrian2_avg = self._calculate_average_trajectory(self.human_trajectories, 'pedestrian2')

        # 绘制前车
        if ahead_avg:
            ax.plot(ahead_avg[0], ahead_avg[1],
                    color=self.color_config['ahead']['color'],
                    linestyle=':',
                    label=self.color_config['ahead']['label'])

        # 绘制静止车辆
        if pedestrian1_avg:
            ax.plot(pedestrian1_avg[0], pedestrian1_avg[1],
                    color=self.color_config['pedestrian']['color'],
                    linestyle=':',
                    label=self.color_config['pedestrian']['label'])

        if pedestrian2_avg:
            ax.plot(pedestrian2_avg[0], pedestrian2_avg[1],
                    color=self.color_config['pedestrian']['color'],
                    linestyle=':',
                    label=self.color_config['pedestrian']['label'])

    def _calculate_average_trajectory(self, trajectories: list, vehicle_type: str) -> tuple:
        """Calculate average trajectory for vehicle type"""
        valid_trajs = []
        for traj in trajectories:
            if len(traj[vehicle_type]['x']) > 0:
                valid_trajs.append({
                    'x': np.array(traj[vehicle_type]['x']),
                    'y': np.array(traj[vehicle_type]['y'])
                })

        if not valid_trajs:
            return None

        # Align trajectory lengths
        max_length = max(len(t['x']) for t in valid_trajs)
        aligned_x, aligned_y = [], []

        for traj in valid_trajs:
            x_interp = np.interp(
                np.linspace(0, 1, max_length),
                np.linspace(0, 1, len(traj['x'])),
                traj['x']
            )
            y_interp = np.interp(
                np.linspace(0, 1, max_length),
                np.linspace(0, 1, len(traj['y'])),
                traj['y']
            )
            aligned_x.append(x_interp)
            aligned_y.append(y_interp)

        return np.mean(aligned_x, axis=0), np.mean(aligned_y, axis=0)

    def _configure_axes(self, ax):
        """Configure axes settings"""
        ax.set_xlabel('Longitudinal Position (m)')
        ax.set_ylabel('Lateral Position (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    def _add_colorbar(self, ax):
        """Add color scale bar"""
        sm = ScalarMappable(cmap=self.cmap, norm=plt.Normalize(0, 100))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Safety Score (%)', pad=0.02)


    def generate_heatmap_figure(self) -> Figure:
        """生成模拟轨迹安全得分的熱力图"""
        x_coords = []
        y_coords = []
        scores = []
        
        # 收集所有模拟轨迹点及其得分
        for traj in self.simulated_trajectories:
            x = np.asarray(traj['ego']['x'])
            y = np.asarray(traj['ego']['y'])
            mask = x > 0  # 与原图一致过滤x>0
            x_filtered = x[mask]
            y_filtered = y[mask]
            if len(x_filtered) == 0:
                continue
            x_coords.extend(x_filtered)
            y_coords.extend(y_filtered)
            scores.extend([traj['score']] * len(x_filtered))
        
        if not x_coords:
            raise ValueError("No simulated trajectory data available for heatmap.")
        
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        scores = np.array(scores)
        
        # 确定网格范围
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # 处理单点情况
        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0
        

        # 根据数据点数量动态选择grid_size

        grid_size = 0.5  # 低密度数据用0.5米
        x_bins =  np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)
        
        # 计算二维直方图
        sum_scores, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=(x_bins, y_bins), weights=scores
        )
        counts, _, _ = np.histogram2d(x_coords, y_coords, bins=(x_bins, y_bins))
        
        # 计算平均得分
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_scores = sum_scores / counts
            avg_scores[counts == 0] = np.nan
        
        # 创建新的白-绿色彩映射
        heatmap_cmap = LinearSegmentedColormap.from_list('heatmap_color', ['white', 'green'])
        
        # 创建图像
        fig = Figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

        # 绘制热力图
        heatmap = ax.pcolormesh(xedges, yedges, avg_scores.T,
                               cmap=heatmap_cmap, vmin=0, vmax=100, 
                               shading='auto', zorder=1)
        
        # 叠加所有模拟轨迹（灰色半透明）
        for traj in self.simulated_trajectories:
            x = np.asarray(traj['ego']['x'])
            y = np.asarray(traj['ego']['y'])
            mask = x > 0
            x_filtered = x[mask]
            y_filtered = y[mask]
            
            if len(x_filtered) > 1:  # 需要至少两个点画线
                ax.plot(x_filtered, y_filtered,
                        color='gray',
                        alpha=0.3,  # 半透明
                        linewidth=0.8,
                        zorder=2)  # 轨迹在热力图层上方
        
        # 添加颜色条和标签
        fig.colorbar(heatmap, ax=ax, label='Average Safety Score (%)')
        ax.set(xlabel='Longitudinal Position (m)',
              ylabel='Lateral Position (m)',
              title='Simulated Trajectories Heatmap with Score Distribution')
        ax.grid(True, alpha=0.3)
        
        return fig




