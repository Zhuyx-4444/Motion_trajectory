import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese font configuration
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display


class EnhancedTrajectoryComparator:
    def __init__(self,
                 human_data_folder: str,
                 human_score_path: str,
                 reference_path: str,
                 reference_score_path: str,
                 simulated_folder: str,
                 simulated_score_path: str):
        """
        Trajectory comparison analyzer with multiple data sources

        :param human_data_folder: Path to human trajectory folder (Human_trajectory)
        :param human_score_path: Path to human trajectory scores CSV
        :param reference_path: Path to reference trajectory CSV
        :param reference_score_path: Path to reference trajectory score CSV
        :param simulated_folder: Path to simulated trajectories folder
        :param simulated_score_path: Path to simulated trajectory scores CSV
        """
        # Initialize path configurations
        self.human_folder = human_data_folder
        self.human_scores = pd.read_csv(human_score_path)
        self.reference_path = reference_path
        self.ref_score = pd.read_csv(reference_score_path).iloc[0]['average']
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
            'pedestrian': {'color': 'gray', 'label': 'Stationary Vehicles'},
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

        df = pd.read_csv(self.reference_path)
        self.reference_trajectory = {
            'source': 'reference',
            'score': self.ref_score,
            'ego': {'x': df['position_x'], 'y': df['position_y']},
            'pedestrian': self._extract_vehicle_data(df, 2),
            'ahead': self._extract_vehicle_data(df, 1)
        }

    def _load_human_trajectories(self):
        """Process human trajectory files"""
        for filename in os.listdir(self.human_folder):
            if filename.startswith('processed_') and filename.endswith('.csv'):
                file_path = os.path.join(self.human_folder, filename)
                base_name = self._extract_base_name(filename, 'processed_')

                # Match with score records
                score_record = self.human_scores[self.human_scores['filename'] == base_name]
                if not score_record.empty:
                    self._add_trajectory(file_path, score_record.iloc[0], 'human')
                else:
                    print(f"Warning: Skipping unscored human trajectory {filename}")

    def _load_simulated_trajectories(self):
        """Process simulated trajectory files"""
        for filename in os.listdir(self.simulated_folder):
            if filename.startswith('simulated_trajectory_') and filename.endswith('.csv'):
                file_path = os.path.join(self.simulated_folder, filename)
                base_name = self._extract_base_name(filename, prefix='')

                # Match with score records
                score_record = self.simulated_scores[self.simulated_scores['filename'] == base_name]
                if not score_record.empty:
                    self._add_trajectory(file_path, score_record.iloc[0], 'simulated')
                else:
                    print(f"Warning: Skipping unscored simulated trajectory {filename}")

    def _add_trajectory(self, file_path: str, score_record: pd.Series, source_type: str):
        """Add trajectory data to container"""
        try:
            data = pd.read_csv(file_path)
            rotation_angle = self._calculate_rotation_angle(data)

            trajectory = {
                'source': source_type,
                'score': score_record['average'],
                'ego': self._extract_ego_data(data, rotation_angle),
                'pedestrian': self._extract_vehicle_data(data, 2, rotation_angle),
                'ahead': self._extract_vehicle_data(data, 1, rotation_angle)
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
        return filename.replace(prefix, '').replace('.csv', '')

    def _calculate_rotation_angle(self, data: pd.DataFrame) -> float:
        """Calculate rotation angle using leading vehicle trajectory"""
        if 'vehicle_1_x' in data.columns and 'vehicle_1_y' in data.columns:
            vehicle_data = data[['vehicle_1_x', 'vehicle_1_y']].dropna()
            if len(vehicle_data) > 1:
                start = vehicle_data.iloc[0].values
                end = vehicle_data.iloc[-1].values
                direction = end - start
                return -np.arctan2(direction[1], direction[0])
        return 0.0

    def _extract_ego_data(self, data: pd.DataFrame, angle: float) -> dict:
        """Extract ego vehicle trajectory"""
        if 'ego_x' not in data.columns:
            return {'x': [], 'y': []}

        x = data['ego_x'] - data['ego_x'].iloc[0]
        y = data['ego_y'] - data['ego_y'].iloc[0]
        rotated_x, rotated_y = self._rotate_coordinates(x, y, angle)
        return {'x': rotated_x, 'y': -rotated_y}

    def _extract_vehicle_data(self, data: pd.DataFrame, vehicle_id: int, angle: float) -> dict:
        """Extract surrounding vehicle data"""
        x_col = f'vehicle_{vehicle_id}_x'
        y_col = f'vehicle_{vehicle_id}_y'

        if x_col not in data.columns or y_col not in data.columns:
            return {'x': [], 'y': []}

        x = data[x_col] - data['ego_x'].iloc[0]
        y = data[y_col] - data['ego_y'].iloc[0]
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

        # Configure figure elements
        self._configure_axes(ax)
        self._add_colorbar(ax)
        return fig

    def _plot_reference(self, ax):
        """Plot reference trajectory"""
        ref = self.reference_trajectory
        ax.plot(ref['ego']['x'], ref['ego']['y'],
                linestyle='--', linewidth=2,
                color=self.color_config['reference']['color'],
                label=self.color_config['reference']['label'])

        # Add reference score annotation
        ax.annotate(
            f"Score: {ref['score']:.1f}",
            xy=(ref['ego']['x'].iloc[-1], ref['ego']['y'].iloc[-1]),
            xytext=(20, 20),
            textcoords='offset points',
            color=self.color_config['reference']['color'],
            fontsize=10,
            bbox=dict(boxstyle="round", fc="white")
        )

    def _plot_trajectories(self, ax, trajectories: list, source_type: str):
        """Plot specific type of trajectories"""
        config = self.color_config[source_type]
        for traj in trajectories:
            line = ax.plot(traj['ego']['x'], traj['ego']['y'],
                           color=config['color'],
                           alpha=0.6,
                           linewidth=1,
                           label=config['label'])

            # Add score annotation
            ax.annotate(
                f"{traj['score']:.1f}",
                xy=(traj['ego']['x'][-1], traj['ego']['y'][-1]),
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

    def _plot_surrounding_vehicles(self, ax):
        """Plot average surrounding vehicle trajectories"""
        # Combine all valid trajectories
        all_trajs = self.human_trajectories + self.simulated_trajectories

        # Calculate average trajectories
        ped_avg = self._calculate_average_trajectory(all_trajs, 'pedestrian')
        ahead_avg = self._calculate_average_trajectory(all_trajs, 'ahead')

        # Plot stationary vehicles
        if ped_avg:
            ax.plot(ped_avg[0], ped_avg[1],
                    color=self.color_config['pedestrian']['color'],
                    linestyle=':',
                    label=self.color_config['pedestrian']['label'])

        # Plot leading vehicles
        if ahead_avg:
            ax.plot(ahead_avg[0], ahead_avg[1],
                    color=self.color_config['ahead']['color'],
                    linestyle=':',
                    label=self.color_config['ahead']['label'])

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



