import numpy as np
import os
from MotionPrimitiveGenerator import MotionPrimitiveGenerator as MPG
from preprocess import TrajectoryProcessor  # Import scoring processor


class MotionPrimitiveHandler:
    def __init__(self, speed_range=(-1.4, 13.3), angle_range=(-0.09, 0.07)):
        """
        Enhanced motion primitive handler with integrated scoring

        :param speed_range: Tuple (min_speed, max_speed) in m/s
        :param angle_range: Tuple (min_steering_angle, max_steering_angle) in radians
        """
        self.speed_range = speed_range
        self.angle_range = angle_range

        # Initialize sampling parameters
        self.n_bins = 5
        self._initialize_bins()

    def _load_configuration(self, config_path):
        """
        Load configuration from the specified YAML file.
        """
        MPG.load_configuration(config_path)
        self.primitives = MPG.generate_motion_primitives()
        # self.mirrored_primitives = [MPG.create_mirrored_primitive(p) for p in self.primitives]
        self.mirrored_primitives = []
        for p in self.primitives:
            mirrored = MPG.create_mirrored_primitive(p)
            if mirrored is not None:
                self.mirrored_primitives.append(mirrored)
            else:
                print(f"Warning: Failed to mirror primitive {getattr(p, 'primitive_id', 'unknown')}")

    def _initialize_bins(self, n_bins=5):
        """
        Initialize bins for Latin Hypercube Sampling.
        """
        self.n_bins = n_bins
        self.speed_bins = np.linspace(self.speed_range[0], self.speed_range[1], num=n_bins + 1)
        self.angle_bins = np.linspace(self.angle_range[0], self.angle_range[1], num=n_bins + 1)

    def _latin_hypercube_sample(self, n_samples):
        """
        Generate Latin Hypercube samples for speed and steering angle.
        """
        samples = []
        for i in range(n_samples):
            speed_sample = np.random.uniform(self.speed_bins[i], self.speed_bins[i + 1])
            angle_sample = np.random.uniform(self.angle_bins[i], self.angle_bins[i + 1])
            samples.append((speed_sample, angle_sample))
        return samples

    def _find_nearest_primitive(self, speed, angle):
        """
        Find the nearest primitive based on speed and steering angle.
        """
        min_distance = float('inf')
        best_id = -1
        for idx, traj in enumerate(self.primitives):
            # 获取终态特征
            end_state = traj.state_list[-1]
            traj_speed = end_state.velocity
            traj_angle = end_state.steering_angle
            
            # 计算欧氏距离
            distance = np.sqrt((speed - traj_speed)**2 + (angle - traj_angle)**2)
            if distance < min_distance:
                min_distance = distance
                best_id = idx
        return best_id

        
    def _get_mirrored_primitive_id(self, pid):
        """
        Get the mirrored primitive ID.
        """
        group = pid // 5  
        position = pid % 5 
        mirrored_position = 4 - position  
        return group * 5 + mirrored_position

    def generate_single_trajectory(self, config_path, output_dir="reference_trajectory"):
        """
        Generate a single trajectory with 5 motion primitives, process, and save it.

        :param config_path: Path to the configuration YAML file for 5 primitives
        :param output_dir: Output directory path
        """
        # Load configuration and generate primitives
        self._load_configuration(config_path)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)


        # Save trajectory
        MPG.generate_sample_trajectories(
            self.mirrored_primitives,
            primitive_ids= [5, 9, 5, 7, 11],
            duration_scales= [0.8, 0.6, 0.5, 0.5, 0.6],
            initial_pose=(np.array([0.0, 0.0]), -0.6),
            save_path=os.path.join(output_dir, "reference_trajectory.csv")
        )
        print(f"Generated single trajectory saved to {output_dir}")

    def generate_symmetric_trajectories(self, config_path, num_pairs=100, output_dir="simulated_trajectory"):
        """
        Generate multiple symmetric trajectories with 6 motion primitives, process, and save them.

        :param config_path: Path to the configuration YAML file for 6 primitives
        :param num_pairs: Number of trajectory pairs to generate
        :param output_dir: Output directory path
        """
        # Load configuration and generate primitives
        self._load_configuration(config_path)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        processed_dir = os.path.join(output_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # 轨迹生成主循环
        for pair_idx in range(num_pairs):
            # 生成LHS采样序列
            lhs_samples = self._latin_hypercube_sample(5)
            primitive_sequence = [self._find_nearest_primitive(s[0], s[1]) for s in lhs_samples]
            
            # 添加初始随机基元
            initial_primitive = np.random.randint(0, 25)
            full_sequence = [initial_primitive] + primitive_sequence
            
            # 生成原始轨迹
            self._generate_trajectory(
                primitive_ids=full_sequence,
                output_path=os.path.join(output_dir, f"simulated_trajectory_{2*pair_idx+1}.csv")
            )
            
            # 生成镜像轨迹
            mirrored_sequence = [self._get_mirrored_primitive_id(pid) for pid in full_sequence]
            self._generate_trajectory(
                primitive_ids=mirrored_sequence,
                output_path=os.path.join(output_dir, f"simulated_trajectory_{2*pair_idx+2}.csv")
            )

        # 后处理流程
        processor = TrajectoryProcessor(input_folder=output_dir, output_folder=processed_dir)
        processor.batch_process_csv()
        scoring_data = processor.load_all_files()
        processor.calculate_percentiles(scoring_data)
        processor.process_and_score_directory(os.path.join(output_dir, "scores.csv"))


    def _generate_trajectory(self, primitive_ids, output_path):
        """生成单个轨迹的封装方法"""
        MPG.generate_sample_trajectories(
            self.mirrored_primitives,
            primitive_ids=primitive_ids,
            duration_scales=[0.5] + [0.8, 0.6, 0.5, 0.5, 0.6],
            initial_pose=(np.array([0.0, 0.0]), 0),
            save_path=output_path
        )

