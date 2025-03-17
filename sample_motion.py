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
        self.mirrored_primitives = [MPG.create_mirrored_primitive(p) for p in self.primitives]

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
        # Placeholder for actual implementation
        return np.random.randint(0, len(self.primitives))

    def _get_mirrored_primitive_id(self, pid):
        """
        Get the mirrored primitive ID.
        """
        return pid  # Placeholder for actual implementation

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
        processed_dir = os.path.join(output_dir, "processed")
        scores_path = os.path.join(output_dir, "scores.csv")

        # Generate trajectory
        initial_primitive = np.random.randint(0, 25)
        lhs_samples = self._latin_hypercube_sample(5)
        primitive_sequence = [self._find_nearest_primitive(s[0], s[1]) for s in lhs_samples]

        # Save trajectory
        MPG.generate_sample_trajectories(
            self.mirrored_primitives,
            primitive_ids=[initial_primitive] + primitive_sequence,
            duration_scales=[0.5] + [1.0] * 4,
            save_path=os.path.join(output_dir, "reference_trajectory.csv")
        )

        # Process and score trajectory
        processor = TrajectoryProcessor(
            input_folder=output_dir,
            output_folder=processed_dir
        )
        processor.batch_process_csv()
        scoring_data = processor.load_all_files()
        processor.calculate_percentiles(scoring_data)
        processor.process_and_score_directory(scores_path)

        print(f"Generated single trajectory with scores saved to {scores_path}")

    def generate_symmetric_trajectories(self, config_path, num_pairs=50, output_dir="simulated_trajectory"):
        """
        Generate multiple symmetric trajectories with 6 motion primitives, process, and save them.

        :param config_path: Path to the configuration YAML file for 6 primitives
        :param num_pairs: Number of trajectory pairs to generate
        :param output_dir: Output directory path
        """
        # Load configuration and generate primitives
        self._load_configuration(config_path)

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        processed_dir = os.path.join(output_dir, "processed")
        scores_path = os.path.join(output_dir, "scores.csv")

        # Generate trajectories
        self._initialize_bins(n_bins=6)
        for pair_idx in range(num_pairs):
            # Original trajectory
            initial_primitive = np.random.randint(0, 25)
            lhs_samples = self._latin_hypercube_sample(6)
            primitive_sequence = [self._find_nearest_primitive(s[0], s[1]) for s in lhs_samples]

            # Save trajectory pair
            self._save_trajectory_pair(
                primitives=[initial_primitive] + primitive_sequence,
                output_dir=output_dir,
                pair_idx=pair_idx
            )

        # Process and score trajectories
        processor = TrajectoryProcessor(
            input_folder=output_dir,
            output_folder=processed_dir
        )
        processor.batch_process_csv()
        scoring_data = processor.load_all_files()
        processor.calculate_percentiles(scoring_data)
        processor.process_and_score_directory(scores_path)

        print(f"Generated {num_pairs} pairs with scores saved to {scores_path}")

    def _save_trajectory_pair(self, primitives, output_dir, pair_idx):
        """Save trajectory pair with mirrored version"""
        # Original trajectory
        MPG.generate_sample_trajectories(
            self.mirrored_primitives,
            primitive_ids=primitives,
            duration_scales=[0.5] + [1.0] * (len(primitives) - 1),
            save_path=os.path.join(output_dir, f"simulated_trajectory_{2 * pair_idx + 1}.csv")
        )

        # Mirrored trajectory
        mirrored_primitives = [self._get_mirrored_primitive_id(pid) for pid in primitives]
        MPG.generate_sample_trajectories(
            self.mirrored_primitives,
            primitive_ids=mirrored_primitives,
            duration_scales=[0.5] + [1.0] * (len(primitives) - 1),
            save_path=os.path.join(output_dir, f"simulated_trajectory_{2 * pair_idx + 2}.csv")
        )


#