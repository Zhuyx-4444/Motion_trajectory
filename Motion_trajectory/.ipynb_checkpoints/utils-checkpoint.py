import os
import click
from sample_motion import MotionPrimitiveHandler
from preprocess import TrajectoryProcessor
from paper_plot import EnhancedTrajectoryComparator


class ExperimentManager:
    def __init__(self, config_paths):
        """
        Unified experiment management system

        :param config_paths: Dictionary of configuration paths:
            {
                'single_primitive': 'path/to/single_primitives.yaml',
                'symmetric_primitives': 'path/to/symmetric_primitives.yaml'
            }
        """
        self.config = config_paths
        self.handler = MotionPrimitiveHandler()

        # Standardized directory structure
        self.dirs = {
            'human_data': 'Human_trajectory',
            'reference': 'Reference_trajectory',
            'simulated': 'Simulated_trajectories',
            'results': 'Analysis_Results'
        }

    def setup_directories(self):
        """Create standardized directory structure"""
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    def generate_reference_data(self):
        """Generate and process reference trajectory"""
        print("\n=== Generating Reference Trajectory ===")
        self.handler.generate_single_trajectory(
            self.config['single_primitive'],
            output_dir=self.dirs['reference']
        )
       

    def generate_simulated_data(self, num_pairs=50):
        """Generate simulated trajectory pairs"""
        print("\n=== Generating Simulated Trajectories ===")
        self.handler.generate_symmetric_trajectories(
            self.config['symmetric_primitives'],
            num_pairs=num_pairs,
            output_dir=self.dirs['simulated']
        )

    def analyze_results(self):
        """Run complete analysis pipeline"""
        print("\n=== Performing Comparative Analysis ===")
        comparator = self._create_comparator()

        comparator.load_all_data() 

        # Generate both comparison figures
        fig1 = comparator.generate_comparison_figure('human_vs_reference')
        fig1.savefig(
            os.path.join(self.dirs['results'], 'human_vs_reference.png'),
            dpi=300, bbox_inches='tight'
        )

        fig2 = comparator.generate_comparison_figure('simulated_vs_reference')
        fig2.savefig(
            os.path.join(self.dirs['results'], 'simulated_vs_reference.png'),
            dpi=300, bbox_inches='tight'
        )

    def _create_comparator(self):
        """Initialize trajectory comparator with validated paths"""
        reference_path = os.path.join(self.dirs['reference'], 'reference_trajectory.csv')
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"参考轨迹文件未找到: {reference_path}。请先生成参考轨迹。")
        print(f"参考轨迹路径验证成功: {reference_path}") 
    
        return EnhancedTrajectoryComparator(
            human_data_folder=os.path.join(self.dirs['human_data'], 'processed'),
            human_score_path=os.path.join(self.dirs['human_data'], 'scores.csv'),
            reference_path=reference_path,  
            simulated_folder=os.path.join(self.dirs['simulated'], 'processed'),
            simulated_score_path=os.path.join(self.dirs['simulated'], 'scores.csv')
        )

@click.command()
# @click.option('--config-single', required=True, help='Path to single primitive config')
# @click.option('--config-symmetric', required=True, help='Path to symmetric primitives config')
# @click.option('--num-pairs', default=50, help='Number of trajectory pairs to generate')
@click.option('--config-single', 
              default=os.path.join(os.path.dirname(__file__), 'generator_config_excellent.yaml'),
              help='Path to single primitive config (default: ./Motion_trajectory/generator_config_excellent.yaml)')
@click.option('--config-symmetric',
              default=os.path.join(os.path.dirname(__file__), 'generator_config_simulation.yaml'),
              help='Path to symmetric primitives config (default: ./Motion_trajectory/generator_config_simulation.yaml)')
@click.option('--num-pairs', default=100, help='Number of trajectory pairs to generate (default: 25)')
def main(config_single, config_symmetric, num_pairs):
    """Main execution pipeline"""
    manager = ExperimentManager({
        'single_primitive': config_single,
        'symmetric_primitives': config_symmetric
    })

    # Setup environment
    manager.setup_directories()

    # Data generation phase
    manager.generate_reference_data()
    manager.generate_simulated_data(num_pairs)

    # Analysis phase
    manager.analyze_results()

    print("\n=== Experiment Completed Successfully ===")


if __name__ == "__main__":
    main()
