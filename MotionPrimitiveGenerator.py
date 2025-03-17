import copy
import random
import warnings

warnings.filterwarnings("ignore")
from tqdm.notebook import tqdm
import itertools
import numpy as np
import matplotlib.pyplot as plt
import yaml
from commonroad.scenario.trajectory import Trajectory
from commonroad.common.solution import VehicleType
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
from commonroad.scenario.state import KSState, InputState
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3


class MotionPrimitiveGenerator:
    """
    Class for generating motion primitives for vehicle trajectory planning.
    """

    class Parameter:
        """
        Class to hold all necessary parameters for motion primitive generation.
        """

        def __init__(self, configuration):
            # output related
            self.dir_output = configuration["outputs"]["output_directory"]

            # vehicle related
            self.id_type_vehicle = configuration["vehicles"]["vehicle_type_id"]

            assert self.id_type_vehicle in [1, 2, 3], "Wrong vehicle type id!"

            if self.id_type_vehicle == 1:
                self.type_vehicle = VehicleType.FORD_ESCORT
                self.param_vehicle = parameters_vehicle1()
                self.name_vehicle = "FORD_ESCORT"
                self.vehicle_dynamics = VehicleDynamics.KS(VehicleType.FORD_ESCORT)

            elif self.id_type_vehicle == 2:
                self.type_vehicle = VehicleType.BMW_320i
                self.param_vehicle = parameters_vehicle2()
                self.name_vehicle = "BMW_320i"
                self.vehicle_dynamics = VehicleDynamics.KS(VehicleType.BMW_320i)

            elif self.id_type_vehicle == 3:
                self.type_vehicle = VehicleType.VW_VANAGON
                self.param_vehicle = parameters_vehicle3()
                self.name_vehicle = "VW_VANAGON"
                self.vehicle_dynamics = VehicleDynamics.KS(VehicleType.VW_VANAGON)

            # time related
            self.duration = configuration["primitives"]["duration"]
            self.dt = configuration["primitives"]["dt"]
            self.dt_sim = configuration["primitives"]["dt_simulation"]
            # times 100 for two-significant-digits accuracy, turns into centi-seconds
            self.time_stamps = (
                    (np.arange(0, self.duration, self.dt_sim) + self.dt_sim) * 100
            ).astype(int)

            self.acceleration_values = configuration["primitives"]["acceleration_values"]
            self.steering_rate_values = configuration["primitives"]["steering_rate_values"]
            self.initial_velocity = configuration["primitives"]["initial_velocity"]
            self.initial_steering_angle = configuration["primitives"]["initial_steering_angle"]

            # velocity related
            self.velocity_sample_min = configuration["primitives"][
                "velocity_sample_min"
            ]
            self.velocity_sample_max = configuration["primitives"][
                "velocity_sample_max"
            ]
            self.num_sample_velocity = configuration["primitives"][
                "num_sample_velocity"
            ]

            # steering angle related
            self.steering_angle_sample_min = configuration["primitives"][
                "steering_angle_sample_min"
            ]
            self.steering_angle_sample_max = configuration["primitives"][
                "steering_angle_sample_max"
            ]
            self.num_sample_steering_angle = configuration["primitives"][
                "num_sample_steering_angle"
            ]

            if np.isclose(self.steering_angle_sample_max, 0):
                self.steering_angle_sample_max = self.param_vehicle.steering.max

            # sample trajectories
            self.num_segments_trajectory = configuration["sample_trajectories"][
                "num_segments_trajectory"
            ]
            self.num_simulations = configuration["sample_trajectories"][
                "num_simulations"
            ]
            # 新增持续时间列表
            self.duration_values = configuration["primitives"].get(
                "duration_values",
                [configuration["primitives"]["duration"]]  # 默认使用全局duration
            )

    # Class attributes
    configuration = None
    parameter: Parameter = None
    VELOCITY_MATCH_TOLERANCE = 2.0
    STEERING_MATCH_TOLERANCE = 0.1
    POSITION_TOLERANCE = 0.3
    ORIENTATION_TOLERANCE = 0.1

    @classmethod
    def load_configuration(cls, path_file_config):
        """
        Load input configuration file.
        """
        with open(path_file_config, "r") as stream:
            try:
                configuration = yaml.safe_load(stream)
                cls.configuration = configuration
                cls.parameter = cls.Parameter(configuration)

            except yaml.YAMLError as exc:
                print(exc)

    @classmethod
    def generate_motion_primitives(cls):
        """
        Generate motion primitives without fixed duration.
        """
        list_samples_a = cls.parameter.acceleration_values
        list_samples_sa_rate = cls.parameter.steering_rate_values

        list_motion_primitives = []
        count_accepted = 0

        print("Generating base motion primitives...")
        total_combinations = len(list_samples_a) * len(list_samples_sa_rate)
        bar_progress = tqdm(total=total_combinations, desc="Generating progress", unit="prim")

        for a, sa_rate in itertools.product(list_samples_a, list_samples_sa_rate):
            initial_state = KSState(
                position=np.array([0.0, 0.0]),
                velocity=cls.parameter.initial_velocity,
                steering_angle=cls.parameter.initial_steering_angle,
                orientation=0.0,
                time_step=0,
            )

            # Simulate until steady state or max steps
            list_states = [initial_state]
            valid = True
            for _ in range(100):  # Max simulation steps
                state_next = cls.parameter.vehicle_dynamics.simulate_next_state(
                    list_states[-1],
                    InputState(acceleration=a, steering_angle_speed=sa_rate),
                    cls.parameter.dt,
                    throw=False
                )
                if not state_next or len(list_states) >= 50:  # Min steps limit
                    break
                list_states.append(state_next)

            if len(list_states) < 2:
                continue

            # Generate base primitive (without fixed duration)
            primitive = Trajectory(initial_time_step=0, state_list=list_states)
            primitive.control_params = (a, sa_rate)
            list_motion_primitives.append(primitive)
            count_accepted += 1

            bar_progress.update(1)

        print(f"Generated base primitives: {count_accepted}")
        return list_motion_primitives

    @classmethod
    def create_mirrored_primitive(cls, primitive):
        """
        Create a mirrored version of the given primitive.
        """
        list_states_mirrored = []

        for state in primitive.state_list:
            try:
                mirrored_position = np.array([state.position[0], -state.position[1]])
                mirrored_orientation = -state.orientation
                mirrored_steering = -state.steering_angle
                mirrored_state = KSState(
                    position=mirrored_position,
                    velocity=state.velocity,
                    steering_angle=mirrored_steering,
                    orientation=mirrored_orientation,
                    time_step=state.time_step
                )
                mirrored_state.custom_fields = getattr(state, 'custom_fields', {}).copy()
                list_states_mirrored.append(mirrored_state)
            except Exception as e:
                print(f"State mirroring error: {str(e)}")
                return None
        try:
            mirrored_traj = Trajectory(
                initial_time_step=0,
                state_list=list_states_mirrored
            )
            mirrored_traj.primitive_id = primitive.primitive_id + "_mirrored"
            mirrored_traj.duration = primitive.duration
            mirrored_traj.custom_fields = primitive.custom_fields.copy()

            return mirrored_traj
        except Exception as e:
            print(f"Trajectory mirroring validation failed: {str(e)}")
            return None

    @classmethod
    def generate_sample_trajectories(cls, list_primitives,
                                     primitive_ids=None,
                                     duration_scales=None,
                                     duration_per_prim=1.0,
                                     visualize=True,
                                     save_path=None):
        """
        Generate continuous trajectories (support for specifying primitive IDs and saving functionality).
        :param save_path: Path to save the trajectory, if provided, the trajectory will be saved.
        """
        # Parameter validation
        if primitive_ids is None:
            primitive_ids = [random.randrange(len(list_primitives))
                             for _ in range(cls.parameter.num_segments_trajectory)]

        if len(primitive_ids) != cls.parameter.num_segments_trajectory:
            raise ValueError(
                f"Need {cls.parameter.num_segments_trajectory} primitive IDs, currently provided {len(primitive_ids)}")

        # Trajectory generation core logic
        current_time_step = 0
        final_states = []
        current_pose = (np.array([0.0, 0.0]), 0.0)  # Ensure starting point is (0,0)

        for seg_idx, pid in enumerate(primitive_ids):
            # Validate primitive ID
            if pid < 0 or pid >= len(list_primitives):
                raise ValueError(f"Invalid primitive ID: {pid} (valid range 0-{len(list_primitives) - 1})")

            # Get current primitive
            primitive = copy.deepcopy(list_primitives[pid])

            # Apply duration scaling
            if duration_scales and seg_idx < len(duration_scales):
                scaled_prim = cls._scale_primitive(primitive, duration_scales[seg_idx])
            else:
                scaled_prim = cls._get_scaled_primitive(
                    [primitive],  # Note: pass single primitive list
                    current_pose,
                    duration=duration_per_prim
                )

            # Coordinate transformation
            transformed_states = []
            for state in scaled_prim.state_list:
                transformed = cls._transform_state(state, current_pose)

                transformed.primitive_id = pid  # Add new attribute

                transformed_states.append(transformed)

            # Update time step
            for state in transformed_states:
                state.time_step = current_time_step
                current_time_step += 1
                final_states.append(state)

            # Update pose
            current_pose = (
                transformed_states[-1].position,
                transformed_states[-1].orientation
            )

        # Build trajectory object
        trajectory = Trajectory(initial_time_step=0, state_list=final_states)

        # Maintain smoothness
        cls._scale_y_coordinate(trajectory, scale_factor=-0.1)

        # Visualization
        if visualize:
            cls.visualize_trajectory(
                trajectory,
                title=f"Specified primitive trajectory (IDs: {primitive_ids})"
            )
        # Save trajectory as CSV file
        if save_path:
            cls.save_trajectory_to_csv(trajectory, save_path)

        return trajectory

    @classmethod
    def _scale_y_coordinate(cls, trajectory, scale_factor):
        """
        Scale y-coordinate while maintaining trajectory smoothness.
        :param trajectory: Trajectory object
        :param scale_factor: Scaling factor (e.g., 0.1)
        """
        for state in trajectory.state_list:
            state.position = (state.position[0], state.position[1] * scale_factor)
            # Adjust velocity or orientation if needed to maintain smoothness

    @classmethod
    def save_trajectory_to_csv(cls, trajectory, file_path):
        """
        Save trajectory as CSV file.
        :param trajectory: Trajectory object
        :param file_path: Path to save the file
        """
        import csv

        # Extract trajectory data
        states = trajectory.state_list
        headers = ["time_step", "position_x", "position_y", "velocity", "steering_angle", "orientation", "primitive_id"]
        rows = []

        for state in states:
            row = [
                state.time_step,
                state.position[0],  # x-coordinate
                state.position[1],  # y-coordinate
                state.velocity,
                state.steering_angle,
                state.orientation,
                state.primitive_id
            ]
            rows.append(row)

        # Write to CSV file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write header
            writer.writerows(rows)  # Write data rows

        print(f"Trajectory saved to: {file_path}")

    @classmethod
    def _scale_primitive(cls, primitive, scale_factor):
        """
        Adjust primitive duration.
        :param primitive: Primitive to be scaled
        :param scale_factor: Scaling factor
        """
        if scale_factor == 1.0:
            return primitive

        original_steps = len(primitive.state_list)
        new_steps = max(int(original_steps * scale_factor), 3)

        # Linear interpolation
        scaled_states = []
        for t in np.linspace(0, original_steps - 1, num=new_steps):
            idx = int(t)
            alpha = t - idx
            next_idx = min(idx + 1, original_steps - 1)

            state = cls._interpolate_states(
                primitive.state_list[idx],
                primitive.state_list[next_idx],
                alpha
            )
            scaled_states.append(state)

        return Trajectory(
            initial_time_step=0,
            state_list=scaled_states
        )

    @classmethod
    def visualize_trajectory(cls, trajectory, show_orientation=True, show_velocity=True,
                             figsize=(10, 5), title="Trajectory Visualization"):
        """
        Visualize trajectory.
        :param trajectory: Trajectory to visualize
        :param show_orientation: Whether to show orientation arrows
        :param show_velocity: Whether to show velocity heatmap
        :param figsize: Figure size
        :param title: Title of the plot
        """
        # Defensive type check
        if not hasattr(trajectory, 'state_list'):
            raise AttributeError(
                f"Trajectory object missing state_list attribute, check Trajectory class definition. Trajectory object type: {type(trajectory)}")

        # Extract trajectory data
        positions = np.array([state.position for state in trajectory.state_list])
        orientations = np.array([state.orientation for state in trajectory.state_list])
        velocities = np.array([state.velocity for state in trajectory.state_list])

        # Create figure
        plt.figure(figsize=figsize)

        # Plot path
        plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Path', linewidth=2)

        # Show orientation arrows
        if show_orientation:
            arrow_step = max(1, len(positions) // 10)
            for i in range(0, len(positions), arrow_step):
                dx = 0.5 * np.cos(orientations[i])
                dy = 0.5 * np.sin(orientations[i])
                plt.arrow(positions[i, 0], positions[i, 1], dx, dy,
                          shape='full', color='r', width=0.1)

        # Show velocity heatmap
        if show_velocity:
            sc = plt.scatter(positions[:, 0], positions[:, 1],
                             c=velocities, cmap='viridis', s=20, alpha=0.7)
            plt.colorbar(sc, label='Velocity (m/s)')

        # Add annotation
        info_text = f"Duration: {len(trajectory.state_list) * cls.parameter.dt:.1f}s\n" \
                    f"Points: {len(trajectory.state_list)}\n" \
                    f"Max Velocity: {np.max(velocities):.2f}m/s"
        plt.annotate(info_text, xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title(title)
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    @classmethod
    def _get_scaled_primitive(cls, primitives, current_pose, duration=1.0):
        """
        Get scaled primitive.
        :param primitives: List of primitives
        :param current_pose: Current pose (position, orientation)
        :param duration: Duration of the primitive
        """
        if not primitives:
            print("Error: Primitive list is empty")
            return None

        # Filter invalid primitives (add state list length check)
        valid_primitives = [p for p in primitives if len(p.state_list) > 1]
        if not valid_primitives:
            print("Warning: All primitives have insufficient state list length, generating default primitive")
            return cls._generate_default_primitive(duration)

        try:
            # Randomly select a primitive (add exception handling)
            selected_prim = random.choice(valid_primitives)
            min_steps = max(int(duration / cls.parameter.dt), 5)
            scaled_states = []

            # Fix variable name spelling error (selected_prim was misspelled as selected_rim)
            for t in np.linspace(0, len(selected_prim.state_list) - 1, num=min_steps):
                idx = int(t)
                alpha = t - idx
                next_idx = min(idx + 1, len(selected_prim.state_list) - 1)  # Fix variable name

                state = cls._interpolate_states(
                    selected_prim.state_list[idx],
                    selected_prim.state_list[next_idx],
                    alpha
                )
                transformed_state = cls._transform_state(state, current_pose)
                scaled_states.append(transformed_state)

            return Trajectory(initial_time_step=0, state_list=scaled_states)

        except Exception as e:
            print(f"Primitive scaling exception: {str(e)}")
            return cls._generate_default_primitive(duration)

    @classmethod
    def _generate_default_primitive(cls, duration):
        """
        Generate a default straight-line motion primitive.
        :param duration: Duration of the primitive
        """
        num_steps = max(int(duration / cls.parameter.dt), 1)
        states = []

        # Initial state
        current_state = KSState(
            position=np.array([0.0, 0.0]),
            velocity=cls.parameter.initial_velocity,
            steering_angle=0.0,
            orientation=0.0,
            time_step=0  # Ensure initial time step is 0
        )
        states.append(current_state)

        # Generate subsequent states
        for step in range(1, num_steps):
            next_state = cls.parameter.vehicle_dynamics.simulate_next_state(
                current_state,
                InputState(acceleration=0, steering_angle_speed=0),
                cls.parameter.dt
            )
            next_state.time_step = step  # Increment time step
            states.append(next_state)
            current_state = next_state

        return Trajectory(initial_time_step=0, state_list=states)

    @staticmethod
    def _interpolate_states(s1, s2, alpha):
        """
        Linear interpolation between two states.
        :param s1: First state
        :param s2: Second state
        :param alpha: Interpolation factor
        """
        new_state = KSState(
            position=s1.position * (1 - alpha) + s2.position * alpha,
            velocity=s1.velocity * (1 - alpha) + s2.velocity * alpha,
            steering_angle=s1.steering_angle * (1 - alpha) + s2.steering_angle * alpha,
            orientation=s1.orientation * (1 - alpha) + s2.orientation * alpha,
            time_step=int(s1.time_step * (1 - alpha) + s2.time_step * alpha)
        )
        return new_state

    @staticmethod
    def _transform_state(state, ref_pose):
        """
        Coordinate transformation.
        :param state: State to transform
        :param ref_pose: Reference pose (position, orientation)
        """
        rotation = np.array([
            [np.cos(ref_pose[1]), -np.sin(ref_pose[1])],
            [np.sin(ref_pose[1]), np.cos(ref_pose[1])]
        ])

        new_pos = ref_pose[0] + rotation @ state.position
        new_orientation = ref_pose[1] + state.orientation

        return KSState(
            position=new_pos,
            velocity=state.velocity,
            steering_angle=state.steering_angle,
            orientation=new_orientation,
            time_step=state.time_step
        )