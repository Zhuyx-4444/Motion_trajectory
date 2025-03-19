import os
import numpy as np
import pandas as pd

class TrajectoryProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.scores_dict = {}
        self.lateral_p25, self.lateral_p50, self.lateral_p75 = None, None, None
        self.longitudinal_p25, self.longitudinal_p50, self.longitudinal_p75 = None, None, None
        self.ttc_p25, self.ttc_p50, self.ttc_p75 = None, None, None

    def process_csv(self, file_path):
        """Process a single CSV file."""
        df = pd.read_csv(file_path)

        # Smooth velocity and position data with min_periods=1 to avoid NaNs
        df['smoothed_velocity'] = df['velocity'].rolling(window=5, min_periods=1, center=False).mean()
        df['smoothed_position_x'] = df['position_x'].rolling(window=5, min_periods=1, center=False).mean()
        df['smoothed_position_y'] = df['position_y'].rolling(window=5, min_periods=1, center=False).mean()
    
        # Calculate lateral velocity and acceleration, filling initial NaNs
        df['lateral_velocity'] = df['smoothed_position_y'].diff()/ 0.1
        # df['lateral_velocity'] = df['lateral_velocity'].rolling(window=5, min_periods=1, center=True).mean()
        df['lateral_acceleration'] = df['lateral_velocity'].diff()/ 0.1
        # df['lateral_acceleration'] = df['lateral_acceleration'].rolling(window=5, min_periods=1, center=True).mean()
    
        # Calculate longitudinal velocity and acceleration, filling initial NaNs
        df['longitudinal_velocity'] = df['smoothed_position_x'].diff() / 1
        # df['longitudinal_velocity'] = df['longitudinal_velocity'].rolling(window=5, min_periods=1, center=True).mean()
        df['longitudinal_acceleration'] = df['longitudinal_velocity'].diff() / 0.1
        # df['longitudinal_acceleration'] = df['longitudinal_acceleration'].rolling(window=5, min_periods=1, center=True).mean()

        # Determine position based on y-axis values
        y_min = df['smoothed_position_y'].min()
        y_max = df['smoothed_position_y'].max()
        if y_min < -1.5:
            df['position'] = 3
        elif y_max > 6:
            df['position'] = 3
        elif y_max > 3:
            df['position'] = 2
        else:
            df['position'] = 0

        # Define static positions
        static_position1 = (85, 2.1)
        static_position2 = (70, 0.0)

        # Calculate TTC for static positions
        def calculate_ttc(position_x, position_y, velocity, static_x, static_y):
            distance = np.sqrt((position_x - static_x)**2 + (position_y - static_y)**2)
            if distance < 1.5:
                return 0
            if velocity == 0:
                return float('inf')
            return distance / velocity

        df['ttc_static1'] = df.apply(lambda row: calculate_ttc(row['smoothed_position_x'], row['smoothed_position_y'], row['smoothed_velocity'], static_position1[0], static_position1[1]), axis=1)
        df['ttc_static2'] = df.apply(lambda row: calculate_ttc(row['smoothed_position_x'], row['smoothed_position_y'], row['smoothed_velocity'], static_position2[0], static_position2[1]), axis=1)

        # Calculate minimum TTC
        df['min_ttc'] = df[['ttc_static1', 'ttc_static2']].min(axis=1)

        # Save processed data to new CSV file
        file_name = os.path.basename(file_path)
        output_path = os.path.join(self.output_folder, f"processed_{file_name}")
        df.to_csv(output_path, index=False)
        # print(f"文件 {file_name} 处理完成，已保存到 {output_path}")

    def batch_process_csv(self):
        """Batch process all CSV files in the input folder."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for file_name in os.listdir(self.input_folder):
            if file_name.endswith(".csv"):
                file_path = os.path.join(self.input_folder, file_name)
                self.process_csv(file_path)

    def load_and_process_file(self, file_path):
        """Load and process a single file to calculate max accelerations and min TTC."""
        df = pd.read_csv(file_path)

        lateral_acceleration_abs = np.abs(df['lateral_acceleration'])
        longitudinal_acceleration_abs = np.abs(df['longitudinal_acceleration'])

        lateral_acceleration_abs = np.clip(lateral_acceleration_abs, None, 10)
        longitudinal_acceleration_abs = np.clip(longitudinal_acceleration_abs, None, 10)

        max_lateral_acceleration = np.max(lateral_acceleration_abs)
        max_longitudinal_acceleration = np.max(longitudinal_acceleration_abs)
        min_ttc = np.min(df['min_ttc'])
        min_ttc = np.clip(min_ttc, 0, None)

        return max_lateral_acceleration, max_longitudinal_acceleration, min_ttc

    def load_all_files(self):
        """Load and process all files in the directory."""
        lateral_accels = []
        longitudinal_accels = []
        ttcs = []
        file_names = []

        for filename in os.listdir(self.output_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.output_folder, filename)
                max_lateral_acc, max_longitudinal_acc, min_ttc = self.load_and_process_file(file_path)

                lateral_accels.append(max_lateral_acc)
                longitudinal_accels.append(max_longitudinal_acc)
                ttcs.append(min_ttc)
                file_names.append(filename)

        data = pd.DataFrame({
            'File': file_names,
            'Max_Lateral_Acceleration': lateral_accels,
            'Max_Longitudinal_Acceleration': longitudinal_accels,
            'Min_TTC': ttcs
        })

        return data

    def calculate_percentiles(self, data):
        """Calculate percentiles for lateral and longitudinal accelerations and TTC."""
        lateral_accels = data['Max_Lateral_Acceleration']
        longitudinal_accels = data['Max_Longitudinal_Acceleration']
        ttcs = data['Min_TTC']

        self.lateral_p25 = np.percentile(lateral_accels, 25)
        self.lateral_p50 = np.percentile(lateral_accels, 50)
        self.lateral_p75 = np.percentile(lateral_accels, 75)

        self.longitudinal_p25 = np.percentile(longitudinal_accels, 25)
        self.longitudinal_p50 = np.percentile(longitudinal_accels, 50)
        self.longitudinal_p75 = np.percentile(longitudinal_accels, 75)

        self.ttc_p25 = np.percentile(ttcs, 25)
        self.ttc_p50 = np.percentile(ttcs, 50)
        self.ttc_p75 = np.percentile(ttcs, 75)

        # print("\n=== 分位数详细数值 ===")
        # print(f'\n横向加速度百分位 | 25%: {self.lateral_p25:.2f} | 50%: {self.lateral_p50:.2f} | 75%: {self.lateral_p75:.2f}')
        # print(f'纵向加速度百分位 | 25%: {self.longitudinal_p25:.2f} | 50%: {self.longitudinal_p50:.2f} | 75%: {self.longitudinal_p75:.2f}')
        # print(f'TTC百分位       | 25%: {self.ttc_p25:.2f} | 50%: {self.ttc_p50:.2f} | 75%: {self.ttc_p75:.2f}')

    def calculate_acceleration_score(self, max_acc, breakpoints, scores):
        """Calculate acceleration score based on breakpoints and scores."""
        return np.interp(max_acc, breakpoints, scores)

    def calculate_ttc_score(self, min_ttc, ttc_breakpoints, ttc_scores):
        """Calculate TTC score based on breakpoints and scores."""
        return np.interp(min_ttc, ttc_breakpoints, ttc_scores)

    def calculate_position_score(self, df):
        """Calculate position score based on position values."""
        total_count = len(df)
        count_0_1 = len(df[df['position'].isin([0, 1])])
        count_2 = len(df[df['position'] == 2])
        count_3 = len(df[df['position'] == 3])

        proportion_0_1 = count_0_1 / total_count
        proportion_2 = count_2 / total_count
        proportion_3 = count_3 / total_count

        score_0_1 = 100
        score_2 = 50
        score_3 = 0

        if proportion_3 == 0:
            score = score_0_1 * proportion_0_1 + score_2 * proportion_2 + score_3 * proportion_3
        else:
            score = 0

        return score

    def process_and_score_file(self, file_path):
        """Process and score a single file."""
        df = pd.read_csv(file_path)

        max_lateral_acc = df['lateral_acceleration'].abs().max()
        max_longitudinal_acc = df['longitudinal_acceleration'].abs().max()
        min_ttc = df['min_ttc'].min()

        if 'collision' in df.columns and (df['collision'] == 1).any():
            ttc_score = 0
        else:
            ttc_breakpoints = [0, self.ttc_p25, self.ttc_p50, self.ttc_p75, 5]
            ttc_scores = [0, 25, 50, 75, 100]
            ttc_score = self.calculate_ttc_score(min_ttc, ttc_breakpoints, ttc_scores)

        lateral_breakpoints = [0, self.lateral_p25, self.lateral_p50, self.lateral_p75, 10]
        lateral_scores = [100, 75, 50, 25, 0]

        longitudinal_breakpoints = [0, self.longitudinal_p25, self.longitudinal_p50, self.longitudinal_p75, 10]
        longitudinal_scores = [100, 75, 50, 25, 0]

        lateral_acc_score = self.calculate_acceleration_score(max_lateral_acc, lateral_breakpoints, lateral_scores)
        longitudinal_acc_score = self.calculate_acceleration_score(max_longitudinal_acc, longitudinal_breakpoints, longitudinal_scores)
        position_score = self.calculate_position_score(df)

        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0].replace("processed_", "", 1)

        return {
            'file': base_name,
            'lateral_acc_score': lateral_acc_score,
            'longitudinal_acc_score': longitudinal_acc_score,
            'ttc_score': ttc_score,
            'position_score': position_score
        }

    def process_and_score_directory(self, output_file):
        """Process and score all files in the directory."""
        scores = []
        for filename in os.listdir(self.output_folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.output_folder, filename)
                scores.append(self.process_and_score_file(file_path))

        scores_df = pd.DataFrame(scores)
        scores_df['average'] = scores_df[
            ['lateral_acc_score', 
             'longitudinal_acc_score',
             'ttc_score',
             'position_score']
        ].mean(axis=1)
        
        scores_df.to_csv(output_file, index=False, columns=[
            'file',
            'lateral_acc_score',
            'longitudinal_acc_score',
            'ttc_score',
            'position_score',
            'average'
        ])
        print(f'得分已写入 {output_file}')



