"""
Step 0: Data Generation
Generate simulated atomic clock telemetry data with:
- Numerous normal units (slow drift + high noise)
- Few anomaly units
- Non-uniform time intervals (data loss)
- Variable segment lengths
"""

import numpy as np
import csv
import os
import pickle
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal as scipy_signal

# Import configuration
from config import DATA_CONFIG, FILE_PATHS


# ==================== Utility Functions ====================

def print_step_header(step_num, step_name):
    """Print step header"""
    print("\n" + "="*70)
    print(f"Step {step_num}: {step_name}")
    print("="*70)


def print_completion(step_name):
    """Print completion message"""
    print("\n" + "="*70)
    print(f"✓ {step_name} Complete!")
    print("="*70)


def save_pickle(data, filepath):
    """Save pickle file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"  ✓ Saved: {filepath}")


def export_to_csv(dataset, filename):
    """
    Export dataset to CSV format

    CSV format:
    - Each row represents data at a time point
    - Columns: unit_id, segment_idx, timestamp, channel_0, ..., channel_N,
               is_anomaly, anomaly_type, trend_pattern, drift_rate, noise_level
    - IMPORTANT: timestamp values include gaps (data loss periods)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = None
        total_rows = 0
        max_time_gap = 0

        for unit_id, unit_data in dataset.items():
            segments = unit_data['segments']
            timestamps = unit_data['timestamps']  # Already includes gaps
            n_channels = segments[0].shape[1]

            if writer is None:
                fieldnames = ['unit_id', 'segment_idx', 'timestamp']
                fieldnames += [f'channel_{i}' for i in range(n_channels)]
                fieldnames += ['is_anomaly', 'anomaly_type', 'trend_pattern',
                             'drift_rate', 'noise_level']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            # Track time gaps for verification
            all_unit_times = np.concatenate(timestamps)
            if len(all_unit_times) > 1:
                time_diffs = np.diff(all_unit_times)
                unit_max_gap = np.max(time_diffs)
                max_time_gap = max(max_time_gap, unit_max_gap)

            for seg_idx, (segment, seg_timestamps) in enumerate(zip(segments, timestamps)):
                for time_idx, (timestamp, data_point) in enumerate(zip(seg_timestamps, segment)):
                    row = {
                        'unit_id': unit_id,
                        'segment_idx': seg_idx,
                        'timestamp': float(timestamp),  # Real timestamp with gaps
                        'is_anomaly': unit_data['is_anomaly'],
                        'anomaly_type': unit_data.get('anomaly_type', ''),
                        'trend_pattern': unit_data.get('trend_pattern', ''),
                        'drift_rate': float(unit_data.get('drift_rate', 0)),
                        'noise_level': float(unit_data.get('noise_level', 0))
                    }

                    for ch_idx in range(n_channels):
                        row[f'channel_{ch_idx}'] = float(data_point[ch_idx])

                    writer.writerow(row)
                    total_rows += 1

    print(f"  ✓ CSV saved: {filename} ({total_rows:,} rows)")
    print(f"    - Max time gap in data: {max_time_gap:.0f} steps (verifies gaps included)")


# ==================== Data Generator ====================

class DataGenerator:
    """Atomic clock telemetry data generator"""

    def __init__(self, config):
        self.n_channels = config['n_channels']
        self.seq_len = config['seq_len']
        self.n_segments = config['n_segments']
        self.drift_rate_base = config['drift_rate_base']
        self.drift_rate_std = config['drift_rate_std']
        self.drift_timescale = config['drift_timescale']
        self.noise_level_base = config['noise_level_base']
        self.noise_level_std = config['noise_level_std']
        self.trend_patterns = config.get('trend_patterns', ['monotonic_increase'])
        self.gap_duration = config['gap_duration']
        self.gap_probability = config.get('gap_probability', 0.1)
        self.anomaly_start_ratio = config.get('anomaly_start_ratio', 0.7)
        self.anomaly_acceleration = config.get('anomaly_acceleration', 0.002)
        self.segment_length_variation = 0.05  # ±5%

    def generate_segment_length(self):
        """Generate random segment length (±5% variation)"""
        variation = int(self.seq_len * self.segment_length_variation)
        return self.seq_len + np.random.randint(-variation, variation + 1)

    def generate_gap_duration(self):
        """Generate random gap duration (50%-150% of base)"""
        min_gap = int(self.gap_duration * 0.5)
        max_gap = int(self.gap_duration * 1.5)
        return np.random.randint(min_gap, max_gap + 1)

    def generate_monotonic_trend(self, total_length, drift_rate, trend_type='increase'):
        """Generate monotonic trend (no periodicity)"""
        time = np.arange(total_length)

        if trend_type == 'monotonic_increase':
            trend = drift_rate * (np.exp(time / self.drift_timescale) - 1)
            trend += drift_rate * time * 0.5

        elif trend_type == 'monotonic_decrease':
            trend = -drift_rate * (np.exp(time / self.drift_timescale) - 1)
            trend -= drift_rate * time * 0.5

        elif trend_type == 'increase_then_decrease':
            turning_point = total_length // 2
            time_increase = time[:turning_point]
            increase_part = drift_rate * (np.exp(time_increase / (self.drift_timescale * 0.5)) - 1)
            peak_value = increase_part[-1]
            time_decrease = np.arange(total_length - turning_point)
            decrease_part = peak_value - drift_rate * (np.exp(time_decrease / (self.drift_timescale * 0.5)) - 1) * 0.8
            trend = np.concatenate([increase_part, decrease_part])

        elif trend_type == 'decrease_then_increase':
            turning_point = total_length // 2
            time_decrease = time[:turning_point]
            decrease_part = -drift_rate * (np.exp(time_decrease / (self.drift_timescale * 0.5)) - 1)
            valley_value = decrease_part[-1]
            time_increase = np.arange(total_length - turning_point)
            increase_part = valley_value + drift_rate * (np.exp(time_increase / (self.drift_timescale * 0.5)) - 1) * 0.8
            trend = np.concatenate([decrease_part, increase_part])

        elif trend_type == 'increase_then_stable':
            stable_point = int(total_length * 0.6)
            time_increase = time[:stable_point]
            increase_part = drift_rate * (1 - np.exp(-time_increase / (self.drift_timescale * 0.3))) * self.drift_timescale * 0.5
            stable_value = increase_part[-1]
            time_stable = np.arange(total_length - stable_point)
            stable_part = stable_value + drift_rate * time_stable * 0.01
            trend = np.concatenate([increase_part, stable_part])

        elif trend_type == 'decrease_then_stable':
            stable_point = int(total_length * 0.6)
            time_decrease = time[:stable_point]
            decrease_part = -drift_rate * (1 - np.exp(-time_decrease / (self.drift_timescale * 0.3))) * self.drift_timescale * 0.5
            stable_value = decrease_part[-1]
            time_stable = np.arange(total_length - stable_point)
            stable_part = stable_value - drift_rate * time_stable * 0.01
            trend = np.concatenate([decrease_part, stable_part])

        else:
            trend = drift_rate * time

        if len(trend) != total_length:
            if len(trend) < total_length:
                trend = np.pad(trend, (0, total_length - len(trend)), mode='edge')
            else:
                trend = trend[:total_length]

        random_walk = np.cumsum(np.random.randn(total_length) * drift_rate * 0.05)
        return trend + random_walk

    def generate_white_noise(self, length, noise_level):
        """Generate Gaussian white noise"""
        return np.random.randn(length) * noise_level

    def add_outliers(self, data, outlier_ratio=0.001):
        """Add occasional outliers"""
        n_outliers = int(len(data) * outlier_ratio)
        if n_outliers > 0:
            outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
            outlier_magnitudes = np.random.randn(n_outliers) * np.std(data) * 5
            data[outlier_indices] += outlier_magnitudes
        return data

    def generate_unit_data(self, unit_id, is_anomaly=False, anomaly_type='acceleration'):
        """Generate data for a single unit with non-uniform intervals and variable lengths"""
        segments = []
        timestamps = []
        segment_lengths = []

        unit_drift_rate = self.drift_rate_base + np.random.randn() * self.drift_rate_std
        unit_noise_level = self.noise_level_base + np.random.rand() * self.noise_level_std
        unit_trend_pattern = np.random.choice(self.trend_patterns)

        avg_segment_length = self.seq_len
        avg_gap_length = self.gap_duration
        expected_gaps = int(self.n_segments * self.gap_probability)
        total_time_steps = int(self.n_segments * avg_segment_length * 1.2 +
                              expected_gaps * avg_gap_length * 1.5)

        print(f"  Generating {unit_id}:")
        print(f"    - Trend pattern: {unit_trend_pattern}")
        print(f"    - Drift rate: {unit_drift_rate:.6f}")
        print(f"    - Noise level: {unit_noise_level:.4f}")

        channel_trends = {}
        current_time = 0

        for seg_idx in range(self.n_segments):
            seg_len = self.generate_segment_length()
            segment_lengths.append(seg_len)

            t = np.arange(seg_len)
            segment = np.zeros((seg_len, self.n_channels))

            # Global time includes gaps - this is the TRUE timeline
            global_time = current_time + t

            for ch in range(self.n_channels):
                channel_factor = 1 + 0.2 * ch
                channel_drift_rate = unit_drift_rate * channel_factor
                channel_noise_level = unit_noise_level * (0.8 + 0.4 * np.random.rand())

                if ch < 3:
                    channel_trend_pattern = unit_trend_pattern
                elif ch < 5:
                    if 'increase' in unit_trend_pattern:
                        channel_trend_pattern = unit_trend_pattern.replace('increase', 'decrease')
                    elif 'decrease' in unit_trend_pattern:
                        channel_trend_pattern = unit_trend_pattern.replace('decrease', 'increase')
                    else:
                        channel_trend_pattern = unit_trend_pattern
                else:
                    other_patterns = [p for p in self.trend_patterns if p != unit_trend_pattern]
                    channel_trend_pattern = np.random.choice(other_patterns) if other_patterns else unit_trend_pattern

                channel_key = (ch, channel_trend_pattern)
                if channel_key not in channel_trends:
                    channel_trends[channel_key] = self.generate_monotonic_trend(
                        total_time_steps,
                        channel_drift_rate,
                        channel_trend_pattern
                    )

                trend_full = channel_trends[channel_key]
                end_time = min(current_time + seg_len, len(trend_full))
                trend_segment = trend_full[current_time:end_time]

                if len(trend_segment) < seg_len:
                    padding_length = seg_len - len(trend_segment)
                    trend_segment = np.pad(trend_segment, (0, padding_length), mode='edge')

                noise = self.generate_white_noise(seg_len, channel_noise_level)
                base_signal = trend_segment + noise

                if is_anomaly and seg_idx > self.n_segments * self.anomaly_start_ratio:
                    anomaly_progress = (seg_idx - self.n_segments * self.anomaly_start_ratio) / \
                                     (self.n_segments * (1 - self.anomaly_start_ratio))

                    if anomaly_type == 'acceleration':
                        if 'increase' in channel_trend_pattern:
                            acceleration = self.anomaly_acceleration * (anomaly_progress ** 2) * global_time
                        else:
                            acceleration = -self.anomaly_acceleration * (anomaly_progress ** 2) * global_time
                        base_signal += acceleration

                    elif anomaly_type == 'shift':
                        if anomaly_progress > 0.5:
                            shift = np.std(base_signal) * 3 * np.sign(np.mean(np.diff(trend_segment)))
                            base_signal += shift

                    elif anomaly_type == 'oscillation':
                        decay = np.exp(-t / (seg_len * 0.3))
                        oscillation = np.std(base_signal) * 0.8 * \
                                    np.sin(2 * np.pi * t / 10) * anomaly_progress * decay
                        base_signal += oscillation

                base_signal = self.add_outliers(base_signal, outlier_ratio=0.001)
                segment[:, ch] = base_signal

            segments.append(segment)
            timestamps.append(global_time)

            # Update current_time: add segment length
            current_time += seg_len

            # Randomly add gap (data loss in time axis)
            if np.random.rand() < self.gap_probability:
                gap_length = self.generate_gap_duration()
                current_time += gap_length  # Skip time during gap
                print(f"    - Gap after segment {seg_idx}: {gap_length} steps")

        print(f"    - Segment stats: min={min(segment_lengths)}, max={max(segment_lengths)}, "
              f"mean={np.mean(segment_lengths):.1f}")

        # Verify timestamps include gaps
        if len(timestamps) > 1:
            all_times_flat = np.concatenate(timestamps)
            time_diffs = np.diff(all_times_flat)
            max_gap = np.max(time_diffs)
            print(f"    - Max time gap: {max_gap:.0f} steps (includes data loss intervals)")

        return {
            'unit_id': unit_id,
            'segments': segments,
            'timestamps': timestamps,
            'segment_lengths': segment_lengths,
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type if is_anomaly else None,
            'drift_rate': unit_drift_rate,
            'noise_level': unit_noise_level,
            'trend_pattern': unit_trend_pattern,
            'metadata': {
                'total_time_steps': current_time,
                'n_segments': self.n_segments,
                'avg_segment_length': np.mean(segment_lengths),
                'segment_length_std': np.std(segment_lengths)
            }
        }

    def generate_dataset(self, n_units, is_anomaly=False):
        """Generate dataset"""
        dataset = {}
        anomaly_types = ['acceleration', 'shift', 'oscillation']

        for i in range(n_units):
            unit_id = f"{'anomaly' if is_anomaly else 'normal'}_unit_{i}"
            anomaly_type = anomaly_types[i % len(anomaly_types)] if is_anomaly else None
            unit_data = self.generate_unit_data(unit_id, is_anomaly, anomaly_type)
            dataset[unit_id] = unit_data

        return dataset


# ==================== Visualization ====================

def visualize_long_term_drift(dataset, save_path=None):
    """Visualize long-term trends with non-uniform intervals"""
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    fig.suptitle('Long-term Monotonic Trend Visualization\n(Non-uniform Intervals, Variable Segment Lengths)',
                 fontsize=16, fontweight='bold')

    normal_units = [k for k in dataset.keys() if 'normal' in k]
    anomaly_units = [k for k in dataset.keys() if 'anomaly' in k]

    if len(normal_units) > 0:
        normal_unit = dataset[normal_units[0]]
        all_segments = np.vstack(normal_unit['segments'])
        all_times = np.concatenate([ts for ts in normal_unit['timestamps']])

        # 1. Complete time series
        axes[0, 0].plot(all_times, all_segments[:, 0], 'b-', linewidth=0.5, alpha=0.7)
        segment_lengths = normal_unit['segment_lengths']
        current_pos = 0
        for seg_len in segment_lengths[:-1]:
            current_pos += seg_len
            axes[0, 0].axvline(x=all_times[current_pos], color='red',
                              linestyle='--', alpha=0.3, linewidth=0.8)
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title(f'Normal Unit - Channel 0\nTrend: {normal_unit["trend_pattern"]}')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Zoomed view
        zoom_start = len(all_times) // 2
        zoom_end = zoom_start + 10000
        axes[0, 1].plot(all_times[zoom_start:zoom_end],
                       all_segments[zoom_start:zoom_end, 0], 'b-', linewidth=1)
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].set_title('Zoomed View (High Noise)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Multi-channel comparison
        for ch in range(min(5, all_segments.shape[1])):
            axes[1, 0].plot(all_times, all_segments[:, ch],
                          alpha=0.6, linewidth=0.5, label=f'Ch{ch}')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].set_title('Multi-channel Comparison')
        axes[1, 0].legend(loc='best', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Moving average
        window = 500
        ch0_data = all_segments[:, 0]
        smoothed = np.convolve(ch0_data, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(all_times[:len(smoothed)], smoothed, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Moving Average')
        axes[1, 1].set_title(f'Pure Trend (MA Window={window})')
        axes[1, 1].grid(True, alpha=0.3)

        # 5. Segment length distribution
        axes[2, 0].hist(segment_lengths, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[2, 0].axvline(x=np.mean(segment_lengths), color='red',
                          linestyle='--', linewidth=2, label=f'Mean={np.mean(segment_lengths):.1f}')
        axes[2, 0].set_xlabel('Segment Length')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title(f'Segment Length Distribution')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # 6. Spectral analysis
        fft_vals = np.abs(fft(ch0_data))
        freqs = fftfreq(len(ch0_data), d=1.0)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_vals[:len(fft_vals)//2]
        axes[2, 1].semilogy(positive_freqs[1:100], positive_fft[1:100], 'b-', linewidth=1)
        axes[2, 1].set_xlabel('Frequency')
        axes[2, 1].set_ylabel('Power Spectrum (Log)')
        axes[2, 1].set_title('Spectral Analysis')
        axes[2, 1].grid(True, alpha=0.3)

    # Anomaly comparison
    if len(anomaly_units) > 0 and len(normal_units) > 0:
        anomaly_unit = dataset[anomaly_units[0]]
        all_segments_anom = np.vstack(anomaly_unit['segments'])
        all_times_anom = np.concatenate([ts for ts in anomaly_unit['timestamps']])

        # 7. Normal vs Anomaly
        normal_smooth = np.convolve(all_segments[:, 0], np.ones(500)/500, mode='valid')
        anomaly_smooth = np.convolve(all_segments_anom[:, 0], np.ones(500)/500, mode='valid')
        axes[3, 0].plot(all_times[:len(normal_smooth)], normal_smooth, 'b-',
                      linewidth=2, alpha=0.7, label='Normal')
        axes[3, 0].plot(all_times_anom[:len(anomaly_smooth)], anomaly_smooth, 'r-',
                       linewidth=2, alpha=0.7, label=f'Anomaly ({anomaly_unit["anomaly_type"]})')
        axes[3, 0].set_xlabel('Time Steps')
        axes[3, 0].set_ylabel('Amplitude (Smoothed)')
        axes[3, 0].set_title('Normal vs Anomaly Comparison')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)

        # 8. Time interval distribution
        time_diffs = np.diff(all_times)
        axes[3, 1].hist(time_diffs, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[3, 1].axvline(x=np.median(time_diffs), color='red',
                          linestyle='--', linewidth=2, label=f'Median={np.median(time_diffs):.1f}')
        axes[3, 1].set_xlabel('Time Interval')
        axes[3, 1].set_ylabel('Frequency')
        axes[3, 1].set_title('Time Interval Distribution')
        axes[3, 1].legend()
        axes[3, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Visualization saved: {save_path}")
    plt.show()


# ==================== Main ====================

def main():
    print_step_header(0, "Data Generation - Long-term Gradual Change")

    print("\nData Generation Parameters:")
    print(f"  Normal units: {DATA_CONFIG['n_normal_units']}")
    print(f"  Anomaly units: {DATA_CONFIG['n_anomaly_units']}")
    print(f"  Segments per unit: {DATA_CONFIG['n_segments']}")
    print(f"  Base segment length: {DATA_CONFIG['seq_len']} (±5%)")
    print(f"  Base drift rate: {DATA_CONFIG['drift_rate_base']}")
    print(f"  Base noise level: {DATA_CONFIG['noise_level_base']}")
    print(f"  Gap probability: {DATA_CONFIG['gap_probability']}")

    generator = DataGenerator(DATA_CONFIG)

    print(f"\nGenerating {DATA_CONFIG['n_normal_units']} normal units...")
    print("="*70)
    normal_data = generator.generate_dataset(
        n_units=DATA_CONFIG['n_normal_units'],
        is_anomaly=False
    )
    print("="*70)

    print(f"\nGenerating {DATA_CONFIG['n_anomaly_units']} anomaly units...")
    print("="*70)
    anomaly_data = generator.generate_dataset(
        n_units=DATA_CONFIG['n_anomaly_units'],
        is_anomaly=True
    )
    print("="*70)

    print(f"\nDataset Statistics:")
    print(f"  Normal units: {len(normal_data)}")
    print(f"  Anomaly units: {len(anomaly_data)}")

    sample_unit = list(normal_data.values())[0]
    all_segments = np.vstack(sample_unit['segments'])
    print(f"\nSample Unit Info:")
    print(f"  Complete shape: {all_segments.shape}")
    print(f"  Value range: [{np.min(all_segments):.4f}, {np.max(all_segments):.4f}]")
    print(f"  Mean: {np.mean(all_segments):.4f}")
    print(f"  Std: {np.std(all_segments):.4f}")

    print(f"\nSaving data...")
    save_pickle(normal_data, FILE_PATHS['raw_normal_data'])
    save_pickle(anomaly_data, FILE_PATHS['raw_anomaly_data'])

    print(f"\nExporting CSV...")
    export_to_csv(normal_data, 'data/normal_data.csv')
    export_to_csv(anomaly_data, 'data/anomaly_data.csv')
    combined_data = {**normal_data, **anomaly_data}
    export_to_csv(combined_data, 'data/all_data.csv')

    print(f"\nGenerating visualization...")
    visualize_long_term_drift(combined_data, save_path=FILE_PATHS['data_visualization'])

    print_completion("Data Generation")

    print("\n" + "="*70)
    print("Generated Files:")
    print(f"  Pickle: {FILE_PATHS['raw_normal_data']}")
    print(f"  Pickle: {FILE_PATHS['raw_anomaly_data']}")
    print(f"  CSV: data/normal_data.csv")
    print(f"  CSV: data/anomaly_data.csv")
    print(f"  CSV: data/all_data.csv")
    print(f"  Figure: {FILE_PATHS['data_visualization']}")
    print("="*70)


if __name__ == "__main__":
    main()