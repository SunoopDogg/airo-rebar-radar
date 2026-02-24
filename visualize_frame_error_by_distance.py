#!/usr/bin/env python3
"""
Frame-based Error Visualization by Distance

Visualizes rebar detection error rates at different confidence levels (50%, 75%, 90%, 95%, 99%)
overlaid on the structure for each case and distance combination.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.structure.config import (
    create_ppvc_linear_config,
    create_ppvc_cluster_2_config,
    create_ppvc_cluster_4_config,
)


# Confidence level to frame and CSV interval mapping
# Based on convergence analysis results
CONFIDENCE_LEVELS = {
    50: {'frame': 59, 'interval': 50},
    75: {'frame': 79, 'interval': 70},
    90: {'frame': 96, 'interval': 90},
    95: {'frame': 100, 'interval': 100},
    99: {'frame': 109, 'interval': 100},  # Using interval 100 as closest available
}

# Confidence level colors
CONFIDENCE_COLORS = {
    50: '#90EE90',  # Light green
    75: '#87CEEB',  # Sky blue
    90: '#FFD700',  # Gold
    95: '#FFA500',  # Orange
    99: '#FF6347',  # Tomato red
}

# Center X mapping by lidar distance (cm) - new values from plan
CENTER_X_MAPPING_LINEAR = {
    60: 1.25, 120: 0.65, 180: 0.05,
    240: -0.55, 300: -1.15, 360: -1.75
}
CENTER_X_MAPPING_CLUSTER = {
    60: 1.1, 120: 0.5, 180: -0.1,
    240: -0.7, 300: -1.3, 360: -1.9
}

# Center Y values by structure type
CENTER_Y = {
    'LINEAR': 0,
    'CLUSTER_2': -0.1,
    'CLUSTER_4': -0.045
}

# Case to structure type mapping
STRUCTURE_TYPE = {
    'case1': 'LINEAR',
    'case2': 'CLUSTER_2',
    'case3': 'CLUSTER_4',
}

CASE_TITLES = {
    'case1': 'Case 1 (LINEAR - 4 rebars)',
    'case2': 'Case 2 (CLUSTER_2 - 4 rebars)',
    'case3': 'Case 3 (CLUSTER_4 - 8 rebars)',
}

# Expected rebar radius in meters
EXPECTED_RADIUS = 0.0125  # 12.5mm


def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Parse filename to extract case, scan number, and distance.
    Example: S3_1-1_120_tracks.csv -> ('case1', 1, 120)
    """
    match = re.match(r'S3_(\d)-(\d)_(\d+)_tracks\.csv', filename)
    if match:
        case_num = int(match.group(1))
        scan_num = int(match.group(2))
        distance = int(match.group(3))
        return f'case{case_num}', scan_num, distance
    return None, None, None


def load_all_tracks(results_dir: Path, cases: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load all tracks.csv files from results directory.

    Args:
        results_dir: Path to results directory
        cases: List of cases to include (e.g., ['case1', 'case2', 'case3'])

    Returns:
        Combined DataFrame with metadata columns: case, scan, distance
    """
    if cases is None:
        cases = ['case1', 'case2', 'case3']

    all_data = []

    for case_folder in cases:
        case_path = results_dir / case_folder
        if not case_path.exists():
            print(f"Warning: Case folder not found: {case_path}")
            continue

        for subdir in case_path.iterdir():
            if not subdir.is_dir():
                continue

            tracks_file = subdir / f'{subdir.name}_tracks.csv'
            if not tracks_file.exists():
                continue

            case, scan, distance = parse_filename(tracks_file.name)
            if case is None:
                continue

            df = pd.read_csv(tracks_file)
            df['case'] = case
            df['scan'] = scan
            df['distance'] = distance
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def get_position_at_interval(track_row: pd.Series, interval: int) -> Optional[Tuple[float, float, float]]:
    """
    Extract position and radius at a specific interval from a track row.

    Args:
        track_row: Single row from tracks DataFrame
        interval: Interval value (e.g., 50, 70, 90, 100)

    Returns:
        Tuple of (x, y, radius) or None if data is missing
    """
    # Build column names based on interval
    if interval < 100:
        x_col = f'{interval:02d}_center_x'
        y_col = f'{interval:02d}_center_y'
        r_col = f'{interval:02d}_radius'
    else:
        x_col = f'{interval}_center_x'
        y_col = f'{interval}_center_y'
        r_col = f'{interval}_radius'

    # Check if columns exist
    if x_col not in track_row.index:
        # Try without leading zero
        x_col = f'{interval}_center_x'
        y_col = f'{interval}_center_y'
        r_col = f'{interval}_radius'

    if x_col not in track_row.index:
        return None

    x = track_row.get(x_col)
    y = track_row.get(y_col)
    r = track_row.get(r_col)

    if pd.isna(x) or pd.isna(y) or pd.isna(r):
        return None

    return (float(x), float(y), float(r))


def calculate_errors(
    detected_pos: Tuple[float, float, float],
    expected_positions: List[Tuple[float, float]],
    expected_radius: float = EXPECTED_RADIUS,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Calculate position and radius errors for a detected rebar.

    Args:
        detected_pos: (x, y, radius) of detected rebar
        expected_positions: List of (x, y) expected rebar positions
        expected_radius: Expected radius in meters

    Returns:
        Tuple of (position_error_mm, radius_error_mm, nearest_expected_pos)
    """
    detected_x, detected_y, detected_r = detected_pos

    # Find minimum distance to any expected position
    min_dist = float('inf')
    nearest_pos = expected_positions[0] if expected_positions else (0, 0)

    for exp_x, exp_y in expected_positions:
        dist = np.sqrt((detected_x - exp_x)**2 + (detected_y - exp_y)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_pos = (exp_x, exp_y)

    position_error_mm = min_dist * 1000  # Convert to mm
    radius_error_mm = (detected_r - expected_radius) * 1000  # Convert to mm

    return position_error_mm, radius_error_mm, nearest_pos


def get_structure_config(case: str, distance: int):
    """
    Create structure configuration for a given case and distance.

    Args:
        case: Case identifier ('case1', 'case2', 'case3')
        distance: Lidar distance in cm

    Returns:
        StructureConfig instance
    """
    structure_type = STRUCTURE_TYPE[case]

    if structure_type == 'LINEAR':
        center_x = CENTER_X_MAPPING_LINEAR[distance]
    else:
        center_x = CENTER_X_MAPPING_CLUSTER[distance]

    center_y = CENTER_Y[structure_type]

    if case == 'case1':
        return create_ppvc_linear_config(center_x=center_x, center_y=center_y)
    elif case == 'case2':
        return create_ppvc_cluster_2_config(center_x=center_x, center_y=center_y)
    else:  # case3
        return create_ppvc_cluster_4_config(center_x=center_x, center_y=center_y)


def find_nearest_expected_index(
    pos: Tuple[float, float, float],
    expected_positions: List[Tuple[float, float]],
) -> int:
    """
    Find the index of the nearest expected rebar position.

    Args:
        pos: (x, y, radius) of detected rebar
        expected_positions: List of (x, y) expected rebar positions

    Returns:
        Index of the nearest expected rebar
    """
    min_dist = float('inf')
    nearest_idx = 0

    for idx, (exp_x, exp_y) in enumerate(expected_positions):
        dist = np.sqrt((pos[0] - exp_x)**2 + (pos[1] - exp_y)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx

    return nearest_idx


def plot_frame_error_comparison(
    case: str,
    distance: int,
    tracks_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Dict:
    """
    Plot error comparison across confidence levels for a single case/distance.
    Shows individual cropped views per rebar in a grid layout.

    Args:
        case: Case identifier
        distance: Lidar distance in cm
        tracks_df: DataFrame containing tracks data
        output_path: Path to save the plot
        show: Whether to display the plot

    Returns:
        Dictionary with error statistics
    """
    # Filter to specific case and distance
    df = tracks_df[(tracks_df['case'] == case) & (tracks_df['distance'] == distance)]

    if df.empty:
        print(f"No data for {case} at {distance}cm")
        return {}

    # Get structure configuration
    config = get_structure_config(case, distance)
    expected_positions = config.get_track_positions()
    expected_radius = config.track_diameter / 2
    n_rebars = len(expected_positions)

    # Determine grid layout based on number of rebars
    if n_rebars <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 4

    # Create figure with GridSpec (grid + stats panel)
    fig = plt.figure(figsize=(18, 10), dpi=150)
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1]*ncols + [0.5],
                          hspace=0.3, wspace=0.3)

    # Create subplot axes for each rebar
    rebar_axes = []
    for idx in range(n_rebars):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        rebar_axes.append(ax)

    # Statistics panel (rightmost column, spans all rows)
    ax_stats = fig.add_subplot(gs[:, -1])
    ax_stats.axis('off')

    # Crop margin (±50mm = 0.05m)
    crop_margin = 0.05

    # Initialize data structure to collect detections per rebar
    # rebar_detections[rebar_idx][confidence] = [(x, y, r), ...]
    rebar_detections = {idx: {conf: [] for conf in CONFIDENCE_LEVELS.keys()}
                        for idx in range(n_rebars)}

    # Statistics storage
    stats = {}

    # Process each confidence level and assign detections to rebars
    for confidence, conf_info in CONFIDENCE_LEVELS.items():
        interval = conf_info['interval']

        position_errors = []
        radius_errors = []

        # Process each track
        for _, row in df.iterrows():
            pos = get_position_at_interval(row, interval)
            if pos is None:
                continue

            pos_error, rad_error, nearest = calculate_errors(
                pos, expected_positions, expected_radius
            )
            position_errors.append(pos_error)
            radius_errors.append(rad_error)

            # Find nearest expected rebar and assign detection
            nearest_idx = find_nearest_expected_index(pos, expected_positions)
            rebar_detections[nearest_idx][confidence].append(pos)

        # Calculate statistics
        if position_errors:
            stats[confidence] = {
                'position_mean': np.mean(position_errors),
                'position_std': np.std(position_errors),
                'position_max': np.max(position_errors),
                'radius_mean': np.mean(radius_errors),
                'radius_std': np.std(radius_errors),
                'count': len(position_errors),
            }

    # Draw each rebar subplot
    for idx, (exp_x, exp_y) in enumerate(expected_positions):
        ax = rebar_axes[idx]

        # Set crop range
        ax.set_xlim(exp_x - crop_margin, exp_x + crop_margin)
        ax.set_ylim(exp_y - crop_margin, exp_y + crop_margin)
        ax.set_aspect('equal')
        ax.set_title(f'Rebar #{idx+1}', fontsize=10, fontweight='bold')

        # Draw expected rebar (blue dashed circle with center marker)
        expected_circle = plt.Circle(
            (exp_x, exp_y),
            expected_radius,
            fill=False,
            edgecolor='blue',
            linestyle=':',
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(expected_circle)
        ax.plot(exp_x, exp_y, '+', color='blue', markersize=10, markeredgewidth=2)

        # Draw detected rebars for each confidence level
        for confidence in CONFIDENCE_LEVELS.keys():
            color = CONFIDENCE_COLORS[confidence]
            detections = rebar_detections[idx][confidence]

            for pos in detections:
                # Draw detected rebar circle
                circle = plt.Circle(
                    (pos[0], pos[1]),
                    pos[2],
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax.add_patch(circle)

                # Draw center marker
                ax.plot(
                    pos[0], pos[1],
                    'o',
                    color=color,
                    markersize=4,
                    alpha=0.8,
                )

        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

    # Add main title
    fig.suptitle(
        f'{CASE_TITLES.get(case, case)} - Distance: {distance}cm\n'
        f'Rebar Detection Error by Confidence Level (Cropped View)',
        fontsize=14, fontweight='bold'
    )

    # Create legend in first subplot
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linestyle=':', linewidth=2,
                   marker='+', markersize=8, label='Expected rebar'),
    ]
    for confidence in CONFIDENCE_LEVELS.keys():
        color = CONFIDENCE_COLORS[confidence]
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linewidth=2,
                       marker='o', markersize=5,
                       label=f'{confidence}%')
        )
    rebar_axes[0].legend(handles=legend_elements, loc='upper left', fontsize=7)

    # Draw statistics panel
    # Header
    ax_stats.text(0.5, 0.95, 'Error Summary (mm)',
                  ha='center', va='top', fontsize=12, fontweight='bold',
                  transform=ax_stats.transAxes)

    # Column headers
    col_headers = ['Conf.', 'Pos.Mean', 'Pos.Std', 'Pos.Max', 'Rad.Mean', 'N']
    y_pos = 0.88

    for i, header in enumerate(col_headers):
        x_pos = 0.02 + i * 0.16
        ax_stats.text(x_pos, y_pos, header, ha='left', va='top',
                      fontsize=8, fontweight='bold',
                      transform=ax_stats.transAxes)

    # Draw horizontal line
    ax_stats.plot([0.02, 0.98], [0.85, 0.85], color='black', linewidth=0.5,
                  transform=ax_stats.transAxes)

    # Data rows
    y_pos = 0.82
    for confidence in sorted(CONFIDENCE_LEVELS.keys()):
        if confidence not in stats:
            continue

        s = stats[confidence]
        color = CONFIDENCE_COLORS[confidence]

        row_data = [
            f'{confidence}%',
            f'{s["position_mean"]:.1f}',
            f'{s["position_std"]:.1f}',
            f'{s["position_max"]:.1f}',
            f'{s["radius_mean"]:.1f}',
            f'{s["count"]}',
        ]

        for i, val in enumerate(row_data):
            x_pos = 0.02 + i * 0.16
            ax_stats.text(x_pos, y_pos, val, ha='left', va='top',
                          fontsize=8, color=color if i == 0 else 'black',
                          fontweight='bold' if i == 0 else 'normal',
                          transform=ax_stats.transAxes)

        y_pos -= 0.05

    # Add summary statistics
    y_pos -= 0.05
    ax_stats.plot([0.02, 0.98], [y_pos + 0.02, y_pos + 0.02], color='black',
                  linewidth=0.5, transform=ax_stats.transAxes)

    ax_stats.text(0.5, y_pos - 0.02, 'Analysis Notes',
                  ha='center', va='top', fontsize=10, fontweight='bold',
                  transform=ax_stats.transAxes)

    notes = [
        f'- Expected radius: {expected_radius*1000:.1f} mm',
        f'- Total tracks: {len(df)}',
        f'- Scans: {df["scan"].nunique()}',
        f'- Crop margin: ±{crop_margin*1000:.0f} mm',
    ]

    y_pos -= 0.08
    for note in notes:
        ax_stats.text(0.02, y_pos, note, ha='left', va='top',
                      fontsize=8, transform=ax_stats.transAxes)
        y_pos -= 0.04

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return stats


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_all_plots(
    tracks_df: pd.DataFrame,
    cases: List[str],
    distances: List[int],
    output_dir: Path,
    show: bool = False,
) -> Dict:
    """
    Generate frame error comparison plots for all case/distance combinations.

    Args:
        tracks_df: DataFrame with all tracks data
        cases: List of cases to process
        distances: List of distances to process
        output_dir: Output directory for plots
        show: Whether to display plots

    Returns:
        Dictionary with all statistics
    """
    all_stats = {}

    for case in cases:
        all_stats[case] = {}

        for distance in distances:
            print(f"Processing {case} at {distance}cm...")

            output_path = output_dir / f'{case}_{distance}_frame_comparison.png'
            stats = plot_frame_error_comparison(
                case=case,
                distance=distance,
                tracks_df=tracks_df,
                output_path=output_path,
                show=show,
            )

            if stats:
                all_stats[case][distance] = stats

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Visualize frame-based rebar detection errors by distance'
    )
    parser.add_argument(
        '--cases', nargs='+', default=['case1', 'case2', 'case3'],
        help='Cases to include (default: case1 case2 case3)'
    )
    parser.add_argument(
        '--distances', nargs='+', type=int, default=[60, 120, 180, 240, 300, 360],
        help='Distances to include in cm (default: 60 120 180 240 300 360)'
    )
    parser.add_argument(
        '--output-dir', default='output/comparison_plots',
        help='Output directory for plots (default: output/comparison_plots)'
    )
    parser.add_argument(
        '--results-dir', default='output/results',
        help='Results directory containing tracks.csv files (default: output/results)'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Display plots interactively'
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    output_dir = script_dir / args.output_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tracks data...")
    df = load_all_tracks(results_dir, args.cases)

    if df.empty:
        print("No tracks data found.")
        return 1

    print(f"Loaded {len(df)} tracks from {df['case'].nunique()} cases")
    print(f"Distances: {sorted(df['distance'].unique())}")

    print("\nGenerating frame error comparison plots...")
    all_stats = generate_all_plots(
        tracks_df=df,
        cases=args.cases,
        distances=args.distances,
        output_dir=output_dir,
        show=args.show,
    )

    # Save summary JSON
    summary_path = output_dir / 'frame_error_summary.json'
    summary = {
        'confidence_levels': {
            str(k): v for k, v in CONFIDENCE_LEVELS.items()
        },
        'expected_radius_mm': EXPECTED_RADIUS * 1000,
        'statistics': convert_to_native_types(all_stats),
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    print(f"\nDone! Output files saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
