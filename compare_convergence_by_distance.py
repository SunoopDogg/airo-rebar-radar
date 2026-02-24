#!/usr/bin/env python3
"""
Structure Overlay Visualization by Distance

Generates cropped rebar overlay plots showing detected positions from all distances
(60, 120, 180, 240, 300, 360 cm) for each case.
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.structure.config import (
    create_ppvc_linear_config,
    create_ppvc_cluster_2_config,
    create_ppvc_cluster_4_config,
)
from src.visualization.structure_renderer import StructureRenderer

# Center X mapping by lidar distance (cm) - from analyze_results.py
CENTER_X_MAPPING_LINEAR = {
    60: -0.617, 120: -1.204, 180: -1.797,
    240: -2.407, 300: -3.013, 360: -3.618
}
CENTER_X_MAPPING_CLUSTER = {
    60: -0.689, 120: -1.282, 180: -1.896,
    240: -2.495, 300: -3.100, 360: -3.709
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

# Distance color mapping (cool to warm gradient)
DISTANCE_COLORS = {
    60: '#1f77b4',    # Blue
    120: '#2ca02c',   # Green
    180: '#ff7f0e',   # Orange
    240: '#d62728',   # Red
    300: '#9467bd',   # Purple
    360: '#8c564b',   # Brown
}

# Distance line style mapping (for colorblind accessibility)
DISTANCE_LINESTYLES = {
    60: '-',      # Solid
    120: '--',    # Dashed
    180: '-.',    # Dash-dot
    240: ':',     # Dotted
    300: '-',     # Solid
    360: '--',    # Dashed
}

# Distance marker mapping (for colorblind accessibility)
DISTANCE_MARKERS = {
    60: 'o',      # Circle
    120: 's',     # Square
    180: '^',     # Triangle up
    240: 'D',     # Diamond
    300: 'v',     # Triangle down
    360: 'p',     # Pentagon
}


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


def plot_structure_overlay_cropped(
    tracks_df: pd.DataFrame,
    case: str,
    output_path: Optional[Path] = None,
    show: bool = False,
    crop_margin: float = 0.15,  # meters
) -> None:
    """
    Plot cropped zoom views of each rebar with detected positions from all distances.

    Creates a subplot for each expected rebar position, showing a zoomed-in view
    of the expected position and nearby detections.

    Args:
        tracks_df: DataFrame containing tracks data with columns:
                   track_id, center_x, center_y, radius, case, scan, distance
        case: Case identifier ('case1', 'case2', or 'case3')
        output_path: Path to save the plot
        show: Whether to display the plot
        crop_margin: Margin around each rebar for cropping (meters)
    """
    # Filter to specific case
    case_df = tracks_df[tracks_df['case'] == case].copy()

    if case_df.empty:
        print(f"No data for {case}")
        return

    # Get structure type and reference distance (180cm as middle value)
    structure_type = STRUCTURE_TYPE[case]
    reference_distance = 180

    # Get center positions for reference distance
    if structure_type == 'LINEAR':
        center_x = CENTER_X_MAPPING_LINEAR[reference_distance]
    else:
        center_x = CENTER_X_MAPPING_CLUSTER[reference_distance]
    center_y = CENTER_Y[structure_type]

    # Create structure config for this case
    if case == 'case1':
        config = create_ppvc_linear_config(center_x=center_x, center_y=center_y)
    elif case == 'case2':
        config = create_ppvc_cluster_2_config(center_x=center_x, center_y=center_y)
    else:  # case3
        config = create_ppvc_cluster_4_config(center_x=center_x, center_y=center_y)

    # Get expected rebar positions
    expected_positions = config.get_track_positions()
    n_rebars = len(expected_positions)

    # Determine subplot layout
    if n_rebars <= 4:  # LINEAR, CLUSTER_2
        rows, cols = 2, 2
        figsize = (12, 10)
    else:  # CLUSTER_4 (8 rebars)
        rows, cols = 2, 4
        figsize = (20, 10)

    # Sort rebar positions for visual consistency (by Y, then by X)
    sorted_positions = sorted(expected_positions, key=lambda p: (p[1], p[0]))

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=150)
    axes = axes.flatten()

    # Get wall offset for coordinate transformation
    if structure_type == 'LINEAR':
        wall_offset = -1.85
    else:
        wall_offset = -1.775

    # Get rebar radius for drawing
    rebar_radius = config.track_diameter / 2

    distances = sorted(case_df['distance'].unique())

    for idx, (exp_x, exp_y) in enumerate(sorted_positions):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Draw expected rebar position (blue cross and circle)
        ax.plot(exp_x, exp_y, 'b+', markersize=15, markeredgewidth=2,
                label='Expected position')
        expected_circle = plt.Circle(
            (exp_x, exp_y),
            rebar_radius,
            fill=False,
            edgecolor='blue',
            linewidth=1.5,
            linestyle=':',
            alpha=0.8,
        )
        ax.add_patch(expected_circle)

        # Plot detected positions by distance
        for distance in distances:
            dist_df = case_df[case_df['distance'] == distance]
            color = DISTANCE_COLORS.get(distance, 'gray')

            # Calculate offset to align with reference distance
            if structure_type == 'LINEAR':
                offset_x = CENTER_X_MAPPING_LINEAR[reference_distance] - CENTER_X_MAPPING_LINEAR[distance]
            else:
                offset_x = CENTER_X_MAPPING_CLUSTER[reference_distance] - CENTER_X_MAPPING_CLUSTER[distance]

            # Filter tracks within crop range and calculate average
            tracks_in_range = []
            for _, row in dist_df.iterrows():
                cx = row['center_x'] + offset_x + wall_offset
                cy = row['center_y']
                radius = row['radius']

                # Check if detection is within crop margin of this rebar
                if (abs(cx - exp_x) <= crop_margin and
                    abs(cy - exp_y) <= crop_margin):
                    tracks_in_range.append({'cx': cx, 'cy': cy, 'radius': radius})

            # Plot average position if tracks found
            if tracks_in_range:
                avg_cx = np.mean([t['cx'] for t in tracks_in_range])
                avg_cy = np.mean([t['cy'] for t in tracks_in_range])
                avg_radius = np.mean([t['radius'] for t in tracks_in_range])

                linestyle = DISTANCE_LINESTYLES.get(distance, '-')
                marker = DISTANCE_MARKERS.get(distance, 'o')

                # Draw circle for averaged detected rebar
                circle = plt.Circle(
                    (avg_cx, avg_cy),
                    avg_radius,
                    fill=False,
                    edgecolor=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax.add_patch(circle)

                # Draw center marker
                ax.plot(avg_cx, avg_cy, marker, color=color, markersize=5, alpha=0.8)

        # Set axis limits (crop around rebar)
        ax.set_xlim(exp_x - crop_margin, exp_x + crop_margin)
        ax.set_ylim(exp_y - crop_margin, exp_y + crop_margin)

        # Set axis properties
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(f'Rebar {idx + 1}\n({exp_x:.3f}, {exp_y:.3f})', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_rebars, len(axes)):
        axes[idx].set_visible(False)

    # Create legend elements for the figure
    legend_elements = [
        plt.Line2D([0], [0], color='blue', marker='+', markersize=10,
                   linestyle='None', markeredgewidth=2, label='Expected position'),
    ]
    for distance in distances:
        color = DISTANCE_COLORS.get(distance, 'gray')
        linestyle = DISTANCE_LINESTYLES.get(distance, '-')
        marker = DISTANCE_MARKERS.get(distance, 'o')
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2,
                       marker=marker, markersize=6, label=f'{distance} cm')
        )

    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.99), fontsize=9)

    fig.suptitle(f'Cropped Rebar View - {CASE_TITLES.get(case, case)}\n'
                 f'Crop margin: ±{crop_margin*100:.0f}cm per rebar',
                 fontsize=12, y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_structure_overlay_full(
    tracks_df: pd.DataFrame,
    case: str,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Plot full structure view with detected positions from all distances overlaid.
    """
    # Filter to specific case
    case_df = tracks_df[tracks_df['case'] == case].copy()

    if case_df.empty:
        print(f"No data for {case}")
        return

    # Get structure type and reference distance
    structure_type = STRUCTURE_TYPE[case]
    reference_distance = 180

    # Get center positions for reference distance
    if structure_type == 'LINEAR':
        center_x = CENTER_X_MAPPING_LINEAR[reference_distance]
    else:
        center_x = CENTER_X_MAPPING_CLUSTER[reference_distance]
    center_y = CENTER_Y[structure_type]

    # Create structure config
    if case == 'case1':
        config = create_ppvc_linear_config(center_x=center_x, center_y=center_y)
    elif case == 'case2':
        config = create_ppvc_cluster_2_config(center_x=center_x, center_y=center_y)
    else:
        config = create_ppvc_cluster_4_config(center_x=center_x, center_y=center_y)

    # Get wall offset for coordinate transformation of detected positions
    if structure_type == 'LINEAR':
        wall_offset = -1.85
    else:
        wall_offset = -1.775

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Use StructureRenderer for consistent visualization
    renderer = StructureRenderer(config)
    renderer.draw_boundary(ax)
    renderer.draw_expected_rebars(ax)

    # Get expected positions for averaging detected positions
    expected_positions = config.get_track_positions()

    # Plot detected positions by distance (averaged per rebar)
    distances = sorted(case_df['distance'].unique())
    crop_margin = 0.15  # meters - same as cropped view

    for distance in distances:
        dist_df = case_df[case_df['distance'] == distance]
        color = DISTANCE_COLORS.get(distance, 'gray')
        marker = DISTANCE_MARKERS.get(distance, 'o')

        # Calculate offset to align with reference distance
        if structure_type == 'LINEAR':
            offset_x = CENTER_X_MAPPING_LINEAR[reference_distance] - CENTER_X_MAPPING_LINEAR[distance]
        else:
            offset_x = CENTER_X_MAPPING_CLUSTER[reference_distance] - CENTER_X_MAPPING_CLUSTER[distance]

        # For each expected rebar position, find nearby tracks and average
        for exp_x, exp_y in expected_positions:
            tracks_in_range = []
            for _, row in dist_df.iterrows():
                cx = row['center_x'] + offset_x + wall_offset
                cy = row['center_y']

                # Check if detection is within crop margin of this rebar
                if (abs(cx - exp_x) <= crop_margin and
                    abs(cy - exp_y) <= crop_margin):
                    tracks_in_range.append({'cx': cx, 'cy': cy})

            # Plot average position if tracks found
            if tracks_in_range:
                avg_cx = np.mean([t['cx'] for t in tracks_in_range])
                avg_cy = np.mean([t['cy'] for t in tracks_in_range])
                ax.plot(avg_cx, avg_cy, marker, color=color, markersize=8, alpha=0.8)

    # Set axis properties
    ax.set_xlim(-4.2, 0.2)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Concrete boundary'),
        plt.Line2D([0], [0], color='blue', linestyle=':', linewidth=2,
                   marker='+', markersize=8, markeredgewidth=2, label='Expected rebar'),
    ]
    for distance in distances:
        color = DISTANCE_COLORS.get(distance, 'gray')
        marker = DISTANCE_MARKERS.get(distance, 'o')
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linestyle='None',
                       marker=marker, markersize=8, label=f'{distance} cm')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Title
    ax.set_title(f'Structure Overlay - {CASE_TITLES.get(case, case)}\n'
                 f'Reference: {reference_distance}cm, All scans overlaid', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate structure overlay plots showing detections from all distances'
    )
    parser.add_argument(
        '--cases', nargs='+', default=['case1', 'case2', 'case3'],
        help='Cases to include (default: case1 case2 case3)'
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

    print("\nGenerating structure overlay plots...")
    for case in args.cases:
        # Full structure overlay
        plot_structure_overlay_full(
            df,
            case=case,
            output_path=output_dir / f'structure_overlay_{case}.png',
            show=args.show,
        )
        # Cropped rebar view
        plot_structure_overlay_cropped(
            df,
            case=case,
            output_path=output_dir / f'structure_overlay_{case}_cropped.png',
            show=args.show,
        )

    print("\nDone! Output files saved to:", output_dir)
    return 0


if __name__ == '__main__':
    exit(main())
