#!/usr/bin/env python3
"""
Rebar Detection Analysis Report Generator

Analyzes tracks.csv files from rebar detection results and generates
an error analysis report comparing detected positions with actual rebar positions.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment


# Center X mapping by lidar distance (cm)
# Based on actual detected positions (lidar relative coordinate system)
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

# Expected rebar counts
EXPECTED_REBAR_COUNT = {
    'case1': 4,  # LINEAR
    'case2': 4,  # CLUSTER_2
    'case3': 8   # CLUSTER_4
}

STRUCTURE_TYPE = {
    'case1': 'LINEAR',
    'case2': 'CLUSTER_2',
    'case3': 'CLUSTER_4'
}


def get_actual_rebar_positions(case: str, distance: int) -> np.ndarray:
    """
    Calculate actual rebar positions based on case type and lidar distance.

    Returns:
        np.ndarray: Array of shape (n_rebars, 2) with (x, y) positions
    """
    structure = STRUCTURE_TYPE[case]

    if structure == 'LINEAR':
        center_x = CENTER_X_MAPPING_LINEAR[distance]
        center_y = CENTER_Y['LINEAR']
        # 4 rebars at y = [-1.5, -0.5, 0.5, 1.5] (1m interval)
        y_offsets = [-1.5, -0.5, 0.5, 1.5]
        positions = np.array([[center_x, center_y + y] for y in y_offsets])

    elif structure == 'CLUSTER_2':
        center_x = CENTER_X_MAPPING_CLUSTER[distance]
        center_y = CENTER_Y['CLUSTER_2']
        # 2 clusters at y = [-1.55, 1.55], each with 2 rebars at x = ±0.075
        cluster_y = [-1.55, 1.55]
        x_offsets = [-0.075, 0.075]
        positions = []
        for cy in cluster_y:
            for xo in x_offsets:
                positions.append([center_x + xo, center_y + cy])
        positions = np.array(positions)

    elif structure == 'CLUSTER_4':
        center_x = CENTER_X_MAPPING_CLUSTER[distance]
        center_y = CENTER_Y['CLUSTER_4']
        # 2 clusters at y = [-1.55, 1.55], each with 2x2 grid (±0.075 in x and y)
        cluster_y = [-1.55, 1.55]
        xy_offsets = [(-0.075, -0.075), (-0.075, 0.075), (0.075, -0.075), (0.075, 0.075)]
        positions = []
        for cy in cluster_y:
            for xo, yo in xy_offsets:
                positions.append([center_x + xo, center_y + cy + yo])
        positions = np.array(positions)

    return positions


def load_tracks_csv(filepath: str) -> pd.DataFrame:
    """Load tracks.csv file and return dataframe with essential columns."""
    df = pd.read_csv(filepath)
    return df[['track_id', 'center_x', 'center_y', 'radius', 'hits']].copy()


def calculate_distance_error(detected: np.ndarray, actual: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate optimal matching between detected and actual positions using Hungarian algorithm.

    Returns:
        matched_errors: Array of matched distance errors
        unmatched_actual_indices: Indices of actual rebars that were not matched
        mean_error: Mean error of matched pairs
    """
    if len(detected) == 0:
        return np.array([]), np.arange(len(actual)), np.nan

    # Build cost matrix (distance between each detected-actual pair)
    cost_matrix = np.zeros((len(detected), len(actual)))
    for i, d in enumerate(detected):
        for j, a in enumerate(actual):
            cost_matrix[i, j] = np.sqrt((d[0] - a[0])**2 + (d[1] - a[1])**2)

    # Use Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_errors = cost_matrix[row_ind, col_ind]
    unmatched_actual = set(range(len(actual))) - set(col_ind)

    return matched_errors, np.array(list(unmatched_actual)), np.mean(matched_errors)


def parse_filename(filename: str) -> Tuple[str, int, int]:
    """
    Parse filename to extract case, test number, and distance.
    Example: S3_1-1_120_tracks.csv -> ('case1', 1, 120)
    """
    match = re.match(r'S3_(\d)-(\d)_(\d+)_tracks\.csv', filename)
    if match:
        case_num = int(match.group(1))
        test_num = int(match.group(2))
        distance = int(match.group(3))
        return f'case{case_num}', test_num, distance
    return None, None, None


def analyze_all_results(results_dir: str) -> pd.DataFrame:
    """
    Analyze all tracks.csv files and return analysis results.
    """
    results = []

    for case_folder in ['case1', 'case2', 'case3']:
        case_path = Path(results_dir) / case_folder
        if not case_path.exists():
            continue

        for subdir in case_path.iterdir():
            if not subdir.is_dir():
                continue

            tracks_file = subdir / f'{subdir.name}_tracks.csv'
            if not tracks_file.exists():
                continue

            case, test_num, distance = parse_filename(tracks_file.name)
            if case is None:
                continue

            # Load detected positions
            df = load_tracks_csv(tracks_file)
            detected = df[['center_x', 'center_y']].values

            # Get actual positions
            actual = get_actual_rebar_positions(case, distance)

            # Calculate errors
            matched_errors, unmatched_indices, mean_error = calculate_distance_error(detected, actual)

            expected_count = EXPECTED_REBAR_COUNT[case]
            detected_count = len(detected)
            matched_count = len(matched_errors)
            detection_rate = (matched_count / expected_count) * 100 if expected_count > 0 else 0

            results.append({
                'case': case,
                'structure': STRUCTURE_TYPE[case],
                'test': test_num,
                'distance': distance,
                'expected_count': expected_count,
                'detected_count': detected_count,
                'matched_count': matched_count,
                'detection_rate': detection_rate,
                'mean_error': mean_error if not np.isnan(mean_error) else None,
                'max_error': np.max(matched_errors) if len(matched_errors) > 0 else None,
                'min_error': np.min(matched_errors) if len(matched_errors) > 0 else None,
                'std_error': np.std(matched_errors) if len(matched_errors) > 0 else None,
                'errors': matched_errors.tolist() if len(matched_errors) > 0 else [],
            })

    return pd.DataFrame(results)


def generate_report(df: pd.DataFrame, output_path: str):
    """Generate markdown report from analysis results."""

    report = []
    report.append("# 철근 검출 분석 보고서")
    report.append(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overview
    report.append("## 1. 개요\n")
    report.append("본 보고서는 라이더 기반 철근 검출 시스템의 성능을 분석합니다.")
    report.append("각 케이스별로 검출된 철근과 실제 철근 위치 간의 오차를 측정하고,")
    report.append("라이더 거리에 따른 검출 안정성을 평가합니다.\n")

    # Measurement conditions
    report.append("### 측정 조건\n")
    report.append("| 항목 | 값 |")
    report.append("|------|-----|")
    report.append("| 케이스 수 | 3 (LINEAR, CLUSTER_2, CLUSTER_4) |")
    report.append("| 테스트 반복 | 5회 |")
    report.append("| 라이더 거리 | 60, 120, 180, 240, 300, 360 cm |")
    report.append("| 총 측정 횟수 | 90회 (3 × 5 × 6) |\n")

    # Case descriptions
    report.append("### 케이스별 구조물 설명\n")
    report.append("| 케이스 | 구조 유형 | 예상 철근 수 | 배치 설명 |")
    report.append("|--------|----------|-------------|-----------|")
    report.append("| Case 1 | LINEAR | 4 | 4개 철근이 1m 간격으로 일렬 배치 |")
    report.append("| Case 2 | CLUSTER_2 | 4 | 2개 클러스터 × 2개 철근 (150mm 간격) |")
    report.append("| Case 3 | CLUSTER_4 | 8 | 2개 클러스터 × 4개 철근 (2×2 격자, 150mm 간격) |\n")

    # Overall summary
    report.append("## 2. 전체 요약\n")

    total_expected = df['expected_count'].sum()
    total_matched = df['matched_count'].sum()
    overall_detection_rate = (total_matched / total_expected) * 100

    all_errors = []
    for errors in df['errors']:
        all_errors.extend(errors)
    all_errors = np.array(all_errors)

    report.append("### 전체 통계\n")
    report.append("| 항목 | 값 |")
    report.append("|------|-----|")
    report.append(f"| 총 예상 철근 수 | {total_expected} |")
    report.append(f"| 총 매칭된 철근 수 | {total_matched} |")
    report.append(f"| 전체 검출률 | {overall_detection_rate:.1f}% |")
    if len(all_errors) > 0:
        report.append(f"| 평균 오차 | {np.mean(all_errors)*1000:.2f} mm |")
        report.append(f"| 최대 오차 | {np.max(all_errors)*1000:.2f} mm |")
        report.append(f"| 최소 오차 | {np.min(all_errors)*1000:.2f} mm |")
        report.append(f"| 표준편차 | {np.std(all_errors)*1000:.2f} mm |")
    report.append("")

    # Case-by-case analysis
    report.append("## 3. 케이스별 상세 분석\n")

    for case in ['case1', 'case2', 'case3']:
        case_df = df[df['case'] == case]
        if case_df.empty:
            continue

        structure = case_df['structure'].iloc[0]
        expected = case_df['expected_count'].iloc[0]

        report.append(f"### {case.upper()} ({structure})\n")
        report.append(f"**예상 철근 수:** {expected}개\n")

        # Summary by distance
        report.append("#### 거리별 검출 결과\n")
        report.append("| 거리 (cm) | 검출률 (%) | 평균 오차 (mm) | 최대 오차 (mm) | 표준편차 (mm) |")
        report.append("|-----------|------------|----------------|----------------|---------------|")

        for distance in sorted(case_df['distance'].unique()):
            dist_df = case_df[case_df['distance'] == distance]
            avg_detection = dist_df['detection_rate'].mean()

            dist_errors = []
            for errors in dist_df['errors']:
                dist_errors.extend(errors)
            dist_errors = np.array(dist_errors)

            if len(dist_errors) > 0:
                mean_err = np.mean(dist_errors) * 1000
                max_err = np.max(dist_errors) * 1000
                std_err = np.std(dist_errors) * 1000
                report.append(f"| {distance} | {avg_detection:.1f} | {mean_err:.2f} | {max_err:.2f} | {std_err:.2f} |")
            else:
                report.append(f"| {distance} | {avg_detection:.1f} | - | - | - |")

        report.append("")

        # Detailed table by test
        report.append("#### 테스트별 상세 결과\n")
        report.append("| 테스트 | 거리 | 검출 수 | 검출률 | 평균 오차 (mm) |")
        report.append("|--------|------|---------|--------|----------------|")

        for _, row in case_df.sort_values(['distance', 'test']).iterrows():
            mean_err = f"{row['mean_error']*1000:.2f}" if row['mean_error'] is not None else "-"
            report.append(f"| {row['test']} | {row['distance']} | {row['matched_count']}/{row['expected_count']} | {row['detection_rate']:.1f}% | {mean_err} |")

        report.append("")

        # Case summary
        case_errors = []
        for errors in case_df['errors']:
            case_errors.extend(errors)
        case_errors = np.array(case_errors)

        total_matched_case = case_df['matched_count'].sum()
        total_expected_case = case_df['expected_count'].sum()
        case_detection_rate = (total_matched_case / total_expected_case) * 100

        report.append(f"**{case.upper()} 요약:**")
        report.append(f"- 전체 검출률: {case_detection_rate:.1f}%")
        if len(case_errors) > 0:
            report.append(f"- 평균 오차: {np.mean(case_errors)*1000:.2f} mm")
            report.append(f"- 오차 표준편차: {np.std(case_errors)*1000:.2f} mm")
        report.append("")

    # Distance-based comparison
    report.append("## 4. 라이더 거리별 비교 분석\n")

    report.append("### 거리별 전체 검출률\n")
    report.append("| 거리 (cm) | Case1 검출률 | Case2 검출률 | Case3 검출률 | 전체 검출률 |")
    report.append("|-----------|--------------|--------------|--------------|-------------|")

    for distance in sorted(df['distance'].unique()):
        dist_df = df[df['distance'] == distance]
        rates = []
        for case in ['case1', 'case2', 'case3']:
            case_dist_df = dist_df[dist_df['case'] == case]
            if not case_dist_df.empty:
                rate = case_dist_df['detection_rate'].mean()
                rates.append(f"{rate:.1f}%")
            else:
                rates.append("-")

        overall_rate = dist_df['detection_rate'].mean()
        report.append(f"| {distance} | {rates[0]} | {rates[1]} | {rates[2]} | {overall_rate:.1f}% |")

    report.append("")

    report.append("### 거리별 평균 오차\n")
    report.append("| 거리 (cm) | Case1 오차 (mm) | Case2 오차 (mm) | Case3 오차 (mm) | 전체 오차 (mm) |")
    report.append("|-----------|-----------------|-----------------|-----------------|----------------|")

    for distance in sorted(df['distance'].unique()):
        dist_df = df[df['distance'] == distance]
        errors_by_case = []

        for case in ['case1', 'case2', 'case3']:
            case_dist_df = dist_df[dist_df['case'] == case]
            case_errors = []
            for errors in case_dist_df['errors']:
                case_errors.extend(errors)
            if len(case_errors) > 0:
                errors_by_case.append(f"{np.mean(case_errors)*1000:.2f}")
            else:
                errors_by_case.append("-")

        all_dist_errors = []
        for errors in dist_df['errors']:
            all_dist_errors.extend(errors)
        if len(all_dist_errors) > 0:
            overall_err = f"{np.mean(all_dist_errors)*1000:.2f}"
        else:
            overall_err = "-"

        report.append(f"| {distance} | {errors_by_case[0]} | {errors_by_case[1]} | {errors_by_case[2]} | {overall_err} |")

    report.append("")

    # Test consistency analysis
    report.append("## 5. 측정 일관성 분석\n")
    report.append("동일 조건(케이스 + 거리)에서 5회 반복 측정의 표준편차를 분석합니다.\n")

    report.append("### 검출률 일관성 (5회 측정 표준편차)\n")
    report.append("| 케이스 | 60cm | 120cm | 180cm | 240cm | 300cm | 360cm |")
    report.append("|--------|------|-------|-------|-------|-------|-------|")

    for case in ['case1', 'case2', 'case3']:
        case_df = df[df['case'] == case]
        stds = []
        for distance in [60, 120, 180, 240, 300, 360]:
            dist_df = case_df[case_df['distance'] == distance]
            if not dist_df.empty:
                std = dist_df['detection_rate'].std()
                stds.append(f"{std:.1f}%")
            else:
                stds.append("-")
        report.append(f"| {case.upper()} | {' | '.join(stds)} |")

    report.append("")

    # Conclusions
    report.append("## 6. 결론 및 권장사항\n")

    # Find best and worst performing distances
    dist_performance = []
    for distance in sorted(df['distance'].unique()):
        dist_df = df[df['distance'] == distance]
        rate = dist_df['detection_rate'].mean()
        dist_performance.append((distance, rate))

    best_dist = max(dist_performance, key=lambda x: x[1])
    worst_dist = min(dist_performance, key=lambda x: x[1])

    report.append("### 주요 발견 사항\n")
    report.append(f"1. **최적 라이더 거리:** {best_dist[0]}cm (검출률 {best_dist[1]:.1f}%)")
    report.append(f"2. **최저 검출률 거리:** {worst_dist[0]}cm (검출률 {worst_dist[1]:.1f}%)")

    # Case performance comparison
    case_performance = []
    for case in ['case1', 'case2', 'case3']:
        case_df = df[df['case'] == case]
        if not case_df.empty:
            rate = case_df['detection_rate'].mean()
            case_performance.append((case, STRUCTURE_TYPE[case], rate))

    best_case = max(case_performance, key=lambda x: x[2])
    worst_case = min(case_performance, key=lambda x: x[2])

    report.append(f"3. **최고 검출률 구조:** {best_case[1]} (평균 {best_case[2]:.1f}%)")
    report.append(f"4. **최저 검출률 구조:** {worst_case[1]} (평균 {worst_case[2]:.1f}%)")

    if len(all_errors) > 0:
        report.append(f"5. **전체 위치 오차:** 평균 {np.mean(all_errors)*1000:.2f}mm, 최대 {np.max(all_errors)*1000:.2f}mm")

    report.append("")

    report.append("### 권장사항\n")
    report.append(f"1. 최적의 검출 성능을 위해 라이더 거리 **{best_dist[0]}cm** 권장")
    if worst_dist[1] < 80:
        report.append(f"2. {worst_dist[0]}cm 거리에서 검출률이 낮으므로 해당 거리에서의 측정 시 주의 필요")
    if worst_case[2] < 80:
        report.append(f"3. {worst_case[1]} 구조물은 검출률이 낮으므로 추가적인 알고리즘 개선 검토 필요")
    report.append("")

    report.append("---\n")
    report.append("*본 보고서는 자동 생성되었습니다.*")

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Report generated: {output_path}")


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'output' / 'results'
    output_path = script_dir / 'output' / 'analysis_report.md'

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("Analyzing rebar detection results...")
    df = analyze_all_results(results_dir)

    if df.empty:
        print("No results found to analyze.")
        return

    print(f"Analyzed {len(df)} measurements")
    print(f"Cases: {df['case'].unique()}")
    print(f"Distances: {sorted(df['distance'].unique())}")

    generate_report(df, output_path)

    # Print summary
    print("\n=== Summary ===")
    for case in ['case1', 'case2', 'case3']:
        case_df = df[df['case'] == case]
        if not case_df.empty:
            rate = case_df['detection_rate'].mean()
            print(f"{case}: Average detection rate = {rate:.1f}%")


if __name__ == '__main__':
    main()
