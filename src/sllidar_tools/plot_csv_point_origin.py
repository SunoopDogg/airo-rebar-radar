#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import sys
import math
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot LiDAR points from CSV with room layout (46Type) and pipes."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file exported from sllidar_tools",
    )
    parser.add_argument(
        "--min-range",
        type=float,
        default=0.05,
        help="Minimum valid range (meters)",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=10.0,
        help="Maximum valid range (meters)",
    )
    parser.add_argument(
        "--draw-room",
        dest="draw_room",
        action="store_true",
        help="Draw room layout and pipes as background",
    )
    parser.add_argument(
        "--lidar-x",
        type=float,
        default=0.0,
        help="LiDAR position X in room frame (meters)",
    )
    parser.add_argument(
        "--lidar-y",
        type=float,
        default=0.0,
        help="LiDAR position Y in room frame (meters)",
    )
    return parser.parse_args()


def load_points(csv_path, min_range, max_range):
    """
    CSV 형식:
      frame,idx,angle_rad,range_m,x,y
    x, y는 LiDAR 기준 좌표계.
    """
    xs = []
    ys = []
    frames = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        expected = ["frame", "idx", "angle_rad", "range_m", "x", "y"]

        if reader.fieldnames is None or any(col not in reader.fieldnames for col in expected):
            print(
                f"CSV 헤더가 예상과 다릅니다. 예상: {expected}, 실제: {reader.fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)

        for row in reader:
            try:
                r = float(row["range_m"])
                x = float(row["x"])
                y = float(row["y"])
                fr = row.get("frame")
            except (ValueError, TypeError):
                continue

            # range 필터링
            if r < min_range or r > max_range:
                continue

            xs.append(x)
            ys.append(y)
            frames.append(int(fr) if fr not in (None, "") else -1)

    return xs, ys, frames


def transform_points_rotate180(xs, ys, lidar_x, lidar_y):
    """
    LiDAR 좌표 → 방 좌표
    회전: 180도 (반전)

    x' = -x
    y' = -y
    """
    xs_world = []
    ys_world = []

    for x, y in zip(xs, ys):
        xr = -y
        yr = x

        xs_world.append(lidar_x + xr)
        ys_world.append(lidar_y + yr)

    return xs_world, ys_world


def draw_pipe(ax, cx, cy, inner_diam=0.040):
    r = inner_diam / 2.0
    circle = plt.Circle((cx, cy), r, fill=False, color="magenta", linewidth=1.2)
    ax.add_patch(circle)


def draw_room_background(ax):
    # room_len = 11.8
    # room_width = 3.3
    module_len = 0.5
    module_width = 0.29
    module_origin =[-0.25,1.105]

    # 방 외곽
    # outer = plt.Rectangle((0.0, 0.0), room_len, room_width, fill=False, linewidth=2.0)
    # ax.add_patch(outer)

    module = plt.Rectangle((module_origin[0],module_origin[1]), module_len, module_width, fill=False, linewidth=2.0)
    ax.add_patch(module)

    pipe_positions = [
        # 좌측 4개점
        (0.09, module_width - 0.08),
        (module_len - 0.09, module_width - 0.08),
        (0.09, 0.08),
        (module_len - 0.09, 0.08),



   
    ]

    for (px, py) in pipe_positions:
        px=module_origin[0]+px
        py=module_origin[1]+py
        draw_pipe(ax, px, py)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal", "box")


def draw_circle(ax, r, style, cx0, cy0):
    """LiDAR 중심(cx0, cy0) 기준으로 반경 r짜리 원을 그림."""
    ths = [t * math.pi / 180.0 for t in range(0, 360, 2)]
    xs = [cx0 + r * math.cos(t) for t in ths]
    ys = [cy0 + r * math.sin(t) for t in ths]
    ax.plot(xs, ys, style, linewidth=0.8, alpha=0.7)


def main():
    args = parse_args()

    print(f"Loading points from {args.csv} ...")
    xs, ys, frames = load_points(args.csv, args.min_range, args.max_range)
    print(f"Loaded {len(xs)} valid points")

    # LiDAR 180° 고정 회전 적용 + 방 좌표로 이동
    xs_w, ys_w = transform_points_rotate180(xs, ys, args.lidar_x, args.lidar_y)
    n = len(xs_w)

    fig, ax = plt.subplots()

    # 방 배경
    if args.draw_room:
        draw_room_background(ax)

    # LiDAR 위치 표시
    ax.scatter([args.lidar_x], [args.lidar_y], marker="x", s=80, label="LiDAR")
    ax.text(args.lidar_x, args.lidar_y, " LiDAR (rot 180°)", fontsize=9, va="bottom")

    # min/max range 원(옵션) – LiDAR를 중심으로 그림
    if args.min_range and args.min_range > 0:
        draw_circle(ax, args.min_range, "--", args.lidar_x, args.lidar_y)
    if args.max_range and args.max_range > 0:
        draw_circle(ax, args.max_range, "-.", args.lidar_x, args.lidar_y)

    # 스캔 포인트(방 좌표계, 180도 회전 반영됨)
    sc = ax.scatter(xs_w, ys_w, s=6, picker=True)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"LiDAR Scan (180° rotated, valid={n})")
    ax.grid(True)
    ax.set_aspect("equal", "box")

    # 화면 내 유효 포인트 수 표시용 텍스트 박스
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )

    def update_inview_count(*_):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cnt = 0
        for x, y in zip(xs_w, ys_w):
            if x0 <= x <= x1 and y0 <= y <= y1:
                cnt += 1
        info_text.set_text(f"in-view valid: {cnt} / {n}")
        fig.canvas.draw_idle()

    # 확대/이동 이벤트에 연결
    fig.canvas.mpl_connect("draw_event", update_inview_count)
    ax.callbacks.connect("xlim_changed", update_inview_count)
    ax.callbacks.connect("ylim_changed", update_inview_count)

    # 클릭한 점 하이라이트 & 좌표 표시용
    sel_dot, = ax.plot([], [], marker="o", markersize=8, fillstyle="none", linewidth=1.0)

    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6),
        arrowprops=dict(arrowstyle="->"),
    )
    ann.set_visible(False)

    def onclick(event):
        if event.inaxes != ax or n == 0 or event.xdata is None or event.ydata is None:
            return

        cx, cy = event.xdata, event.ydata

        # 최근접 점 찾기 (방 좌표계에서)
        min_i = 0
        min_d2 = (xs_w[0] - cx) ** 2 + (ys_w[0] - cy) ** 2
        for i in range(1, n):
            d2 = (xs_w[i] - cx) ** 2 + (ys_w[i] - cy) ** 2
            if d2 < min_d2:
                min_d2, min_i = d2, i

        px, py = xs_w[min_i], ys_w[min_i]
        fidx = frames[min_i] if 0 <= min_i < len(frames) else -1

        # 하이라이트 & 주석
        sel_dot.set_data([px], [py])
        ann.xy = (px, py)
        ann.set_text(f"({px:.3f}, {py:.3f})  f{fidx if fidx >= 0 else '?'}")
        ann.set_visible(True)
        fig.canvas.draw_idle()

        print(
            f"Clicked nearest point: x={px:.3f}, y={py:.3f}, frame=f{fidx if fidx>=0 else '?'}"
        )

    fig.canvas.mpl_connect("button_press_event", onclick)

    # 초기 in-view 카운트 갱신
    update_inview_count()
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
