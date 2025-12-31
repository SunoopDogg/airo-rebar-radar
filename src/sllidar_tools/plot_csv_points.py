#!/usr/bin/env python3
import argparse
import csv
import math
import matplotlib.pyplot as plt

def load_points(csv_path):
    xs, ys, frames = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = row.get('x'); x = row.get('y'); fr = row.get('frame')
            if not x or not y:
                continue
            xs.append(float(x))
            ys.append(-float(y))
            frames.append(int(fr) if fr is not None and fr != '' else -1)
    return xs, ys, frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='scan_capture_duration로 생성한 CSV 경로')
    parser.add_argument('--duration', type=float, default=None, help='캡처한 샘플링 시간(sec), 제목에 표시용(옵션)')
    parser.add_argument('--min-range', type=float, default=None, help='표시용: 최소 범위 원(옵션)')
    parser.add_argument('--max-range', type=float, default=None, help='표시용: 최대 범위 원(옵션)')
    args = parser.parse_args()

    xs, ys, frames = load_points(args.csv)
    n = len(xs)
    print(f"Loaded {n} valid points from {args.csv}")

    fig, ax = plt.subplots()
    sc = ax.scatter(xs, ys, s=6, picker=True)  # 유효 포인트 산점도
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # 제목: duration 있으면 함께 표기
    if args.duration is not None:
        ax.set_title(f'LiDAR Points (valid={n}, duration={args.duration:.2f}s)')
    else:
        ax.set_title(f'LiDAR Points (valid={n})')

    # (0,0)에 LiDAR 표기 + 축선
    ax.scatter([0], [0], s=60, marker='*', label='LiDAR (0,0)')
    ax.axhline(0, linewidth=0.8, alpha=0.6)
    ax.axvline(0, linewidth=0.8, alpha=0.6)

    # 선택된 점을 하이라이트할 임시 아티스트
    sel_dot, = ax.plot([], [], marker='o', markersize=8, fillstyle='none', linewidth=1.0)

    # 범위 원(옵션)
    def draw_circle(r, style):
        th = [t*math.pi/180 for t in range(0, 360, 2)]
        cx = [r*math.cos(t) for t in th]
        cy = [r*math.sin(t) for t in th]
        ax.plot(cx, cy, style, linewidth=0.8, alpha=0.7)

    if args.min_range and args.min_range > 0:
        draw_circle(args.min_range, '--')
    if args.max_range and args.max_range > 0:
        draw_circle(args.max_range, '-.')

    # 화면 내 유효 포인트 수 표시용 텍스트 박스
    info_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
    )

    def update_inview_count(*_):
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        cnt = 0
        # 간단한 O(N) 계산 (필요 시 공간 인덱스 적용 가능)
        for x, y in zip(xs, ys):
            if x0 <= x <= x1 and y0 <= y <= y1:
                cnt += 1
        info_text.set_text(f'in-view valid: {cnt} / {n}')
        fig.canvas.draw_idle()

    # 확대/이동 후 갱신: draw_event와 축 변경 콜백에 연결
    fig.canvas.mpl_connect('draw_event', update_inview_count)
    ax.callbacks.connect('xlim_changed', update_inview_count)
    ax.callbacks.connect('ylim_changed', update_inview_count)

 
    ann = ax.annotate(
        '', xy=(0, 0), xytext=(10, 10), textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6),
        arrowprops=dict(arrowstyle='->')
    )
    ann.set_visible(False)

    def onclick(event):
        if event.inaxes != ax or n == 0 or event.xdata is None or event.ydata is None:
            return
        cx, cy = event.xdata, event.ydata
        # 최근접점 탐색
        min_i, min_d2 = 0, (xs[0]-cx)**2 + (ys[0]-cy)**2
        for i in range(1, n):
            d2 = (xs[i]-cx)**2 + (ys[i]-cy)**2
            if d2 < min_d2:
                min_d2, min_i = d2, i
        px, py = xs[min_i], ys[min_i]
        fidx = frames[min_i] if 0 <= min_i < len(frames) else -1
        # 하이라이트 & 주석
        sel_dot.set_data([px], [py])
        ann.xy = (px, py)
        ann.set_text(f"({px:.3f}, {py:.3f})  f{fidx if fidx>=0 else '?'}")
        ann.set_visible(True)
        fig.canvas.draw_idle()
        print(f"Clicked nearest point: x={px:.3f}, y={py:.3f}, frame=f{fidx if fidx>=0 else '?'}")

    fig.canvas.mpl_connect('button_press_event', onclick)

    # 초기 in-view 카운트 갱신
    update_inview_count()
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
