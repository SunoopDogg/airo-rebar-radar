# AIRO Rebar Radar

LIDAR 기반 철근 중심축 검출 시스템. LIDAR 포인트 클라우드 데이터를 처리하여 콘크리트 구조물 내 철근 위치를 5단계 파이프라인으로 검출합니다: 전처리 → 클러스터링 → 원 피팅 → 시간 필터링 → 시각화

## 요구사항

- Python 3.14+
- 주요 의존성: numpy, pandas, scikit-learn, matplotlib, filterpy

---

## 환경 설정

### 1. uv (권장)

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

### 2. Conda

```bash
# 환경 생성
conda create -n airo-rebar python=3.14 -y
conda activate airo-rebar

# 의존성 설치
pip install -e .
```

### 3. Vanilla Python (venv)

```bash
# 가상환경 생성
python3.14 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -e .
```

### 4. Docker

```bash
# CPU 모드
docker compose up -d
docker exec -it airo-rebar-radar bash

# GPU 모드 (NVIDIA)
docker compose --profile gpu up -d
docker exec -it airo-rebar-radar-gpu bash

# 컨테이너 내부에서
uv sync
```

---

## 실행

### uv 사용 시

```bash
uv run python -m src.main
```

### Conda / venv 사용 시

```bash
python -m src.main
```

---

## 시스템 의존성 (GUI 시각화용)

Linux에서 matplotlib GUI를 사용하려면:

```bash
# Debian/Ubuntu
apt-get install -y libgl1 libgtk2.0-dev tk

# X11 포워딩 설정 (원격 접속 시)
export DISPLAY=:0
```
