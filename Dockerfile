# 固定 Debian 发行版，避免基础镜像漂移导致构建行为变化。
FROM python:3.11-slim-bookworm

# 运行时系统依赖：
#   libgl1 —— RapidOCR 依赖的 opencv(cv2) 在 import 时需要 libGL.so.1，
#             slim 镜像默认没有，缺了会直接 ImportError，RapidOCR 整个不可用。
#   libglib2.0-0 —— cv2 的间接依赖。这里用容错写法，因为该包在 Debian 13 已更名为
#             libglib2.0-0t64；两种名字都试一遍，避免基础镜像升级后构建失败。
#   tesseract-ocr + tesseract-ocr-chi-sim —— 后备 OCR 引擎（RapidOCR 优先）。
# 注意：不要装 libtesseract-dev / libgl1-mesa-dev 这类 -dev 包，那是编译期头文件，运行期用不到。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
    && (apt-get install -y --no-install-recommends libglib2.0-0 \
        || apt-get install -y --no-install-recommends libglib2.0-0t64) \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# workers 固定为 1：数据层是同目录 JSON 文件，进程内的写锁无法跨进程生效，
# 多 worker 并发写会互相覆盖。要扩容需先把存储换成数据库。
#
# timeout 放到 300 秒：OCR 在免费实例的受限 CPU 上比本地慢一个数量级
# （本地约 3 秒，线上首次上传实测 83 秒），必须留足余量，否则会被 worker 超时掐断（502）。
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --graceful-timeout 30
