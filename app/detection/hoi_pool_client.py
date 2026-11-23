"""管理两个常驻、预热的 HOI 模型进程。"""

import json
import logging
import os
import select
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config
from app.detection.person_gate_policy import PersonGatePolicy

LOG = logging.getLogger("sku." + __name__)


def split_frame_ranges(count):
    """将帧数分为最多两个无重叠、无遗漏的半开区间。"""
    if count <= 0:
        raise ValueError("视频没有帧")
    middle = (count + 1) // 2
    return [(0, middle)] + ([(middle, count)] if middle < count else [])


def apply_person_gate(frames, hands, flags):
    """按完整视频原始顺序应用状态性人物门控。"""
    if len(frames) != len(hands) or (flags is not None and len(flags) != len(frames)):
        raise ValueError("检测帧数与人物扫描不一致")
    gate = PersonGatePolicy(flags)
    gated_frames, gated_hands = [], []
    # 保持旧版门控的批内决策顺序；旧版一次决定一个检测批的全部帧。
    gate_batch_size = config.hoi_detr_batch_size()
    if gate_batch_size < 1:
        raise ValueError("HOI 人物门控批大小无效")
    for start in range(0, len(frames), gate_batch_size):
        end = min(start + gate_batch_size, len(frames))
        decisions = [gate.allow(index) for index in range(start, end)]
        for index, allowed in zip(range(start, end), decisions):
            objects, frame_hands = frames[index], hands[index]
            if allowed:
                gated_frames.append(objects)
                gated_hands.append(frame_hands)
                gate.observe(any(obj.get("hand_indices") for obj in objects))
            else:
                gated_frames.append([])
                gated_hands.append([])
    return {"frames": gated_frames, "hands": gated_hands, "person_gate": gate.summary()}


class _Worker:
    """独占管道的单个 HOI 隔离进程。"""

    def __init__(self, index):
        self.index = index
        self.lock = threading.Lock()
        self.process = None
        self.log_file = None

    def start(self):
        """启动进程并等待实际预热成功。"""
        self.close()
        config.JOBS.mkdir(parents=True, exist_ok=True)
        self.log_file = (config.JOBS / ("hoi-resident-%s.log" % self.index)).open("a")
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
            env.pop(key, None)
        env["OMP_NUM_THREADS"] = "4"
        self.process = subprocess.Popen(
            [str(config.HOI_PYTHON), "-m", "app.workers.sku_hoi_detr_worker", "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_file,
            text=True, bufsize=1, cwd=str(config.ROOT), env=env,
        )
        try:
            reply = self._read(config.HOI_POOL_START_TIMEOUT_SECONDS)
            if not reply.get("ready"):
                raise RuntimeError("HOI 进程预热未成功: %s" % reply)
        except BaseException:
            self.close()
            raise
        LOG.info("HOI 常驻进程预热完成：编号=%s PID=%s", self.index, self.process.pid)

    def _read(self, timeout):
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise RuntimeError("HOI 进程响应超时")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("HOI 进程已退出")
        return json.loads(line)

    def call(self, request):
        """串行使用当前进程的管道发送检测请求。"""
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("HOI 常驻进程尚未就绪")
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                reply = self._read(config.HOI_POOL_REQUEST_TIMEOUT_SECONDS)
                if not reply.get("ok"):
                    raise RuntimeError("HOI 检测失败: %s" % reply.get("error"))
                return reply
            except Exception:
                self.close()
                raise

    def close(self):
        proc, self.process = self.process, None
        if proc is not None:
            if proc.poll() is None:
                if proc.stdin:
                    proc.stdin.close()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            if proc.stdout:
                proc.stdout.close()
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None


class HOIPoolClient:
    """拥有两个常驻进程，在空闲时自动恢复意外退出的进程。"""

    def __init__(self):
        self.workers = [_Worker(i) for i in range(config.HOI_POOL_PROCESS_COUNT)]
        self.stopping = threading.Event()
        self.monitor = None
        try:
            for worker in self.workers:
                worker.start()
        except BaseException:
            self.close()
            raise
        self.monitor = threading.Thread(target=self._monitor, name="hoi-pool-monitor", daemon=True)
        self.monitor.start()

    def _monitor(self):
        while not self.stopping.wait(2):
            for worker in self.workers:
                if self.stopping.is_set():
                    return
                if not worker.lock.acquire(blocking=False):
                    continue
                try:
                    if worker.process is None or worker.process.poll() is not None:
                        LOG.warning("HOI 常驻进程退出，正在重启：编号=%s", worker.index)
                        try:
                            worker.start()
                        except Exception:
                            LOG.exception("HOI 常驻进程重启失败：编号=%s", worker.index)
                finally:
                    worker.lock.release()

    def detect(self, video, job, frame_count, flags):
        """并行检测两个帧区间，顺序合并并重放人物门控。"""
        ranges = split_frame_ranges(frame_count)
        if flags is not None and len(flags) == frame_count and not any(flags):
            return apply_person_gate([[] for _ in flags], [[] for _ in flags], flags)
        if flags is not None and len(flags) != frame_count:
            raise ValueError("人物扫描帧数与视频不一致")
        requests = [
            {"video": str(video), "output": str(Path(job) / ("hoi-part-%s.json" % i)),
             "start": start, "end": end}
            for i, (start, end) in enumerate(ranges)
        ]
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = [executor.submit(self.workers[i].call, request)
                       for i, request in enumerate(requests)]
            for future in futures:
                future.result()
        frames, hands = [], []
        for request in requests:
            data = json.loads(Path(request["output"]).read_text(encoding="utf-8"))
            expected = request["end"] - request["start"]
            if len(data["frames"]) != expected or len(data["hands"]) != expected:
                raise ValueError("HOI 检测返回帧数不匹配")
            frames.extend(data["frames"])
            hands.extend(data["hands"])
        return apply_person_gate(frames, hands, flags)

    def close(self):
        self.stopping.set()
        if self.monitor is not None:
            self.monitor.join(timeout=config.HOI_POOL_START_TIMEOUT_SECONDS + 5)
        for worker in self.workers:
            with worker.lock:
                worker.close()
