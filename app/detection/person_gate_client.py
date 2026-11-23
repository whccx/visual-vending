"""常驻人物扫描子进程的客户端；调用方负责持有 API 锁。"""

import json
import logging
import os
import select
import subprocess

from app import config

LOG = logging.getLogger("sku." + __name__)


def scan_person_gate(state, video):
    """通过应用状态中的客户端扫描视频，不可用时返回 None。

    参数：
        state: 包含可选 person_gate 客户端的应用状态。
        video: 视频文件路径。
    返回：
        每帧是否有人物的布尔列表；没有客户端或扫描失败时返回 None。
    """
    gate = getattr(state, "person_gate", None)
    return gate.scan(video) if gate is not None else None


class PersonGateClient:
    """管理人物扫描子进程并收发扫描请求。"""

    def __init__(self):
        """启动扫描子进程并等待就绪消息；失败时关闭进程并抛出异常。"""
        self.disabled = False
        env = os.environ.copy()
        env["YOLO_CONFIG_DIR"] = "/tmp/sku-person-yolo"
        env["YOLO_AUTOINSTALL"] = "false"
        self.process = subprocess.Popen(
            [str(config.PERSON_PYTHON), str(config.PERSON_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            reply = self._read(90)
            if reply.get("ready"):
                return
        except Exception:
            self.close()
            raise
        self.close()
        raise RuntimeError("人体检测进程未能启动")

    def _read(self, timeout):
        """在指定秒数内读取并解析一行 JSON；超时或进程退出时抛出异常。"""
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise RuntimeError("人体检测进程超时")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("人体检测进程意外退出")
        return json.loads(line)

    def scan(self, video):
        """请求子进程扫描视频并验证逐帧布尔标记。

        参数：
            video: 待扫描的视频路径。
        返回：
            按帧排列的布尔列表；客户端停用或扫描失败时返回 None。
        """
        if self.disabled:
            return None
        try:
            self.process.stdin.write(json.dumps({"video": str(video)}) + "\n")
            self.process.stdin.flush()
            reply = self._read(900)
            if "error" in reply:
                raise RuntimeError(reply["error"])
            flags = reply["flags"]
            if not flags or not all(isinstance(value, bool) for value in flags):
                raise RuntimeError("人体检测返回的标记无效")
            LOG.info(
                "人物扫描：总帧数=%s 检出人物帧数=%s 耗时=%.3f 秒",
                len(flags),
                sum(flags),
                reply.get("seconds", -1),
            )
            return flags
        except Exception:
            LOG.exception("人物检测不可用，回退为逐帧处理")
            # 超时进程可能稍后返回旧视频的结果，因此停用并关闭管道。
            self.disabled = True
            self.close()
            return None

    def close(self):
        """关闭输入管道并等待子进程退出；必要时终止或强制结束。"""
        proc = self.process
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
