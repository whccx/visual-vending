"""跨进程共享的按日轮转日志和安全的过期清理。"""

import atexit
import copy
import fcntl
import logging
import logging.handlers
import os
import re
import stat
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import (
    LOG_CLEANUP_INTERVAL_SECONDS,
    LOG_FILENAME_UTC_OFFSET_HOURS,
    LOG_LOCK_WAIT_SECONDS,
    LOG_WARNING_INTERVAL_SECONDS,
    log_options,
)

_GROUP = re.compile(
    r"^((?:sku-[a-z0-9_-]+-\d+-[0-9a-f]{32}|"
    r"\d{4}-\d{2}-\d{2}(?: \d{6}(?:_[1-9]\d*)?)?))\.(log(?:\.[1-9]\d*)?|lock)$"
)
_state_lock = threading.RLock()
_state = None
_last_warning = 0.0
# 按模块名匹配日志分类；未匹配时使用默认前缀。
LOG_PREFIX_RULES = (
    (('recognition', 'recognize_', 'hoi_detr'), '推理'),
    (('detection', 'person_gate'), '检测'),
)
LOG_DEFAULT_PREFIX = '服务'
_LEVEL_NAMES = {
    logging.DEBUG: '调试', logging.INFO: '信息', logging.WARNING: '警告',
    logging.ERROR: '错误', logging.CRITICAL: '严重错误',
}
_DETAIL_NAMES = {
    'input_frames': '输入帧数', 'sent_to_detector': '送检帧数',
    'skipped_no_person': '无人物跳过帧数', 'returns_to_search': '返回搜索次数',
    'applied': '已应用', 'frames': '总帧数', 'area_frames': '区域帧数',
    'hand_frames': '手部帧数', 'hand_tracks': '手部轨迹数',
    'near_hand_tracks': '靠近区域的手部轨迹数', 'candidate_count': '候选数',
    'seed_candidates': '初始候选数', 'confirmed_product_tracks': '确认商品轨迹数',
    'continuation_candidates': '延续候选数', 'eligible_candidates': '符合条件候选数',
    'near_ratio': '靠近比例', 'raw_candidates': '原始候选数',
    'moving_candidates': '运动候选数', 'stationary_candidates': '静止候选数',
    'unknown_candidates': '状态未知候选数', 'rankable_points': '可排序点数',
    'tracks': '轨迹数', 'selected_track_points': '选中轨迹点数',
    'confirmed_points': '确认点数', 'valid_tracks': '有效轨迹数',
    'displacement_rejected': '位移不足剔除数', 'relaxed_fallback': '启用宽松回退',
    'hand_size_rejected': '手部尺寸筛除数', 'hand_not_detected': '未检测到手部数',
}


def _translate_detail(value):
    if isinstance(value, dict):
        return {_DETAIL_NAMES.get(key, key): _translate_detail(item) for key, item in value.items()}
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, (tuple, list)):
        return type(value)(_translate_detail(item) for item in value)
    return value


class _ChineseFormatter(logging.Formatter):
    """翻译项目日志级别和统计字段，保留异常堆栈原文。"""

    def format(self, record):
        translated = copy.copy(record)
        translated.levelname = _LEVEL_NAMES.get(record.levelno, record.levelname)
        translated.category = next(
            (prefix for names, prefix in LOG_PREFIX_RULES if any(name in record.name for name in names)),
            LOG_DEFAULT_PREFIX,
        )
        if isinstance(record.args, dict):
            if isinstance(record.msg, str) and '%(' in record.msg:
                # 命名格式参数的键不能修改。
                translated.args = {key: _translate_detail(value) for key, value in record.args.items()}
            else:
                translated.args = _translate_detail(record.args)
        elif isinstance(record.args, tuple):
            translated.args = tuple(_translate_detail(value) for value in record.args)
        return super().format(translated)


def _warn(message):
    global _last_warning
    now = time.monotonic()
    if now - _last_warning >= LOG_WARNING_INTERVAL_SECONDS or not _last_warning:
        _last_warning = now
        try:
            print("SKU 文件日志不可用：" + str(message), file=sys.stderr, flush=True)
        except Exception:
            pass


def _lock_file(path, *, nonblocking=False):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("lock is not a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        return fd
    except BaseException:
        os.close(fd)
        raise


def _close_lock(fd):
    os.close(fd)


def _creation_lock(path):
    deadline = time.monotonic() + LOG_LOCK_WAIT_SECONDS
    while True:
        try:
            return _lock_file(path, nonblocking=True)
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _cleanup_locked(directory, retention_days, total_max_bytes):
    today = datetime.now(timezone(timedelta(hours=LOG_FILENAME_UTC_OFFSET_HOURS))).strftime('%Y-%m-%d')
    groups = {}
    for entry in directory.iterdir():
        match = _GROUP.fullmatch(entry.name)
        if not match or entry.is_symlink() or not entry.is_file():
            continue
        groups.setdefault(match.group(1), []).append(entry)
    records = []
    total = 0
    for group, entries in groups.items():
        # 通用日期文件名可能属于其他程序，仅清理由本程序锁文件标记的组。
        if not any(p.name == group + '.lock' for p in entries):
            continue
        logs = [p for p in entries if '.log' in p.name]
        try:
            size = sum(p.lstat().st_size for p in logs)
            modified = max((p.lstat().st_mtime for p in logs), default=0)
        except OSError as exc:
            _warn(exc)
            continue
        total += size
        records.append((modified, group, logs, size))
    cutoff = time.time() - retention_days * 86400
    for modified, group, logs, size in sorted(records):
        if group == today:
            continue
        if modified >= cutoff and total <= total_max_bytes:
            continue
        lock_path = directory / (group + '.lock')
        try:
            fd = _lock_file(lock_path, nonblocking=True)
        except (OSError, BlockingIOError):
            continue
        try:
            for path in logs:
                if not path.is_symlink() and path.is_file():
                    path.unlink()
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            total -= size
        except OSError as exc:
            _warn(exc)
        finally:
            _close_lock(fd)
    if total > total_max_bytes:
        _warn('活动日志或无法访问的日志超过总容量目标')


def cleanup_logs(directory, retention_days, total_max_bytes):
    """仅清理可独占锁定的本项目日志组。"""
    try:
        fd = _lock_file(directory / '.sku-cleanup.lock', nonblocking=True)
    except (OSError, BlockingIOError):
        return
    try:
        _cleanup_locked(directory, retention_days, total_max_bytes)
    except OSError as exc:
        _warn(exc)
    finally:
        _close_lock(fd)


class _FileHandler(logging.handlers.RotatingFileHandler):
    def __init__(self, directory, max_bytes, backup_count):
        self.directory = directory
        super().__init__(directory / 'pending.log', maxBytes=max_bytes,
                         backupCount=backup_count, encoding='utf-8', delay=True)

    def handleError(self, record):
        self.disabled_after_error = True
        _warn('写入或轮转失败')

    def shouldRollover(self, record):
        if self.stream is None:
            self.stream = self._open()
        self.stream.seek(0, os.SEEK_END)
        size = len((self.format(record) + '\n').encode('utf-8'))
        return self.stream.tell() > 0 and self.stream.tell() + size > self.maxBytes

    def emit(self, record):
        if getattr(self, 'disabled_after_error', False):
            return
        fd = None
        try:
            fd = _creation_lock(self.directory / '.sku-cleanup.lock')
            day = datetime.now(timezone(timedelta(hours=LOG_FILENAME_UTC_OFFSET_HOURS))).strftime('%Y-%m-%d')
            marker = _lock_file(self.directory / (day + '.lock'))
            _close_lock(marker)
            self.baseFilename = str(self.directory / (day + '.log'))
            super().emit(record)
        except (OSError, ValueError):
            self.handleError(record)
        finally:
            if self.stream:
                self.stream.close()
                self.stream = None
            if fd is not None:
                _close_lock(fd)


def _maintenance(stop, options):
    while not stop.wait(LOG_CLEANUP_INTERVAL_SECONDS):
        cleanup_logs(options['directory'], options['retention_days'], options['total_max_bytes'])


def _shutdown():
    global _state
    with _state_lock:
        if not _state or _state['pid'] != os.getpid():
            return
        _state['stop'].set()
        logger = logging.getLogger('sku')
        handler = _state['handler']
        if handler:
            logger.removeHandler(handler)
            handler.close()
        if _state['lock'] is not None:
            _close_lock(_state['lock'])
        _state = None


def setup_logging(role='app'):
    """初始化项目日志；文件日志错误不阻止业务启动。"""
    global _state
    with _state_lock:
        pid = os.getpid()
        if _state and _state['pid'] == pid:
            return
        logger = logging.getLogger('sku')
        # fork 后丢弃继承的 handler 和锁，由子进程创建自己的日志组。
        if _state:
            if _state['handler']:
                logger.removeHandler(_state['handler'])
                _state['handler'].close()
            if _state['lock'] is not None:
                _close_lock(_state['lock'])
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not any(getattr(h, '_sku_console', False) for h in logger.handlers):
            console = logging.StreamHandler(sys.stderr)
            console._sku_console = True
            console.setFormatter(_ChineseFormatter('[%(category)s] %(levelname)s %(name)s：%(message)s'))
            logger.addHandler(console)
        stop = threading.Event()
        _state = {'pid': pid, 'handler': None, 'lock': None, 'stop': stop}
        try:
            options = log_options()
            level = logging.getLevelName(options['level'])
            if (
                not isinstance(level, int)
                or options['max_bytes'] <= 0
                or options['backup_count'] < 1
                or options['retention_days'] < 0
                or options['total_max_bytes'] < 0
            ):
                raise ValueError('日志配置无效')
            directory = options['directory']
            directory.mkdir(parents=True, exist_ok=True)
            handler = _FileHandler(directory, options['max_bytes'], options['backup_count'])
            _state['handler'] = handler
            handler.setFormatter(_ChineseFormatter(
                '%(asctime)s [%(category)s] %(levelname)s 进程=%(process)d 模块=%(name)s：%(message)s'
            ))
            handler.setLevel(level)
            logger.setLevel(level)
            logger.addHandler(handler)
            cleanup_logs(directory, options['retention_days'], options['total_max_bytes'])
            threading.Thread(target=_maintenance, args=(stop, options), daemon=True, name='sku-log-cleanup').start()
        except (OSError, ValueError, TypeError) as exc:
            if _state['lock'] is not None:
                _close_lock(_state['lock'])
                _state['lock'] = None
            _warn(exc)


atexit.register(_shutdown)
