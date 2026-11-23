"""视频逐帧人物搜索与目标跟踪的保守门控策略。"""


def buffered_flags(flags, radius=2):
    """扩展人物出现帧的前后缓冲范围；未提供扫描结果时返回 None。"""
    if flags is None:
        return None
    return [
        any(flags[max(0, i - radius) : min(len(flags), i + radius + 1)]) for i in range(len(flags))
    ]


class PersonGatePolicy:
    """依据人物扫描结果控制逐帧商品检测。"""

    def __init__(self, flags, miss_limit=5):
        """初始化门控状态、人物扫描标记和连续未检测次数上限。"""
        self.flags = buffered_flags(flags)
        self.miss_limit = miss_limit
        self.tracking = False
        self.misses = 0
        self.skipped = 0
        self.detected = 0
        self.returns_to_search = 0

    def allow(self, index):
        """判断零起始帧索引是否需要送入检测器，并更新统计。

        人物扫描结果缺失或不覆盖该帧时，保守地允许检测。
        """
        if self.flags is None or index >= len(self.flags):
            self.detected += 1
            return True
        if not self.tracking:
            if not self.flags[index]:
                self.skipped += 1
                return False
            self.tracking = True
            self.misses = 0
        self.detected += 1
        return True

    def observe(self, detected):
        """根据本帧检测结果更新跟踪状态；连续漏检达到上限后恢复搜索。"""
        if self.flags is None:
            return
        if detected:
            self.tracking = True
            self.misses = 0
        elif self.tracking:
            self.misses += 1
            if self.misses >= self.miss_limit:
                self.tracking = False
                self.misses = 0
                self.returns_to_search += 1

    def summary(self):
        """返回输入帧、送检帧、跳过帧和重新搜索次数的统计字典。"""
        return {
            "input_frames": self.detected + self.skipped,
            "sent_to_detector": self.detected,
            "skipped_no_person": self.skipped,
            "returns_to_search": self.returns_to_search,
        }
