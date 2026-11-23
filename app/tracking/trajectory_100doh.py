"""独立的 100DOH 运动轨迹关联，使用与现有流水线一致的阈值。"""

import math


def select_product_track(tracks, min_points=3, min_displacement=5.0):
    """优先选择持续运动的轨迹，不可靠的运动点不参与选择。"""
    candidates = []
    for track in tracks:
        points = [p for p in track["points"] if p.get("motion_reliable", True)]
        if len(points) < min_points:
            continue
        displacement = math.dist(points[0]["point"], points[-1]["point"])
        if displacement < min_displacement:
            continue
        candidates.append((len(points) * displacement, len(points), -track["track_id"], track))
    return max(candidates, key=lambda item: item[:3])[3] if candidates else None


class ProductTracker:
    """根据检测框中心的运动关联逐帧商品轨迹。"""

    def __init__(self, max_gap=8):
        """初始化轨迹列表和允许的最大帧间隔。"""
        self.max_gap = max_gap
        self.tracks = []
        self.next_id = 1
        self.last_frame = None

    def _compatible(self, t, frame, point, box, ignore_last=False):
        """根据距离、运动方向和速度判断检测框能否接续轨迹。"""
        points = [p for p in t["points"] if p.get("motion_reliable", True)]
        points = points[:-1] if ignore_last else points
        if not points:
            return False, "no_motion_history"
        last = points[-1]
        gap = frame - last["frame"]
        if gap <= 0 or gap > self.max_gap:
            return False, "frame_gap"
        dx = point[0] - last["point"][0]
        dy = point[1] - last["point"][1]
        distance = math.hypot(dx, dy)
        old_box = last["box"]
        diag = min(
            math.hypot(box[2] - box[0], box[3] - box[1]),
            math.hypot(old_box[2] - old_box[0], old_box[3] - old_box[1]),
        )
        speeds = []
        for a, b in zip(points[-3:-1], points[-2:]):
            dt = b["frame"] - a["frame"]
            if dt > 0:
                speeds.append(math.dist(a["point"], b["point"]) / dt)
        prior_speed = max(speeds, default=0)
        limit = max(80.0, 0.75 * diag, 3.5 * prior_speed) * gap
        if distance > limit:
            return False, "distance"
        if len(points) >= 3:
            first = points[-3]
            dt = last["frame"] - first["frame"]
            vx = (last["point"][0] - first["point"][0]) / dt
            vy = (last["point"][1] - first["point"][1]) / dt
            speed = math.hypot(vx, vy)
            noise = max(8.0, 0.08 * diag)
            if speed > noise and distance > noise:
                cosine = (dx * vx + dy * vy) / (distance * speed)
                if cosine < -0.5:
                    return False, "direction"
                error = math.hypot(dx - vx * gap, dy - vy * gap)
                strict = max(45.0, 0.6 * diag, 2.5 * speed) * gap
                relaxed = max(45.0, 0.6 * diag, 4.0 * speed) * gap
                overlap = max(0, min(box[2], old_box[2]) - max(box[0], old_box[0])) * max(
                    0, min(box[3], old_box[3]) - max(box[1], old_box[1])
                )
                if error > relaxed or (error > strict and overlap <= 0):
                    return False, "velocity_change"
        return True, "accepted"

    def step(self, frame, box, ranking):
        """处理一帧检测结果，返回关联结果和历史路径。"""
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("Frame numbers must be strictly increasing")
        self.last_frame = frame
        if box is None:
            return None
        point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        candidates = []
        rejected = []
        for t in self.tracks:
            if frame - t["points"][-1]["frame"] > self.max_gap:
                continue
            ok, reason = self._compatible(t, frame, point, box)
            if ok:
                reliable = [p for p in t["points"] if p.get("motion_reliable", True)]
                candidates.append((math.dist(point, reliable[-1]["point"]), t, False))
            else:
                rejected.append({"track_id": t["id"], "reason": reason})
                reliable = [p for p in t["points"] if p.get("motion_reliable", True)]
                if (
                    len(reliable) >= 4
                    and reliable[-1] is t["points"][-1]
                    and frame - reliable[-1]["frame"] == 1
                ):
                    rescued, _ = self._compatible(t, frame, point, box, ignore_last=True)
                    if rescued:
                        candidates.append((math.dist(point, reliable[-2]["point"]), t, True))
        candidates.sort(key=lambda x: x[0])
        ambiguous = len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 50
        pending = False
        if not candidates and not ambiguous:
            provisional = []
            for old in self.tracks:
                last = old["points"][-1]
                gap = frame - last["frame"]
                if (
                    gap != 1
                    or not last.get("motion_reliable", True)
                    or len([p for p in old["points"] if p.get("motion_reliable", True)]) < 3
                ):
                    continue
                diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
                distance = math.dist(point, last["point"])
                if distance <= max(120.0, diagonal):
                    provisional.append((distance, old))
            provisional.sort(key=lambda x: x[0])
            if provisional and (
                len(provisional) == 1 or provisional[1][0] - provisional[0][0] >= 50
            ):
                t = provisional[0][1]
                pending = True
                association = "pending_jump"
            else:
                t = {"id": self.next_id, "points": []}
                self.next_id += 1
                self.tracks.append(t)
                association = "new"
        elif ambiguous:
            t = {"id": self.next_id, "points": []}
            self.next_id += 1
            self.tracks.append(t)
            association = "new_ambiguous"
        else:
            _, t, skip_last = candidates[0]
            prior_pending = not t["points"][-1].get("motion_reliable", True)
            if skip_last:
                t["points"][-1]["motion_reliable"] = False
            association = "continued_after_jump" if skip_last or prior_pending else "continued"
        t["points"].append(
            {"frame": frame, "point": point, "box": box, "motion_reliable": not pending}
        )
        return {
            "track_id": t["id"],
            "point": point,
            "association": association,
            "rejected_matches": rejected,
            "path": list(t["points"]),
        }

    def summary(self):
        """返回所有轨迹及其观测点。"""
        return [{"track_id": t["id"], "points": t["points"]} for t in self.tracks]
