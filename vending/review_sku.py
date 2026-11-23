"""从本地管理视觉 SKU 的待人工图片审核。"""

import argparse
import json

from vending.config import Settings
from vending.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("list", help="显示待复核 SKU、图片与原因")
    for command, help_text in (
        ("approve", "同意当前图片，继续建模"),
        ("reject", "拒绝当前图片"),
        ("retry", "重新执行自动图片质检"),
    ):
        action = commands.add_parser(command, help=help_text)
        action.add_argument("ks_sku_id")
        if command == "reject":
            action.add_argument("--reason", required=True)
    args = parser.parse_args()
    store = Store(Settings.from_env().data_dir)
    store.initialize()
    if args.action == "list":
        print(json.dumps(store.pending_image_reviews(), ensure_ascii=False, indent=2))
        return
    reason = getattr(args, "reason", "").strip()
    if args.action == "reject" and not 0 < len(reason) <= 256:
        parser.error("拒绝原因须为 1 至 256 个字符")
    try:
        store.decide_image_review(args.ks_sku_id, args.action, reason)
    except ValueError as error:
        parser.error(str(error))
    print(f"{args.ks_sku_id} 已提交，结果由审核任务更新")


if __name__ == "__main__":
    main()
