# Visual Vending

面向开门柜的旷视 OpenAPI 兼容服务。项目包含独立 FastAPI、SQLite 商品与任务库、GPU worker，以及从 SKU Vision Pipeline 复制的 DINOv3、HOI-DETR、TensorRT/BG0 和商品匹配代码。商品参考图在本项目内由 YOLOE 和 OpenCV 审核。默认部署所需的模型、运行库和持久数据均位于本目录。

- [旷视 OpenAPI 接口文档](<旷视对接文档 - 超级收银员-OpenAPI接口文档-2026.09.22.html>)
- [现有 visual-platform 上游切换分析](旷视上游兼容切换分析.md)

## 接口

以下路径省略 `/openapi/sc/v1`。请求使用 HTTPS、`application/json`、`Referer`、毫秒 `X-Request-Date` 和 `Authorization: HMAC-SHA1 <AK>:<signature>`。`X-Request-Id` 可由调用方传入，缺省时服务生成。签名覆盖实际路径、规范化查询串、Host、请求头及原始请求体。

| 方法与路径 | 用途 |
|---|---|
| `POST /sku/` 或 `POST /sku` | 提交审核，返回 `data.ks_sku_id` |
| `GET /sku/{ks_sku_id}` | 查询待审核、审核中、通过或拒绝 |
| `PATCH /sku/{ks_sku_id}` | 提交复审，返回创建时的 `data.sku_id` |
| `DELETE /sku/{ks_sku_id}` | 删除私有 SKU，返回 `data.sku_id` |
| `GET /sku/all_sku/{barcode}` | 精确查询标准库与当前账户的私有库 |
| `GET /sku/skus/{ks_sku_id}` | 查询 SKU 详情 |
| `GET /sku/sku_list` | 按名称或条码搜索、分页，账户需要启用 `sku_search` |
| `POST /recognition` | 提交异步识别，返回 `data.ks_recog_id` |
| `GET /recognition/{ks_recog_id}` | 查询任务和识别结果 |

两种回调各自向提交时的 `notify_url` 发送签名 POST。确认采用 HTTP 200 和文档的 JSON 确认体，失败时依次延迟 10、30、60、120、180 秒重试。结果和待发回调同事务保存，识别任务绑定接单时的有效商品版本。worker 重启会接续处理中任务和未确认回调。接收方应按业务 ID 处理重复回调。

视频识别默认最多同时处理 5 个视频，额外任务排队等待，完成一个视频后自动接续下一个。同一订单的主辅视频共用视频并发名额，全部处理完成并核对一致性后返回订单结果。帧级处理使用 3 个柜区检测线程、2 个 HOI 检测实例和 2 个 BG0 实例。手部和运动联合筛帧保留触发帧前后各 10 帧。

文档的集合 SKU 提交尚未开放。运营方可以导入标准库和集合 SKU 的查询数据。

## 本机启动

需要 Docker Compose、NVIDIA Container Toolkit 和与本机 GPU 匹配的 TensorRT 引擎。Compose 默认使用本机镜像 `visual-vending-runtime:local` 与 `visual-vending-tensorrt:local`。API、GPU worker 和 TensorRT 检测器由同一个 Compose 项目启动，容器名均以 `visual-` 开头。权重与运行库从本项目内只读挂载，检测容器只共享临时视频目录。迁移到另一台机器时需要同时转移本目录与以上 Docker 镜像，目标 GPU 需要兼容对应 TensorRT 引擎。

先准备挂载目录、仅服务进程可读的账户配置和 TLS 文件。

```bash
mkdir -p data/spool
```

将 `data/accounts.json` 配置为下列结构。示例值仅作占位。Compose 服务以 UID 1000 读取运行目录。通过环境变量 `VENDING_ASSET_HOSTS` 限定允许访问的媒体主机，通过 `VENDING_CALLBACK_HOSTS` 限定回调接收端；多个值用逗号分隔，非标准端口要包含端口。

```json
{
  "your-access-key": {"secret": "your-secret-key", "sku_search": true}
}
```

将信任的服务证书和私钥分别放到 `data/tls.crt`、`data/tls.key`。配置完成后启动：

```bash
docker compose up -d
docker compose ps
curl --cacert data/tls.crt https://localhost:18768/health
```

交互文档默认在 `https://localhost:18768/docs`。同机 Java 使用 `vision.host=localhost:18768`，本地自签名证书需要加入其联调信任库。项目配置的本地信任库为 `data/visual-local-trust.p12`，密码为 `changeit`。Java 回调或媒体服务在宿主机运行时，worker 可用 `host.docker.internal` 访问宿主机绑定在可达接口的端口。端口只绑定回环地址，外部对接使用另行配置的受信任 HTTPS 入口。代理应保留验签所需的原始路径和 Host；如代理改写了 Host，通过 `VENDING_SIGNING_HOST` 指定客户端实际用于验签的值。Java 回调入口、商户端地址和源平台请求契约仍由现有 `visual-platform` 管理。

| 环境变量 | 作用 |
|---|---|
| `VENDING_MODEL_DIR`、`VENDING_RUNTIME_DIR` | 指向只读模型与运行库。默认读取本项目的 `models`、`runtime`。 |
| `VENDING_ASSET_HOSTS`、`VENDING_CALLBACK_HOSTS` | 允许的媒体和回调主机。 |
| `VENDING_CALLBACK_DELAY` | 终态首次投递前的等待时间，默认 3 秒。实际切换要按 Java 事务落库窗口核实。 |
| `VENDING_VIDEO_CONCURRENCY` | 同时处理的视频数，默认 5，可设置 1 至 5。修改后需重新创建 worker 容器。 |
| `VENDING_MATCHING_MODEL` | `dinov3_vit7b_arcface` 使用保守判定，读取 `models/held_product/dinov3_vit7b_arcface` 的 BF16 权重和同型号商品向量。切换前需完成向量重建。 |
| `VENDING_PUBLIC_URL` | 用于生成可访问的 300×300 商品封面地址。未配置时使用第一张参考图地址。 |
| `VENDING_CHECK_IMAGE_SIZE` | 默认检查参考图尺寸及比例。设为 0 可用非标准参考图做本机验收。 |
| `VENDING_REVIEW_DETECTOR` | 默认 `onnx`，使用本容器的 YOLO-World。可显式选择 `yoloe`，或用 `auto` 优先选择可用的 YOLO-World。 |
| `VENDING_PORT`、`VENDING_BIND` | 监听端口及绑定地址；默认只绑回环地址，设 `VENDING_BIND=0.0.0.0` 可放通局域网（配合证书 SAN）。 |
| `VENDING_GPU_IMAGE`、`VENDING_TENSORRT_IMAGE` | 运行镜像。 |

`resource_type=1` 接收 URL 列表或 `resource_infos` 中的 URL。 `resource_type=2` 在 `data/resources` 中查找与 `resource_id` 同名的预登记文件。API 内可保留货架层和相机 ID 元数据。库表、测试数据和回调队列均在本项目 `data` 目录。

## 商品图片审核

新建 SKU 或提交允许更换的参考图时，GPU worker 下载图片后使用容器内的 YOLO-World ONNX 检测商品主体，再用 OpenCV 检查居中、面积、清晰度和多主体歧义。透明图片先合成白底再检测。审核使用本项目的 `models/review` 模型和镜像内的 ONNX Runtime。

| 图片结果 | SKU 处理 |
|---|---|
| 全部通过 | 继续提取向量，建模成功后发审核通过回调。 |
| 有图片明确拒绝 | 终止建模，在审核结果与回调中附上第几张图及拒绝原因。 |
| 有图片需人工，或本地检测异常 | SKU 进入人工复核队列，查询返回 `0 / 待审核`，等待人工决定或重试。 |

本地复核用下列命令。人工通过只适用于复核时的同一批图片内容，通过后仍需正常建模。人工拒绝按原协议发送审核不通过回调。

```bash
docker compose exec -T worker /opt/conda/bin/python -m vending.review_sku list
docker compose exec -T worker /opt/conda/bin/python -m vending.review_sku approve <ks_sku_id>
docker compose exec -T worker /opt/conda/bin/python -m vending.review_sku reject <ks_sku_id> --reason "参考图不合格"
docker compose exec -T worker /opt/conda/bin/python -m vending.review_sku retry <ks_sku_id>
```

## 导入旧 SKU

用已核对的清单明确指定每个旧 ID、上游原始 `sku_id` 与该商品的特征来源。私有 SKU 的 `owner` 为新上游验签使用的 AK。旧 ID 必须按 JSON 字符串传入，以保留前导零和完整精度。异名与条码本身只能用于人工核对。

```json
[
  {
    "owner": "your-access-key",
    "sku_id": "visual-platform-sku-id",
    "ks_sku_id": "001234567890123456789",
    "sku_name": "待核对的商品",
    "legacy_cabinet": "cabinet-a",
    "legacy_product": 20
  }
]
```

把清单放在 `data/catalog-import.json`。若已验证本地旧库中的向量对应同一 SKU，把该库的有效快照复制到新项目的 `data`，随后导入。导入时旧库以只读方式打开。

```bash
docker compose exec -T worker /opt/conda/bin/python -m vending.import_catalog \
  /workspace/data/catalog-import.json --legacy-db /workspace/data/catalog-snapshot.sqlite3
```

也可以在清单中用 `image_files` 或允许访问的 `img_urls` 重新提取向量，或用 `aliases` 配置经过核对、唯一的历史别名。 `standard: true` 导入标准库， `sub_sku` 保存已存在的集合 SKU 元数据。导入脚本按条目提交，同一记录可重试，内容冲突时明确失败。识别匹配时只使用本单候选，回调中的 SKU 使用本次请求提交的 ID。

## 验收边界

当前 GPU worker 按确认离架的商品轨迹做向量和颜色匹配。同一项 SKU 的独立轨迹会聚合数量；多路视频以主视角为基准，只有视角结果一致才沿用自动结论。对缺证据或视角矛盾的任务返回无法识别，以便既有 Java 组合审核流程接手。

旷视的训练数据、标准库快照和告警模型需要独立来源。本项目提供标准库导入和查询。复杂拿放、同款多件、跨相机轨迹融合、持物回摆和放回时序、全部告警分类及二次推理尚未达到旷视的业务等价。图片审核的主体检测和画质检查均在本项目内执行。多张参考图是否为同一商品、拍摄角度是否齐全，仍需独立判断。真实旷视和现有生产 Java 服务都无需参与本机启动。

协议和数据库回归可用宿主机执行，不需要 GPU。

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q
```
