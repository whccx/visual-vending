# visual-vending 项目指引

## 项目与服务

visual-vending 是开门柜的旷视 OpenAPI 兼容服务，负责商品参考图审核、SKU 建模、视频识别和结果回调。服务使用 FastAPI、SQLite、GPU worker 和 TensorRT 检测服务，项目内的模型、运行库和商品数据共同构成独立部署。Java visual-platform 负责商户业务入口、平台记录和回调转发。

| Docker Compose 服务 | 容器 | 用途 |
| --- | --- | --- |
| `api` | `visual-vending-api` | HTTPS OpenAPI，默认宿主端口 `18768` 映射到容器端口 `8768` |
| `worker` | `visual-vending-worker` | 商品审核、建模、异步识别和回调 |
| `detector` | `visual-vending-detector` | TensorRT 检测，Compose 网络内端口 `8770` |

端口和地址以当前 Compose 配置与运行状态为准。本机 Java 联调服务位于 `10070`，管理前端位于 `18070`。它们由 visual-platform 仓库管理。

## 维护约定

- 使用中文沟通和编写项目文档，Git 提交说明使用中文。推送前核对当前仓库、分支和 Git 远程。
- 在当前仓库内维护代码、模型、运行库和数据。保留用户已有改动。接入推理能力时保持项目可独立运行，新增本项目容器使用 `visual-` 前缀。
- 使用 `docker compose`。给可能阻塞的命令设置超时。启停或重启前核对项目与目标服务，代理或相邻项目配置按用户授权范围操作。
- OpenAPI 服务配置集中在 `vending/config.py`，底层识别配置集中在 `app/config.py`。识别模块日志通过 `app/logging.py` 管理，项目日志用中文描述事件。
- `auth.md` 是本地凭据清单。提交和交付前核对待提交内容，排除凭据、私钥、运行数据库和未经指定的测试媒体。
- 实验或回归时分别记录服务健康、任务状态、业务识别结果和回调。识别验收用指定柜的真实视频和标注，核对 SKU、数量及回调。只有获授权时才提交会写入平台或服务的数据。

## 代码与文档入口

| 位置 | 内容 |
| --- | --- |
| `vending/api.py`、`vending/schemas.py`、`vending/protocol.py` | OpenAPI 路由、字段约束和签名校验 |
| `vending/worker.py`、`vending/store.py`、`vending/engine.py` | 异步任务、SQLite 持久化和识别 |
| `app/`、`services/tensorrt_detector/` | 底层识别、检测和 TensorRT 服务 |
| `models/`、`runtime/`、`data/` | 模型、运行库和本地数据 |
| `tests/` | 协议、商品图片审核等自动化测试 |

- [README.md](README.md) 记录部署、接口和商品库导入。
- [test.md](test.md) 记录 SKU 与识别的 Java 业务提交和直连测试。
- 本机的 `auth.md` 记录登录和 API 凭据，由维护者在各自环境中保管。
- [旷视 OpenAPI 接口文档](<旷视对接文档 - 超级收银员-OpenAPI接口文档-2026.09.22.html>) 提供对接协议。

## 检查与联调

```bash
timeout 15s docker compose config --services
timeout 15s docker compose ps
curl --max-time 5 --cacert data/tls.crt https://localhost:18768/health
timeout 120s python3 -m pytest -q tests
```

平台商品页和识别记录页读取 Java 自己的业务数据。需要平台落库时，经 `10070` 提交。直接测试识别服务时，经 `18768` 的 `/openapi/sc/v1` 签名提交和查询。请求字段、认证和回调见 [联调文档](test.md)。
