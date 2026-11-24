# SKU 与视频识别联调

| 联调目标 | 提交服务 | 结果查看位置 |
| --- | --- | --- |
| 验证平台 SKU 新增、修改和审核 | Java `visual-platform`，端口 `10070` | [商品 SKU 页面](http://localhost:18070/sku) |
| 验证平台视频识别及结果回写 | Java `visual-platform`，端口 `10070` | [识别记录页面](http://localhost:18070/records) |
| 单独测试 SKU 审核或视频识别 | `visual-vending`，端口 `18768` | [Visual Vending OpenAPI](https://127.0.0.1:18768/docs) 的查询接口或测试回调 |

Java 业务接口使用商户 `APP_ID` 和 `APP_SECRET` 做 HTTP Basic 认证。直调使用 `visual-vending` 账户的 AK 和 SK 做 HMAC-SHA1 签名。下面请求体里的 ID、图片、视频及回调地址均为示例。执行时使用测试账户、唯一的业务 ID，以及对应服务可访问的媒体和回调地址。

## SKU 业务提交

1. 调用 `POST http://localhost:10070/openInterface/open/sku/create`。请求体示例如下。

   ```json
   {
     "gpSkuId": "test-platform-sku-001",
     "skuName": "联调商品",
     "imageUrls": ["http://host.docker.internal:8089/media/sku-001.jpg"],
     "callbackUrl": "http://test-callback.example/sku"
   }
   ```

2. 保存响应中的 `data.rySkuId`。Java 调用识别服务的 `POST /openapi/sc/v1/sku` 后，将平台商品记录和识别 SKU ID 写入 MySQL 的 `open_sku`。
3. 在 [商品 SKU 页面](http://localhost:18070/sku) 按商品名称或 ID 查找记录。页面通过 Java 的 `GET /web/open/sku/page` 和 `GET /web/open/sku/detail/{id}` 读取平台库。审核完成后，识别服务回调 Java 的 `POST /openInterface/open/sku/notifySku`，由 Java 更新页面所显示的审核结果。

平台修改入口为 `PUT http://localhost:10070/openInterface/open/sku/edit/{skuId}`，由 Java 调用识别服务的 SKU `PATCH` 接口提交修改。

## SKU 直接测试

以下路径以 `https://127.0.0.1:18768` 为基址。接口见 [Swagger 文档](https://127.0.0.1:18768/docs)。

| 接口 | 用途 |
| --- | --- |
| `POST /openapi/sc/v1/sku` | 提交图片审核，返回 `data.ks_sku_id` |
| `GET /openapi/sc/v1/sku/{ks_sku_id}` | 查询审核状态，`data.result.code` 为 `2` 表示通过，为 `3` 表示拒绝 |
| `GET /openapi/sc/v1/sku/skus/{ks_sku_id}` | 查询商品详情 |
| `GET /openapi/sc/v1/sku/sku_list` | 按名称或条码搜索并分页，需要账户开启 `sku_search` |
| `PATCH /openapi/sc/v1/sku/{ks_sku_id}` | 修改商品，符合复审条件的修改会重新提交审核 |
| `DELETE /openapi/sc/v1/sku/{ks_sku_id}` | 测试结束后清理本次创建的私有 SKU |

新增请求体示例如下。需要验证审核回调时，加入可达的 `notify_url`。

```json
{
  "sku_id": "test-direct-sku-001",
  "sku_name": "直连联调商品",
  "img_urls": ["http://host.docker.internal:8089/media/sku-001.jpg"]
}
```

直调写入 `visual-vending` 的 SKU 库。它不会创建 Java 的 `open_sku` 记录，商品页也不会自动出现这条 SKU。

## 识别业务提交

1. 从平台商品库选取当前商户的已审核 SKU，将其识别侧 SKU ID 填到 `skuIds`。调用 `POST http://localhost:10070/openInterface/open/recognition/create`。视频提交的请求体示例如下。

   ```json
   {
     "recogId": "test-platform-recog-001",
     "containerId": "862337050531238",
     "resourceType": 1,
     "resourceUrls": ["http://host.docker.internal:8089/media/2-1.mp4"],
     "skuIds": ["862337050531238-2"],
     "notifyUrl": "http://test-callback.example/notify",
     "callbackUrl": "http://test-callback.example/recognition"
   }
   ```

2. 自动识别模式下，Java 先创建 `visual_recognition_record`，再签名调用 `18768` 的 `POST /openapi/sc/v1/recognition`，并把返回的 `data.ksRecogId` 关联到平台记录。
3. 在 [识别记录页面](http://localhost:18070/records) 按单号查看记录。识别服务回调 Java 的 `POST /openInterface/open/recognition/notifyRecog` 后，Java 按 `ksRecogId` 更新已有记录。满足转发条件时，Java 再向商户的 `callbackUrl` 发送结果。

## 识别直接测试

1. 签名调用 `POST https://127.0.0.1:18768/openapi/sc/v1/recognition`。视频地址使用 `resource_type: 1`。将 `origin_time` 替换为本次提交的 13 位毫秒时间戳，并将 `extra_params.sku_scope` 中的 ID 替换为当前签名账户可用于识别的 SKU。需要验证回调时可加入 `notify_url`。

   ```json
   {
     "recog_id": "test-direct-recog-001",
     "container_id": "862337050531238",
     "resource_type": 1,
     "resource_urls": ["http://host.docker.internal:8089/media/2-1.mp4"],
     "extra_params": {
       "origin_time": 1795482000000,
       "sku_scope": [{"sku_id": "862337050531238-2"}]
     }
   }
   ```

2. 用响应的 `data.ks_recog_id` 签名调用 `GET https://127.0.0.1:18768/openapi/sc/v1/recognition/{ks_recog_id}`。`data.state.code` 为 `1` 表示处理中，为 `2` 表示处理完成。再核对 `data.result.status.code`、`data.result.result` 中的 `ks_sku_id` 和 `sku_num`，确认识别结果及数量。

直调只创建 `visual-vending` 的识别任务。即使把回调发给 Java，回调也只更新已有记录，因此该任务不会自动出现在平台的识别记录页面。

## 直调签名与证书

直调请求使用 `Content-Type: application/json`、`Referer`、毫秒 `X-Request-Date` 和 `Authorization: HMAC-SHA1 <AK>:<signature>`。签名使用实际请求的 Host、路径、查询参数、相关请求头及原始请求体。使用本地自签证书时，将 `data/tls.crt` 加入客户端信任库，或在 `curl` 中通过 `--cacert data/tls.crt` 指定。签名实现和调用示例见 [vending/protocol.py](vending/protocol.py) 与 [tests/test_compatibility.py](tests/test_compatibility.py)。
