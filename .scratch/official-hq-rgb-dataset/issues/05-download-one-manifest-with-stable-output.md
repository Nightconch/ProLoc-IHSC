# 05 — 稳定下载单份清单并生成训练 CSV

**What to build:** 让操作者能够处理一份 URL 清单，得到命名稳定的合格图片、只含成功记录的训练 CSV 和完整失败明细；即使并发完成顺序不同，正式结果仍与输入行顺序一致。

**Blocked by:** 04 — 将图片规范化为经过复验的 RGB JPEG

**Status:** done

- [x] 每条输入记录先验证下载所需字段，并按既有 `ProteinId-imageStem-AntibodyId-locations.jpg` 契约推导最终文件名。
- [x] 在发起任何网络请求前检查非法名称和最终文件名碰撞；发现碰撞时停止且绝不覆盖其他记录。
- [x] 下载可以并发执行，但每个任务携带自己的源行上下文，成功 CSV、失败清单和图片对应关系按输入行顺序归位。
- [x] 不同 `Protein Id` 的相同 URL 或相同图像内容均可分别下载和保留，只要最终名称不发生碰撞。
- [x] 每个成功行都对应一个经过复验的磁盘图片，每个正式图片也恰好对应一个成功 CSV 行。
- [x] 最终 CSV 严格使用九列顺序：`File Name`、`locations`、`cytoplasm`、`endoplasmic reticulum`、`mitochondria`、`nucleus`、`plasma membrane`、`Sequence`、`Protein Id`。
- [x] 标签和 sequence 沿用对应源行，`Protein Id` 固定位于最后一列，正式 CSV 不混入来源、质量转换或失败原因等审计列。
- [x] 每条失败记录至少包含质量层级、split、源行、`Protein Id`、URL、失败阶段和原因，便于定位与重试。
- [x] 端到端测试通过可控本地 HTTP 响应覆盖乱序完成、重复 URL、HTTP 错误、图像失败、名称碰撞和最终文件/CSV 一一对应。
