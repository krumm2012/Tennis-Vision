# 第一轮阶段 0–1 验收记录

日期：2026-09-30。完成真实上传 → 标准化 → SSH 云推理 → 哈希校验 → 打包 → 浏览器播放。RTX 4090，复用已有 SAM 环境。任务数据位于被 Git 忽略的 output/video_library，模型和视频不提交到仓库。

## 两段非默认视频

第一段来自 input_videos/input_video_2.mp4 的前 10 秒；第二段来自 input_videos/input_video.mp4 的前 3 秒，缩放为 960×540 后上下填充至 960×720，用于 4:3 尺寸适配测试。均是实际 GPU 推理，并非默认网格或合成姿态。

| Dataset ID | 标准化尺寸 | 25 fps 帧数 | 推理脚本耗时 | 画面内顶点比例 | 最大稳定位移 |
| --- | --- | --- | --- | --- | --- |
| `dbea159caf964785aec007014fb79dd0` | 1280×720 | 250 | 191.6 秒 | 99.94% | 9.50 mm |
| `092dca1d6d824ff7a9a12cf56fe810e6` | 960×720 | 75 | 69.8 秒 | 100.00% | 9.20 mm |

画面内顶点比例只核对相机投影边界；不能作为可见表面纹理覆盖率或模型精度。耗时包含模型载入和逐帧处理，不包含上传、打包及模型文件哈希计算。

浏览器检查：两段均显示本视频人体纹理，连续播放到末帧；第一段结束后能重新播放，原始/稳定切换和前后视角可用。不可见部位灰色仍存在，未声称完整背面贴图。截图保存在 output/gpu-first-round-back.png、output/gpu-second-round-front.png、output/gpu-second-round-back.png。

## 可追溯数据

- `dbea159caf964785aec007014fb79dd0`：remote_job_id `8408c985e1c9443bbb4eda132050c414`；源视频 SHA-256 `b1e464599c058457d9461cf58c214e624c79af9074e533d197f242ee347c9711`；NPZ SHA-256 `63c2ef57e623e2eee5817f9aad94b8629d70234b6ecbc123c0859483a351ded0`。
- `092dca1d6d824ff7a9a12cf56fe810e6`：remote_job_id `3ff36f239e6049df85a454745b74cfe6`；源视频 SHA-256 `6e3ceea2f5786db49f3be3bc6cd4ab1f7ed4709ecbd4f91111e3cb6d8f38f7e3`；NPZ SHA-256 `9bac310f52b39ffc8b8c49b0245a97e967dad7baf9fb04bd37482d54a13ce803`。

模型 SHA-256：

- checkpoint_sha256：`3b1cb897f4bbd977bf81cbb0b30780a9582681ac642ee112865790ceb4d66056`
- model_config_sha256：`d2e772e108b8727e9367681845fecb32806144acd0debc20868d100689470570`
- segmentation_sha256：`55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`

生成时 Git 基准为 650f6ed，组件源码当时尚未提交。第一任务 Worker 尚未加入 code_sha256 字段，不能把 Git 基准视作全部推理源码；第二任务清单已记录三个实际部署 Python 文件的哈希。本轮结束将提交组件与文档并重新部署供后续任务使用。

## 验证与边界

- 8 项本地测试通过：独立上传、Range、默认入口、数据合同、打包、失败/重启、SSH 配置检查、返回哈希保护。GPU 推理由上面真实任务另行验收。
- 服务仅对本机开放，保留逐次尝试的清单及日志。远端任务取消、断线自动恢复和自动清理暂未实现。
- 新视频目前只有人体、遮罩、源视频投影纹理和轻量稳定；球拍、球、镜面标定、实测场地及教学报告待后续阶段。
- graphify 知识图谱更新命令已尝试，但本机缺少 graphify 模块，无法重建；不影响 GPU 推理和 Viewer。

## 用户提供的新视频：48.43.mp4

来源：`/Users/krum5539/Desktop/Camera/2026-09-30/48.43.mp4`；原始 2560×1440 / 50 fps / 约 9.97 秒。标准化为 1280×720 / 25 fps / 249 帧。

- Dataset ID：`85ade7a072984579831f5cb76e8e5fd3`。
- Remote job：`e87b68265506426886b794d5c96af879`。
- 源视频 SHA-256：`5ca38c94d1f432936196a332fbe8f5576d673034d8f15229305f71e989c158d0`。
- NPZ SHA-256：`8516c7bb13f5d87fe8748f0ed062e6404cd92aba3db8ba25b36c494fdc7550a1`。
- 推理耗时：188.8 秒；画面内顶点比例 100.00%；最大稳定位移 9.54 mm。
- Viewer：[新视频结果](http://127.0.0.1:18768/datasets/85ade7a072984579831f5cb76e8e5fd3/result/viewer.html)。

浏览器首帧确认跟踪前景真人，显示衣服/皮肤/鞋子纹理；源视频中的镜中人未进行镜面标定。检查连续播放到第 249 帧、结束后重播、原始/稳定及前后视角切换。验收截图保存在 output/gpu-camera-4843-front.png 和 output/gpu-camera-4843-back.png。
