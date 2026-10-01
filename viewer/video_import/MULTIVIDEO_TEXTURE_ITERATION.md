# 多视频人体纹理与可靠握拍：2026-10-01

## 目标与边界

源目录 `/Users/krum5539/Desktop/Camera/2026-09-30` 包含 9 段约 10 秒的视频。抽样画面为同一人物、衣着和固定镜面场地；这些是不同姿态下的观测，并不是同时拍摄的多个标定相机。原始文件均为 2560×1440；容器帧率字段为 50/1，实际平均帧率约 25，归一化视频按 25 fps 分析。

本轮分别保存各视频的原片、归一化视频、真人与镜中 SAM3D、原生 MHR 参数、SAM2 遮罩、来源哈希、独立镜面校验。通过共享 MHR 顶点/三角形/UV 对应积累外观，不直接合并不同姿态的相机坐标。不能由这些素材确认实测身体尺寸、真实握柄棱位或照片级任意视角。

## 可重复执行

```bash
# 已预装模型环境：部署版本化代码，检查 CUDA/权重；私有配置不进 Git。
python3 -B deploy/3dpose/cloud_gpu/deploy.py \
  --config deploy/3dpose/cloud_gpu/host.local.json

# 一张 GPU 串行处理；同一视频的已有任务按 job id 恢复收取。
python3 -B viewer/video_import/batch_multivideo.py \
  --source-dir /Users/krum5539/Desktop/Camera/2026-09-30 \
  --output output/multivideo_texture \
  --config deploy/3dpose/cloud_gpu/host.local.json \
  --mirror-model /Users/krum5539/Downloads/yolo26m-pose.pt \
  --seed-dataset output/video_library/85ade7a072984579831f5cb76e8e5fd3 \
  --seed-native output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v6/native

# 所有输入完成后才允许正式跨视频合成。
python3 -B viewer/video_import/multivideo_texture.py \
  --batch output/multivideo_texture \
  --layout output/multivideo_texture/mhr_uv.npz \
  --output output/multivideo_texture/fusion --size 2048
```

模型自带 UV 由 `export_mhr_uv.py` 从已安装的 `assets/mhr_model.pt` 导出，不能使用不匹配的网上 UV 模板。导出文件包括 faces、uv、uv_faces、rest_vertices 和模型哈希；每个视频的 faces 必须逐项一致。云脚本存在 GPU 文件锁，独立重拟合与批处理均串行执行。局部 SSH 读取失败会有限重试；远端结果已成功的情况只恢复拉取，不重复推理。

## 本地数据与追溯

- `output/multivideo_texture/inventory.json`：9 个原始文件哈希、尺寸、帧率与时长。
- `scenes.jpg`：跨视频人物/衣着抽样。
- `batch_manifest.json`：各片段状态与远端 job id；失败原因保留，重跑会从已有任务继续。
- `clips/<原文件名>/original.video` 与 `source.mp4`：原始与分析时间轴。
- `clips/<原文件名>/attempts/0001/work/`：验证后的 reconstruction.npz、remote_job.json、remote_manifest.json、worker.log。
- `clips/<原文件名>/result/`：独立人体 Viewer、SAM2 遮罩、镜面估计与留出报告。
- `fusion/`：UV PNG、逐像素 clip/frame/view/face 来源、外观模板和纹理质量报告。
- 这些生成数据在 ignored output 中保留，不将视频和模型权重提交 Git。

## 纹理合成

在原分辨率视频中取色。每 5 帧抽样，每 25 帧留出；仅采用 SAM2 高置信度内部、正深度、可见表面、非掠射角和较清晰观测。三角形选帧加上相邻网格面的视角一致性约束，降低逐三角形切换造成的接缝。原始 UV 的重心坐标确定每个纹理像素对应的表面位置。跨视频曝光调整依据同一表面重叠区域，限制调整幅度。

未知纹理像素保持透明。边缘 RGB 扩展仅用于过滤接缝，不增加 alpha，也不计入有效覆盖。PNG 分辨率是容器尺寸，不表示已恢复相同数量的真实细节。合成报告分别记录面覆盖、纹理像素覆盖、镜面贡献和留出颜色残差；这些都不是三维精度或照片级真实性指标。原画面中人物尺寸和衣物随动作滑动会限制最终效果。

`publish_texture_review.py` 在独立本地 dataset 中生成预览，支持单视频基线、合成候选和灰模，使用同一段姿态播放。它拒绝覆盖既有非纹理评审 dataset。当前已验收的 Viewer 和 bakewell.cloud 未被纹理候选替换。

## 握拍与球拍

可复用 skill：`skills/tennis-racket-fitting/SKILL.md`，已安装到 `/Users/krum5539/.codex/skills/tennis-racket-fitting`。它沿用 demo 的 MCP/PIP 掌内握点与有向杆轴，再加入真假人/镜中轮廓关联、稀疏标注、SO(3) 时序约束、接触代理和留出质量门槛。

Viewer 新增左侧、右侧、斜视、握拍特写，清晰帧入口支持原分辨率局部放大。旋转多个视角只能暴露几何冲突，仍须与原视频/镜面重投影一起验收。已有尺寸为标准资产假设，measured=false；握柄半径与握点位置也不等于实测值。手指关节到假设圆柱的距离不是网格表面接触或真实棱位。

旧档案没有原生 MHR 参数，必须在推理时补采。新的原生回放与源 vertices/joints 的最大差约 7.2e-7 米。首个全身重拟合任务被相机一致性检查拦下：极少数帧相机平移最大变化 4.3 mm，超过 2 mm 门槛。独立 repacked 目录重新采用新相机和本片镜面校验，刚体球拍初值仅随根平移变化转移。

真实云重拟合 job `8ba0f427d4a64e778caa77a9504e8260` 已完成 120 步。掌内握点间隙中位数 6.45→0.44 mm；原始手部/杆轴夹角中位数 32.18→34.01°；留出图像误差 15.239→15.249 规范像素。因此候选未通过数值门槛，未发布到 Viewer。不能只用间隙变小宣布握拍正确。候选和报告保存在当前 dataset 的 `iterations/grip_refit_v6/gpu-fit-repacked/`，原相机不一致任务也保留日志。

## 检验与资料

Python 测试、浏览器检查和实际 GPU 运行分别记录；合成候选要检查正背面、侧面、接缝、衣物边界、快速挥拍和末帧。图谱重建命令在本机因缺少 graphify 模块无法运行，未安装额外依赖。

原生接口与拓扑参考：[SAM 3D Body 官方代码](https://github.com/facebookresearch/sam-3d-body)、[MHR 官方代码](https://github.com/facebookresearch/MHR)。[Tex2Shape 论文](https://arxiv.org/abs/1904.08645) 将可见的局部纹理作为人体细节重建输入；本项目本轮使用真实观测投影，不将模型推断的隐藏内容标为观测。
