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
  --output output/multivideo_texture/fusion_independent --size 2048 \
  --mirror-source independent
```

模型自带 UV 由 `export_mhr_uv.py` 从已安装的 `assets/mhr_model.pt` 导出，不能使用不匹配的网上 UV 模板。导出文件包括 faces、uv、uv_faces、rest_vertices 和模型哈希；每个视频的 faces 必须逐项一致。云脚本存在 GPU 文件锁，独立重拟合与批处理均串行执行。局部 SSH 读取失败会有限重试；远端结果已成功的情况只恢复拉取，不重复推理。

## 本地数据与追溯

- `output/multivideo_texture/inventory.json`：9 个原始文件哈希、尺寸、帧率与时长。
- `scenes.jpg`：跨视频人物/衣着抽样。
- `batch_manifest.json`：各片段状态与远端 job id；失败原因保留，重跑会从已有任务继续。
- `clips/<原文件名>/original.video` 与 `source.mp4`：原始与分析时间轴。
- `clips/<原文件名>/attempts/0001/work/`：验证后的 reconstruction.npz、remote_job.json、remote_manifest.json、worker.log。
- `clips/<原文件名>/result/`：独立人体 Viewer、SAM2 遮罩、镜面估计与留出报告。
- `fusion/`：平面反射对照；`fusion_independent/`：独立镜中 SAM3D 取色候选。两者均有 UV PNG、逐像素 clip/frame/view/face 来源、外观模板和纹理质量报告。
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

多角度复查固定在第 18、66、186 帧（从 1 计数），每帧检查正、背、左、右、斜视，共 15 张有效截图。独立浏览器标签页在每次截图前核对当前帧，保留用户正在编辑的原标签页。截图及哈希在 `output/grip-multiview/audit_report.json`，拼图在 `checked-multiangle-sheet.jpg`。握点大致位于手掌附近，但侧面仍暴露杆轴与手部方向冲突，指尖包裹也没有得到验证。早期 `v4-186-*` 截图受页面操作影响、帧号混杂，已在报告标记为无效，不作为验收证据。默认 v4 球拍 SHA-256 保持 `20220f335990618508316a814a70ae1c82b072871205a53d4ebc7b6f5e684f12`。

后续握拍验收先补完整的真人拍柄/拍框与手部标注，再分离有向杆轴、无向拍面和物理面 A/B 的证据。使用同一组固定留出观测比较接触、重投影和抖动；只有三个维度同时过门槛才替换默认。圆柱接触代理通过后，仍需原生手部网格接触与人体/球拍联合优化，不能由目前的接触间隙推断握柄棱位。

## 检验与资料

Python 测试、浏览器检查和实际 GPU 运行分别记录；合成候选要检查正背面、侧面、接缝、衣物边界、快速挥拍和末帧。图谱重建命令在本机因缺少 graphify 模块无法运行，未安装额外依赖。

本轮 37 项 Python 测试和 `test_racket_layer.cjs` 通过。纹理数组由逐帧访问 NPZ 改为一次解压所需字段；重跑单视频 2048 UV 基线，PNG SHA-256 仍为 `c36358d9d26ef3aeb6a8f393f3ea1f519aab2970e1a5c4373ca1c97e6b64b16e`。基线面覆盖 90.60%，有效 UV 像素覆盖 89.73%，镜面贡献占有效像素 35.63%。这些是候选观测覆盖，不能作为重建准确率。

原生接口与拓扑参考：[SAM 3D Body 官方代码](https://github.com/facebookresearch/sam-3d-body)、[MHR 官方代码](https://github.com/facebookresearch/MHR)。[Tex2Shape 论文](https://arxiv.org/abs/1904.08645) 将可见的局部纹理作为人体细节重建输入；本项目本轮使用真实观测投影，不将模型推断的隐藏内容标为观测。

## 最终运行记录：全部 9 段已完成

2026-10-01，9 段全部完成真实云推理、回传哈希校验及本地包装，共 2,242 个归一化视频帧。每段保留真人 SAM3D 和原生 MHR 参数，镜中独立 SAM3D 共 2,159 个有效帧；真人与镜中 SAM2 均覆盖 2,242 个非空帧。全部 UV 拓扑与模型导出布局逐项一致。独立镜中缺帧保持无观测，不补造模型输出。48.53 的任务因 SSH 状态读取中断被本地标为失败，按原 job 恢复拉取后完成，未重复推理。

| 视频 | 帧数 | 镜中 SAM3D 有效帧 | GPU job |
| --- | ---: | ---: | --- |
| 48.43.mp4 | 249 | 245 | `cfecbc204433489d9cc7b20c8fa8cb1b` |
| 48.53.mp4 | 250 | 250 | `f36c38af38ff40ecb09085da1be78ee9` |
| 49.03.mp4 | 249 | 237 | `fecd0eb47f504a03852d24476e6a17ed` |
| 49.13.mp4 | 249 | 219 | `96a0d1d2985c4ae3b64b726ba1e0bcbb` |
| 49.23.mp4 | 249 | 237 | `87a0649275774fcc967e926cdedb66d8` |
| 49.33.mp4 | 249 | 239 | `a63d5e8a29af451d9d9506549924b958` |
| 49.43.mp4 | 249 | 245 | `4b8ef6852a3e4173be8ce83cad77e43d` |
| 49.53.mp4 | 249 | 249 | `eb63815180f24848813a0a78ca684bcd` |
| 50.03.mp4 | 249 | 238 | `371df50a344348d7850b56f618aaf943` |

全部原片/归一化片/档案 SHA-256、帧数、镜面留出误差、代码版本及目录保存在 `output/multivideo_texture/source_index.json`。云代码运行版本保留在每个 `remote_manifest.json`，本地 UV 版本以源码 SHA-256 标识；不是将最终本地提交冒充为先前的 GPU 运行版本。

## 两种镜中取色对照

`--mirror-source plane` 使用真人网格经标定平面反射的位置。`--mirror-source independent` 使用本视频真实云推理的 `mirror_vertices + mirror_roots`，原始画面 x 轴已在云端恢复，不再次翻转；焦距必须与同画面一致。两种方式都需要本片镜面校验、SAM2 和深度可见性。独立方式在 `mirror_valid=false` 的帧跳过镜中取色。单视频与 9 视频在各自方式内使用同一抽样和留出规则。

| 方法 | 视频数 | 面候选覆盖 | 有效 UV 覆盖 | 首段留出颜色 MAE / 255 | 邻面选帧切换 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 平面反射 | 1 | 90.60% | 89.73% | 11.67 | 20.32% |
| 平面反射 | 9 | 97.91% | 97.41% | 14.71 | 26.75% |
| 独立镜中 SAM3D | 1 | 88.32% | 87.61% | 11.00 | 20.30% |
| 独立镜中 SAM3D | 9 | 95.40% | 96.73% | 14.69 | 27.03% |

跨视频增大了覆盖，但首段颜色留出误差和邻面切换都变差。独立镜中方式仍未实现外观精度的明显提高，而且两种几何方式改变了留出表面采样位置，跨方式误差不属于固定真值精度对比。面覆盖与 UV 覆盖按不同分母计算，也不应直接互换。当前预览采用独立镜中方式以明确使用真实镜中推理与有效帧，仍标记为待复核。未知像素透明，镜面贡献占有效 UV 像素 25.62%。

本地候选：[纹理评审 Viewer](http://127.0.0.1:18769/datasets/a171956813c349bb8fa0911cfd1ae54b/result/viewer.html)。支持合成、相同方法的单视频基线、灰模；已检查正背面、左右侧、模式切换及 249/249 末帧。播放网格使用解码帧时间同步。截图为 `fusion-independent-front.png`、`fusion-independent-back.png`、`fusion-independent-back-final.png`；`mirror-source-comparison.jpg` 保留两种取色方式的同帧背面对照。

```sh
python3 -B viewer/video_import/publish_texture_review.py \
  --batch output/multivideo_texture \
  --fusion output/multivideo_texture/fusion_independent \
  --baseline output/multivideo_texture/baseline_independent \
  --layout output/multivideo_texture/mhr_uv.npz \
  --destination output/video_library/a171956813c349bb8fa0911cfd1ae54b
```

本轮无法由这些固定相机片段确认照片级精度。后续应先限制错误投影：复核镜面/主体身份和衣物边界，补一致的表面对应；再做跨视频颜色一致性及 UV 接缝优化，保留固定留出集进行比较。提升面部细节需要更近、更清晰的原始人体观测；2048 UV PNG 本身不会创造原片没有的细节。

本地 Docker 18769 已更新，18768 写入服务可用；已验收 85ade... 的球拍文件保持原 SHA-256，公网部署没有更新。37 项 Python 测试和球拍插值测试通过，HTML 脚本语法检查通过。graphify 重建依然因本机缺少模块失败；生成数据不进入 Git。
独立合成 PNG SHA-256：`fde36a1283ca8a35adcd2bcbff1b1097e18a70c9a4a051379e622bf17e6d1192`；UV 源码 SHA-256：`b270ced7322873ae427479c9ea4951ab77c85b8f5c26a5e642706fac2584a938`。
