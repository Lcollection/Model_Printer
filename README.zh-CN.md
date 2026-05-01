<p align="center">
  <img src="assets/model-printer-icon.svg" alt="Model Printer icon" width="112">
</p>

<h1 align="center">Model Printer</h1>

> English documentation: [README.md](README.md)

Model Printer 是一个模型结构查看器。它可以导入 PyTorch `.pth` / `.pt`
权重文件或 NumPy `.npz` 权重文件，从参数名称和张量形状推断模型层级结构，并导出可在
draw.io / diagrams.net 中继续编辑的 `.drawio` 架构图。

它适合处理只有权重文件、没有模型源码的情况。由于 `.pth` 权重通常不保存
`stride`、`padding`、激活函数、forward 分支等完整动态图信息，本工具会尽量从参数
形状和命名习惯中恢复结构；如果需要 100% 精确的 forward 图，仍然需要模型源码或
ONNX / TorchScript 等带计算图的格式。

## 功能

- 读取 PyTorch `.pth` / `.pt` checkpoint。
- 读取 NumPy `.npz` 权重归档。
- 支持直接输入 Hugging Face 在线模型链接。
- 对 Hugging Face safetensors 模型优先读取在线元数据，避免下载完整权重。
- 自动识别常见 checkpoint 字段：`state_dict`、`model_state_dict`、`model`、
  `net`、`module` 等。
- 支持 DataParallel 常见的 `module.` 前缀自动剥离。
- 按参数键名生成层级树，展示每层参数形状和参数量。
- 根据权重形状推断常见层类型，例如 `Conv2d`、`Linear`、`Embedding`、
  `BatchNorm`、`LayerNorm`。
- 对连续重复、结构相同的层进行简写，例如 `0..11 x12`。
- 导出 `.drawio` 文件，可直接用 draw.io / diagrams.net 打开编辑。

## 安装

基础安装支持 Windows、macOS、Linux。基础依赖包含 `huggingface-hub`、
`numpy` 和 `rich`，可以打开 TUI、读取 `.npz`、读取 Hugging Face
safetensors 元数据、导出 `.drawio`。

Windows PowerShell：

```powershell
cd E:\github\modal_printer
python -m pip install -e .
```

macOS / Linux：

```bash
cd /path/to/modal_printer
python -m pip install -e .
```

如果需要读取 PyTorch `.pth` / `.pt` 文件，请额外安装 PyTorch 支持：

Windows PowerShell：

```powershell
python -m pip install -e ".[pytorch]"
```

macOS / Linux：

```bash
python -m pip install -e '.[pytorch]'
```

如果你需要 CUDA、MPS 或特定 CPU 版本，请优先按照
[PyTorch 官网](https://pytorch.org/) 给出的对应平台命令安装 `torch`。

## 使用

项目安装后会提供两个等价命令：`model_printer` 和 `model-printer`。

只输入命令会打开一个类似 Vim 的开屏界面：

```powershell
model_printer
```

在开屏界面中可以输入：

```text
:open E:\models\model.npz
```

也可以直接输入 Hugging Face 模型链接：

```text
:open https://huggingface.co/google-bert/bert-base-uncased
```

也可以按 `o` 快速进入 `:open` 命令，或按 `q` 退出。

查看 PyTorch 权重结构并导出 draw.io：

```powershell
model_printer path\to\model.pth -o model.drawio
```

macOS / Linux 路径示例：

```bash
model_printer /path/to/model.pth -o model.drawio
```

查看 NumPy `.npz` 权重结构并导出 draw.io：

```powershell
model_printer path\to\model.npz -o model.drawio
```

打开可视化 TUI 界面：

```powershell
model_printer path\to\model.pth --tui
```

直接查看 Hugging Face 在线模型：

```powershell
model_printer https://huggingface.co/google-bert/bert-base-uncased --tui
```

仅打印结构，不导出：

```powershell
model_printer path\to\model.pth --no-drawio
```

保留 `module.` 前缀：

```powershell
model_printer path\to\model.pth --keep-module-prefix
```

手动剥离某个统一前缀：

```powershell
model_printer path\to\model.pth --strip-prefix backbone.
```

调整重复层合并阈值，默认连续出现 2 个相同结构就合并：

```powershell
model_printer path\to\model.pth --min-repeat 3
```

有些旧 checkpoint 保存了完整 Python 对象，而不是纯 `state_dict`。默认加载会优先使用
PyTorch 的安全权重模式；如果你确认文件可信，可以启用非安全 pickle 加载：

```powershell
model_printer path\to\model.pth --unsafe-load
```

## 平台支持

Model Printer 目标支持：

- Windows 10/11，PowerShell 或 Windows Terminal。
- macOS，系统 Terminal、iTerm2 或其他支持 ANSI 的终端。
- Linux，常见桌面终端或 SSH 终端。

跨平台细节：

- TUI 键盘读取在 Windows 使用 `msvcrt`，在 macOS / Linux 使用 `termios`。
- 路径参数使用 Python `pathlib` 处理，Windows 的 `E:\models\model.pth`
  和 POSIX 的 `/home/me/model.pth` 都可以。
- CI 会在 `ubuntu-latest`、`macos-latest`、`windows-latest` 上运行测试。
- 基础安装不强制安装 PyTorch，避免不同系统、Python 版本、CUDA 版本的 wheel
  差异影响 `.npz` 和 TUI 功能。

## 输出说明

终端会输出类似结构：

```text
Model [params=23.5M]
  stem: Conv2d out=64, in=3, k=7x7 [params=9.4K]
  layer1
    0..2 x3: Block [params=221.7K each]
      conv1: Conv2d out=64, in=64, k=3x3
      bn1: BatchNorm channels=64
```

生成的 `.drawio` 文件可以直接拖入 draw.io / diagrams.net，或通过
`File -> Open From -> Device` 打开。

## TUI 快捷键

`--tui` 会打开一个基于终端的可视化浏览器。左侧是可折叠的模型结构树，右侧是当前
选中层的参数形状、参数量、子层摘要和 draw.io 导出路径。

| 快捷键 | 功能 |
| --- | --- |
| `↑` / `↓` 或 `j` / `k` | 移动选中层 |
| `Enter` / `Space` | 展开或折叠当前层 |
| `←` / `→` 或 `h` / `l` | 折叠当前层或展开当前层 |
| `PageUp` / `PageDown` 或 `u` / `d` | 快速翻动 |
| `a` | 展开全部 |
| `c` | 折叠到根节点 |
| `e` | 导出当前完整结构为 `.drawio` |
| `q` | 退出 |

## 设计限制

`.pth` 权重文件本质上通常只是参数表。本工具根据这些信息做静态推断，因此：

- 可以可靠展示参数层级、参数形状、参数量。
- 可以较好识别常见层类型。
- 无法从普通 `state_dict` 中恢复没有参数的层，例如 `ReLU`、`Dropout`、`Flatten`。
- 无法精确恢复所有 forward 分支、张量尺寸变化、skip connection 的真实连线。
- `.npz` 文件需要把每个数组以参数名保存为 key，例如 `backbone.0.conv.weight`。
  如果 key 使用 `/` 分隔，例如 `backbone/0/conv/weight`，工具会自动转成点号层级。

如果未来需要更完整的计算图，可以在此基础上增加 ONNX / Torch FX 导入器。
