"""`comfy_api.latest.io` 的最小替身，供 StableGen 指定的 TRELLIS.2 节点包使用。

来源核对（PozzettiAndrea/ComfyUI-TRELLIS2 pin 9b878516）：
- 节点只在 `define_schema()` 里用它声明输入/输出，在 `execute()` 里用
  `io.NodeOutput(...)` 包装返回值；
- `io.*` 全部不参与数值计算（真正的计算在 nodes/stages.py 与
  nodes/trellis2/* 里）；
- 真实的 `comfy_api.latest` 会拉 comfy_execution / PyAV / 视频编解码整条链
  （本项目 meow3d/runtime.py 已因此做过一次替身），所以这里也给等价描述层。

需要的属性清单来自节点包实际用到的符号：
ComfyNode / Schema / NodeOutput / Custom / Image / Mask / Combo /
DynamicCombo(.Option) / Boolean / Int / Float / String / Model / Latent /
Conditioning，且每个类型都要有 .Input() / .Output()。
"""
from __future__ import annotations

from typing import Any, List, Optional

MISSING = object()


class Input:
    """一个输入槽；执行器靠 name / default / dynamic 填参数。"""

    def __init__(self, name: str, *, parent: Any = None, default: Any = MISSING,
                 options: Any = None, optional: bool = False, **kw: Any):
        self.name = name
        self.parent = parent
        self.default = default
        self.options = options
        self.optional = optional
        self.kw = kw
        # DynamicCombo 把子输入藏在 Option 里；执行器要把 "on" 展开成 dict。
        self.dynamic = parent is DynamicCombo and options is not None


class Output:
    def __init__(self, *, parent: Any = None, display_name: str = "", **kw: Any):
        self.parent = parent
        self.display_name = display_name
        self.kw = kw


class _Type:
    def __init__(self, io_type: str = ""):
        self.io_type = io_type

    def Input(self, name: str, **kw: Any) -> Input:
        return Input(name, parent=self, **kw)

    def Output(self, **kw: Any) -> Output:
        return Output(parent=self, **kw)


class Option:
    def __init__(self, name: str, inputs: Optional[List[Input]] = None):
        self.name = name
        self.inputs = inputs or []


Image = _Type("IMAGE")
Mask = _Type("MASK")
Boolean = _Type("BOOLEAN")
Int = _Type("INT")
Float = _Type("FLOAT")
String = _Type("STRING")
Model = _Type("MODEL")
Latent = _Type("LATENT")
Conditioning = _Type("CONDITIONING")
Combo = _Type("COMBO")
DynamicCombo = _Type("DYNAMICCOMBO")
DynamicCombo.Option = Option  # type: ignore[attr-defined]


def Custom(io_type: str) -> _Type:
    return _Type(io_type)


class Schema:
    def __init__(self, *, node_id: str = "", display_name: str = "",
                 category: str = "", description: str = "",
                 inputs: Optional[List[Input]] = None,
                 outputs: Optional[List[Output]] = None,
                 is_output_node: bool = False, **kw: Any):
        self.node_id = node_id
        self.display_name = display_name
        self.category = category
        self.description = description
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.is_output_node = is_output_node
        self.kw = kw


class NodeOutput:
    """对应 comfy_api/latest/_io.py:2510-2541。"""

    def __init__(self, *args: Any, ui: Any = None, expand: Any = None,
                 block_execution: Any = None):
        self.args = args
        self.ui = ui
        self.expand = expand
        self.block_execution = block_execution

    @property
    def result(self):
        return self.args if len(self.args) > 0 else None

    def __getitem__(self, index: int):
        return self.args[index]

    def __iter__(self):
        return iter(self.args)

    def __len__(self):
        return len(self.args)


class ComfyNode:
    @classmethod
    def define_schema(cls) -> Schema:
        raise NotImplementedError

    @classmethod
    def execute(cls, **kwargs: Any) -> NodeOutput:
        raise NotImplementedError
