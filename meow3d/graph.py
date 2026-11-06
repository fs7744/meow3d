"""在进程内执行 ComfyUI **UI 格式**工作流（nodes + links）。

为什么不转成 API 格式手写一遍：visualbruno 的 MeshWithTexturing 有 24 个节点、
每个 10~25 个参数，手抄必错。这里按 `INPUT_TYPES()` 的顺序把 `widgets_values`
映射回参数名，连线值再覆盖同名参数（ComfyUI 前端的语义就是这样：被连线的
widget 仍然留在 widgets_values 数组里）。

内置小节点（PrimitiveInt / StringConcatenate / Preview3D）不 import ComfyUI
的整套节点，直接在这里给等价实现。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import runtime

# 这些是 ComfyUI 前端/内置节点，不在 ComfyUI-Trellis2 的映射表里
BUILTIN = {"PrimitiveInt", "PrimitiveString", "PrimitiveFloat",
           "StringConcatenate", "Preview3D", "Note", "MarkdownNote"}
# 纯展示节点：不执行（返回空），但仍参与拓扑，保证它依赖的导出节点先跑
NOOP = {"Preview3D", "Note", "MarkdownNote"}

# ComfyUI 的 seed widget 会额外塞一个 control_after_generate 值
_CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}

_WIDGET_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN"}


class WorkflowError(RuntimeError):
    pass


def load_workflow(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise WorkflowError(f"找不到工作流：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _widget_slots(cls: Any) -> List[str]:
    """按前端 widget 顺序返回参数名（含被连线的，值会被连线覆盖）。"""
    spec = cls.INPUT_TYPES()
    slots: List[str] = []
    for group in ("required", "optional"):
        for name, cfg in (spec.get(group) or {}).items():
            t = cfg[0] if isinstance(cfg, (list, tuple)) else cfg
            if isinstance(t, list) or t in _WIDGET_SCALARS:
                slots.append(name)
    return slots


def _widget_kwargs(cls: Any, widgets: List[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    vals = list(widgets or [])
    i = 0
    for name in _widget_slots(cls):
        if i >= len(vals):
            break
        v = vals[i]
        i += 1
        if isinstance(v, str) and v in _CONTROL_VALUES:
            # control_after_generate：下一个值才是这个 widget 的真值
            if i >= len(vals):
                break
            v = vals[i]
            i += 1
        out[name] = v
    return out


def _normalize_outputs(res: Any) -> List[Any]:
    """节点可能返回 tuple / 单值 / 新版 {'ui':..., 'result':...}。"""
    if isinstance(res, dict) and "result" in res:
        res = res["result"]
    if isinstance(res, tuple):
        return list(res)
    if res is None:
        return []
    return [res]


def _builtin(node: Dict[str, Any], linked: Dict[str, Any]) -> List[Any]:
    t = node["type"]
    wv = node.get("widgets_values") or []
    if t in NOOP:
        return []
    if t == "PrimitiveInt":
        return [int(wv[0])]
    if t == "PrimitiveFloat":
        return [float(wv[0])]
    if t == "PrimitiveString":
        return [str(wv[0])]
    if t == "StringConcatenate":
        parts = [str(v) for v in linked.values()]
        parts += [str(v) for v in wv if isinstance(v, str)]
        # 去掉 control 之类不该参与拼接的残留
        parts = [p for p in parts if p not in _CONTROL_VALUES]
        return ["".join(parts)]
    return []


def _link_index(wf: Dict[str, Any]) -> Dict[int, List[int]]:
    """link_id -> [origin_node, origin_slot, target_node, target_slot, type]"""
    return {int(l[0]): list(l) for l in wf.get("links", [])}


def plan(wf: Dict[str, Any], *, overrides: Optional[Dict[int, List[Any]]] = None,
         kwargs_overrides: Optional[Dict[int, Dict[str, Any]]] = None
         ) -> List[Dict[str, Any]]:
    """只做参数映射预览，不执行节点。用于核对 widgets 对齐，跑真模型前先看一眼。

    kwargs_overrides 会像 execute() 一样覆盖 widget，方便 --plan 显示真正会
    被执行的参数（例如按 StableGen 默认值对齐后的 texture_size）。
    """
    boot = runtime.bootstrap()
    mappings = boot["mappings"]
    links = _link_index(wf)
    kwargs_overrides = kwargs_overrides or {}
    out = []
    for node in wf.get("nodes", []):
        t, nid = node["type"], int(node["id"])
        if overrides and nid in overrides:
            out.append({"id": nid, "type": t, "override": True, "kwargs": {}})
            continue
        linked: Dict[str, Any] = {}
        for inp in node.get("inputs", []):
            lk = inp.get("link")
            if lk is not None and int(lk) in links:
                l = links[int(lk)]
                linked[inp["name"]] = f"<from {l[1]}#{l[2]}>"
        if t in BUILTIN:
            kwargs = {**linked, "widgets_values": node.get("widgets_values")}
        elif t in mappings:
            kwargs = {**_widget_kwargs(mappings[t], node.get("widgets_values") or []),
                      **linked}
            if nid in kwargs_overrides:
                kwargs.update(kwargs_overrides[nid])
        else:
            kwargs = {"??UNKNOWN_NODE??": t}
        out.append({"id": nid, "type": t, "kwargs": kwargs})
    return out


def execute(wf: Dict[str, Any], *, overrides: Optional[Dict[int, List[Any]]] = None,
            kwargs_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
            on_node=None) -> Dict[str, Any]:
    """按拓扑执行。

    overrides: {node_id: [输出0, 输出1, ...]} 可短路任意节点（用来注入图片）。
    kwargs_overrides: {node_id: {参数名: 值}} 覆盖 widget（seed / 文件名等）。

    返回 {results, exports}，exports 是所有 Trellis2ExportMesh 的首个输出路径。
    """
    boot = runtime.bootstrap()
    mappings = boot["mappings"]
    overrides = overrides or {}
    kwargs_overrides = kwargs_overrides or {}
    links = _link_index(wf)
    nodes = {int(n["id"]): n for n in wf.get("nodes", [])}

    # 依赖：所有连线的来源节点（被跳过的展示节点也保留依赖）
    deps: Dict[int, set] = {nid: set() for nid in nodes}
    for l in links.values():
        src, dst = int(l[1]), int(l[3])
        if dst in deps:
            deps[dst].add(src)

    order: List[int] = []
    remaining = dict(deps)
    while remaining:
        ready = [n for n, d in remaining.items() if not d]
        if not ready:
            raise WorkflowError("工作流有环或断线：" + ", ".join(map(str, remaining)))
        for n in sorted(ready):
            order.append(n)
            del remaining[n]
        for d in remaining.values():
            d.difference_update(ready)

    results: Dict[int, List[Any]] = {}
    exports: List[str] = []

    for nid in order:
        node = nodes[nid]
        t = node["type"]
        if nid in overrides:
            results[nid] = list(overrides[nid])
            continue

        linked: Dict[str, Any] = {}
        for inp in node.get("inputs", []):
            lk = inp.get("link")
            if lk is None:
                continue
            l = links.get(int(lk))
            if l is None:
                continue
            src, slot = int(l[1]), int(l[2])
            vals = results.get(src, [])
            linked[inp["name"]] = vals[slot] if slot < len(vals) else None

        if on_node:
            on_node(nid, t, "start")

        if t in BUILTIN:
            res = _builtin(node, linked)
        elif t in mappings:
            cls = mappings[t]
            kwargs = {**_widget_kwargs(cls, node.get("widgets_values") or []), **linked}
            if nid in kwargs_overrides:
                kwargs.update(kwargs_overrides[nid])
            # 只传节点声明过的参数，避免前端残留 widget 造成 TypeError
            allowed = set(_widget_slots(cls))
            for group in ("required", "optional"):
                allowed |= set((cls.INPUT_TYPES().get(group) or {}).keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            fn = cls.FUNCTION
            try:
                obj = cls()
                method = getattr(obj, fn)
            except Exception:  # classmethod / staticmethod
                method = getattr(cls, fn)
            res = _normalize_outputs(method(**kwargs))
        else:
            raise WorkflowError(f"未注册的节点类型：{t} (id={nid})")

        results[nid] = res
        if t == "Trellis2ExportMesh" and res:
            exports.append(str(res[0]))
        if on_node:
            on_node(nid, t, "done")

    return {"results": results, "exports": exports}
