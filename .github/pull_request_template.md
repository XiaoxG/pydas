## 变更说明 / Summary

<!-- 一两句话：改了什么、为什么。 -->

## 类型 / Type

- [ ] 缺陷修复 / Bug fix
- [ ] 新功能 / Feature（不改冻结 API 的名字）
- [ ] 文档 / Docs
- [ ] 测试 / Tests
- [ ] 其它 / Other

## 检查 / Checklist

- [ ] 未改公开方法名、历史驼峰参数（`chName`、`cutoffull`、`updateST` 等）
- [ ] 未改二进制 `.out` pack 布局（`src/pydas/core/io_format.py`）
- [ ] 行为有变时，已同时改 `pyproject.toml` 的 version、`pydas.__version__`、`CHANGELOG.md`
- [ ] `pytest` 可通过（与 CI 一致：Python 3.11 / 3.12）
- [ ] 代码注释 / log / docstring 为英文；面向同事的说明为中文

## 关联 Issue / Linked issue

<!-- 例如：Fixes #123 或 Relates to #123 -->
