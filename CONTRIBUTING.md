# Contributing

**English (short):** Please file issues with the GitHub forms (bug / feature / usage). Include PyDAS version, Python version, OS, a minimal snippet, the traceback, and a **small** data sample or a description of the file format. Development install is `pip install -e ".[dev]"` from a clone; CI runs `pytest` on Python 3.11 and 3.12. Do not rename public methods, historical camelCase parameters, or the binary `.out` pack layout. Versions are SemVer tags `vX.Y.Z` with Keep a Changelog entries. Pull requests are welcome; the maintainer (郭老师) typically triages issues and applies fixes.

---

## 谁会改代码

本仓库对外主要是**用起来、把问题写清楚**。郭老师会看 Issue 并修。欢迎发 Pull Request，但请先开 Issue 说明范围，尤其是会碰到冻结 API 的改动。

## 报告问题

请用模板，不要发空白 Issue：

| 类型 | 模板 |
|------|------|
| 行为不对 / 崩溃 | [缺陷报告](https://github.com/XiaoxG/pydas/issues/new?template=bug_report.yml) |
| 想要新能力 | [功能请求](https://github.com/XiaoxG/pydas/issues/new?template=feature_request.yml) |
| 不会用 / 参数含义 | [使用提问](https://github.com/XiaoxG/pydas/issues/new?template=usage_question.yml) |

缺陷报告请写全：**PyDAS 版本**、**Python 版本**、**操作系统**、**最小复现代码**、**完整 traceback**。涉及读文件、质量链或 Excel 报告时，请附很小的数据样本（数秒到一两分钟、少通道），或说明：文件类型（`.out` / CSV）、通道名、采样率、比尺 `lam`、段数。这是公开仓库，**不要**上传整次水池试验或未脱敏数据。

提问前请先看 [`docs/user-guide.md`](docs/user-guide.md) 和 [`examples/lab_workflow.py`](examples/lab_workflow.py)。`examples/historical/proc.py` 不要运行。

## Issue 如何分流

维护者（或协助分拣的人）按下面贴标签。默认仓库已有 GitHub 自带的大部分标签；`needs-data-sample` 若还不存在，请在仓库 Settings → Labels 里新建（不要删已有标签）。

| 标签 | 含义 |
|------|------|
| `bug` | 确认是缺陷 |
| `enhancement` | 新功能或增强 |
| `question` | 用法问题，不一定改代码 |
| `needs-data-sample` | 还缺小样本或格式说明，暂不能复现 |
| `good first issue` | 范围小，适合学生第一次上手 |
| `wontfix` | 明确不做（超出范围、或会碰到冻结契约） |

其它已有标签（`duplicate`、`invalid`、`help wanted`）可继续用。不要批量改名或删除旧标签。

建议顺序：先看模板有没有填全 → 缺数据就标 `needs-data-sample` 并回复要什么 → 能复现再标 `bug` / `enhancement` → 修完在 CHANGELOG 记一笔并关 Issue。

## 开发安装

需要 Git 和 Python >= 3.11。

```bash
git clone https://github.com/XiaoxG/pydas.git
cd pydas
python -m pip install -U pip
pip install -e ".[dev]"
```

`[dev]` 会装 pytest，以及 `pyproject.toml` 里列出的 black / flake8 / sphinx（文档与本地格式化用）。**不要**为了开发再改依赖分组；可选 extras（例如 `[performance]`）以后再决定。

当前包名在 `pyproject.toml` 里是 `pydas`，但 PyPI 上该名字已被其它项目占用。从 Git 安装即可，不要假定 `pip install pydas` 装到的是本仓库。

## 跑测试

与 GitHub Actions（`.github/workflows/pytest.yml`）一致：

```bash
pip install -e ".[dev]"
pytest
```

CI 矩阵是 **Python 3.11 和 3.12**。`pytest.ini` 设置了 `pythonpath=src`、`testpaths=tests`，并且 **不收集** `tests/legacy`。不要把新的 `test_*.py` 放在 `tests/` 根目录，放到 `tests/unit` 或 `tests/integration`。

## 代码风格（跟 CI 对齐）

CI **只跑 pytest**，不跑 black / flake8，也不把格式检查当合并门槛。提交前请与现有代码一致，细则见 [`docs/coding-standards.md`](docs/coding-standards.md)：

- 注释、docstring、log：**英文**
- 给同事看的 README / 指南 / 示例说明：**中文**
- docstring 用 NumPy 分节（`Parameters` 下划线），不要混 Google 风格
- **公开方法名冻结**；历史参数保持驼峰（`chName`、`chOld`、`newOrder`、`cutoffull`、`plotbackend`、`updateST`）
- 新的内部函数用 snake_case；需要时可以加 snake_case 别名，但不要删旧名
- `src/` 与 `tests/` 的换行是 **CRLF**，改这些文件时保持
- 二进制 `.out` 布局冻结，不要改 `src/pydas/core/io_format.py` 里的常数

`[dev]` 里的 black / flake8 仅供本地选用；不要为了「看起来更现代」去整库重排，那会让 diff 无法审查。

## 版本号与 CHANGELOG

跟现有仓库一样：

- **SemVer**：`MAJOR.MINOR.PATCH`，Git 标签为 `vX.Y.Z`（例如 `v1.4.3`）
- **Keep a Changelog**：`CHANGELOG.md` 最新版本在最上面，常用小节为 Added / Changed / Fixed / Notes
- 一次发版必须对上这三处，避免再出现「pyproject 停在旧号、CHANGELOG 已经超前」：
  1. `pyproject.toml` 的 `version`
  2. `src/pydas/__init__.py` 的 `__version__`
  3. `CHANGELOG.md` 的 `## [X.Y.Z] - YYYY-MM-DD`

不兼容的公开行为才加 MAJOR。冻结契约（方法名、pack 布局、`cutoffull` 单位、19 列报告）的变更默认视为不兼容，需要维护者明确点头。

发版打标签、写 GitHub Release 由维护者做。不要把本包发布到 PyPI，除非维护者另作决定（见上文：名字 `pydas` 在 PyPI 上已被占用）。

## Pull Request

- 对着默认分支 `master`
- 用仓库里的 PR 模板
- 一个 PR 只做一件事；不要顺手改无关格式
- 行为有变时补测试，并更新 CHANGELOG
- 讨论请留在对应 Issue 里，方便以后检索
