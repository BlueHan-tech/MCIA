# Academic Research 配置

本机采用全局配置，供所有 Codex 项目使用。项目 AGENTS.md 保留实验边界，检索工作流只维护在全局 Skill 中；此文档记录安装与验证信息，不复制科研流程。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `~/.codex/AGENTS.md` | 最小代码修改偏好与科研问题的主动检索触发 |
| `~/.codex/skills/academic-research/SKILL.md` | 问题界定、检索、筛选、核验、综合、交付 |
| `~/.codex/config.toml` | live 网页检索、Semantic Scholar MCP、Skill 启用 |
| `~/.codex/tools/semantic-scholar/` | 独立 MCP Python 环境，不写入 MCIA 的 conda 环境 |

`~` 在本机为 `C:/Users/zyh`。Skill 使用显式 `skills.config` 注册；不在项目里重复安装同名 Skill。`CODEX_HOME`、启动参数、项目级配置或更近层级的指令可影响实际生效范围。

## 工具选择

使用社区维护的 [Semantic Scholar MCP](https://github.com/smaniches/semantic-scholar-mcp)，[PyPI 包 s2-mcp-server 1.7.4](https://pypi.org/project/s2-mcp-server/1.7.4/)；它是连接 Semantic Scholar API 的第三方实现，不是 Semantic Scholar 官方 MCP。仅启用六个检索、详情、标题匹配、批量搜索、引用导出和状态工具。

API key 通过 `SEMANTIC_SCHOLAR_API_KEY` 环境变量传入，不写入项目或 TOML。无 key 时尝试公共访问，可能限流；失败时 Skill 使用可用网页检索查找原始论文，不无限重试。是否具备全文以实际获取结果为准。

## 生效与验证

重新启动 Codex CLI / 重载 MCP 后，用 `/mcp` 查看连接、用 `/skills` 查找 `academic-research`；也可显式调用 `$academic-research`。`codex mcp list` 只证明配置被读取，不能替代握手和真实搜索验证。

2026-09-08 验证记录：

- Skill 通过 `quick_validate.py`；在真实用户环境调用 Codex app-server 的 `skills/list`，从 MCIA 与用户目录均发现启用的 user-scope `academic-research`。
- `config/read` 确认 `web_search = "live"` 和新增 MCP 配置生效；原有无关配置保留。修改前备份为 `~/.codex/config.toml.before-academic-research-20260908-110054.bak`。
- MCP stdio 握手、版本 1.7.4 与工具枚举通过，六个配置工具均存在。以 `Ninapro` 实际搜索时公共接口返回 HTTP 429；未取得该次搜索结果，也未完成其后续论文详情调用。无 key 搜索的稳定性尚未验证通过。
- 当前会话网页备用路径已实测：搜索 `Ninapro database electromyography Scientific Data Atzori 2014`，在出版社入口无法打开后，成功打开 [PMC 收录的原始论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4421935/)，核对标题与 DOI `10.1038/sdata.2014.53`。这验证了替代来源获取能力，不代表已做完整文献综述。
- 已检查 UTF-8、文档链接和差异格式；未运行 MCIA 训练或绘图，未验证未来每个任务的自动调用行为。安装脚本与验证记录位于本机忽略目录 `outputs/codex-research-setup/`，不是主实验 run 或规范来源。

新 Skill 和 MCP 不一定即时注入已经运行的任务；自动选择属于指令行为，不是每个问题都联网的硬性程序钩子。重启 CLI / 重载 MCP 后检查实际状态。若公共接口持续限流，可自行申请 Semantic Scholar API key，在启动 Codex 前设置环境变量；不需要把密钥发到对话中。

## 配置依据

- [OpenAI：全局 AGENTS.md 与项目覆盖顺序](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [OpenAI：MCP 配置](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)
- [OpenAI：Skill 构建与自动调用](https://learn.chatgpt.com/docs/build-skills)
- [OpenAI：配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)
