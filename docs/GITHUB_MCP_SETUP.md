# GitHub MCP：源码只读配置

已在 `C:/Users/zyh/.codex/config.toml` 添加全局 `mcp_servers.github`，使用 GitHub 官方远程入口 `https://api.githubcopilot.com/mcp/x/repos/readonly`。

仅向 Codex 暴露七个工具：`search_repositories`、`search_code`、`get_file_contents`、`list_branches`、`list_tags`、`list_commits`、`get_commit`。服务端限制为 repos 只读，客户端再用 `enabled_tools` 限定范围；不启用 issues、pull_requests 或写入工具。此工具限制不会改变 token 本身的 GitHub 权限。

## 认证

当前未检测到 `GITHUB_PAT_TOKEN`。在 GitHub 创建适合目标仓库的最小权限 PAT，源码访问只授予所需读取权限，不授予写入权限。不要把 token 发到对话或写入仓库。

PowerShell 中安全输入，并在同一终端启动 CLI：

```powershell
$githubPat = Read-Host 'GitHub PAT' -AsSecureString
$env:GITHUB_PAT_TOKEN = [System.Net.NetworkCredential]::new('', $githubPat).Password
codex
```

若用于桌面应用，可在 Windows 用户环境变量中配置 `GITHUB_PAT_TOKEN`，然后完全退出并重新启动应用，使其继承变量。仅在终端设置进程变量不会更新已运行的桌面应用。

进入 Codex 后用 `/mcp` 检查，再请求读取一个公开仓库的 README 做短时验证。Codex 不会仅因项目存在 `.env` 就自动加载其中的 token。

## 本次验证

2026-09-08：TOML 解析、无关配置保留和 `codex mcp get github` 读取通过。已备份原配置。由于缺少 token，尚未验证认证握手、远程工具枚举或仓库读取。未修改实验代码或环境，未运行训练或绘图。

配置依据：[GitHub 官方 Codex 安装说明](https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-codex.md)、[只读 toolset 入口](https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md)、[OpenAI MCP 配置](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。
