---
name: sync-check
description: 检查远端仓库是否有对方的新提交，落后则拉取同步（写代码前/提交前执行）
---
# 同步检查（拼图项目双人协作）

仓库：`xiaohaipintu/yiqizuoxiangmu`，默认分支 **master**（不是 main）。另一个人是 yumin / CelineLuna。

## 步骤

1. **拉取远端状态**

```bash
git fetch origin
git status -sb
git log --oneline HEAD..origin/master
```

2. **判断**

- `git log HEAD..origin/master` 有输出 → 对方有新提交，说明本地落后。
- 无输出 → 本地是最新的。

3. **落后就同步**（写代码前 / 提交前都要做）

```bash
git pull --rebase
```

冲突：逐个解决后 `git add <file>` → `git rebase --continue`。**绝不 `git push --force`**。

4. **回头看本地有没有未提交的改动**

```bash
git status --porcelain
```

5. **汇报**（一两句，不啰嗦）

- 对方最新提交：`<短 sha> <作者> <提交信息> <时间>`
- 本地领先 / 落后几个提交
- 已执行的动作（pull / rebase / 无操作）
- 是否可以开始写代码

## 铁律

- 永远先 `git fetch` 再判断，不要凭 `git log` 本地缓存下结论。
- push 前再 fetch 一次；被拒就 rebase 重推，不用 force。
- 每次推送后按项目要求要部署上线（部署方式见项目记忆 `dual-collab-sync-workflow`）。
