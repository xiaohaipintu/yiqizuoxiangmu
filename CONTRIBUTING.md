# 协作规范（双人同步）

仓库：`xiaohaipintu/yiqizuoxiangmu`　默认分支：**master**

本项目由两人共同开发，任何改动都必须先确认对方的进度，避免代码互相覆盖。

## 一、写代码之前

```bash
git fetch origin
git log --oneline HEAD..origin/master
```

- **有输出** = 对方有新提交 → 先同步，再动代码：

```bash
git pull --rebase
```

- **没输出** = 你是最新 → 可以直接写。

## 二、提交与推送

```bash
git fetch origin && git log --oneline HEAD..origin/master   # 先查
git add -A
git commit -m "feat: 描述这次改了什么"
git fetch origin && git pull --rebase                       # push 前再查一次
git push
```

- `git push` 被拒（对方刚推过）→ `git pull --rebase` 后重推。
- **禁止 `git push --force`**，会把对方提交冲掉。
- 一个逻辑改动一个 commit，信息用前缀：`feat:` / `fix:` / `docs:` / `refactor:` / `config:` / `deploy:`。

## 三、冲突处理

```bash
git pull --rebase          # 出现冲突
# 手动改文件，删掉 <<<<<<< ======= >>>>>>> 标记
git add <冲突文件>
git rebase --continue
```

不确定怎么合并时先开分支 `feat/xxx` 推上去，让对方看 PR，别直接改 master。

## 四、上线

每次推送后都要部署上线，并确认线上效果。

**本项目用 GitHub Pages 自动部署：**

- 线上地址：<https://xiaohaipintu.github.io/yiqizuoxiangmu/>
- 部署规则：`master` 分支根目录（`source: master`, `path: /`），**`git push` 到 master 后自动构建上线**，无需手动操作
- 构建进度：<https://github.com/xiaohaipintu/yiqizuoxiangmu/deployments>
- 上线后延迟 1～2 分钟生效，浏览器 `Ctrl+F5` 强刷看最新版

由于 Pages 直接发布仓库里的静态文件，**任何能被浏览器打开的东西（含源码、注释、接口地址）都是公开的**。密钥、密码、私有 API Key 一律不要写进仓库 —— 用环境变量或单独的私有仓库。
