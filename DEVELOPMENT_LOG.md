# 项目开发记录

> 本文档记录了 Taylor Swift Lyrics RAG 系统的开发过程和重要决策。

---

## 2026-04-29 开发会话

### 1. 评估指标体系增强 (evaluate.py)

#### 新增指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **检索指标** | `MRR` | Mean Reciprocal Rank，倒数排名均值 |
| | `Hit Rate@1/3/5` | Top-K 内命中比例 |
| **生成质量** | `Answer Precision` | 回答关键词匹配精确率 |
| | `Answer Recall` | 关键词召回率 |
| | `Answer F1 Score` | F1 调和平均 |
| | `Song Precision` | 检索歌曲匹配精确率 |
| **性能指标** | `Avg/Min/Max Time` | 响应时间统计 |

#### 代码位置
- `code/evaluate.py` - 完整的评估脚本

---

### 2. 测试数据集修复 (test_dataset.json)

#### 修复的问题

| ID | 问题描述 | 修复内容 |
|----|----------|----------|
| 6 | 歌词不存在 | "And if you ever think of me, you know now" → "I hope you think of me" |
| 10 | 歌曲错误 | "I hit the ground running instead of coming undone" → "I hit the ground running" |

#### 数据集结构
```json
{
    "id": 1,
    "type": "lyrics_retrieval | emotion_analysis",
    "question": "英文问题",
    "expected_keywords": ["关键词1", "关键词2"]
}
```

---

### 3. Git 仓库初始化

#### 创建的文件
- `.gitignore` - 忽略索引文件、API密钥等
- `.env.example` - 环境变量模板

#### 已排除的文件
- `index/` - FAISS 索引（可重新生成）
- `.env` - API 密钥

#### 待执行的命令
```bash
# 1. 设置 Git 用户
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# 2. 提交代码
git -C d:/AIGC_PROJECT/sepa/separate_parts commit -m "Initial commit: Taylor Swift Lyrics RAG System"

# 3. 创建 GitHub 仓库并推送
git remote add origin https://github.com/你的用户名/separate_parts.git
git -C d:/AIGC_PROJECT/sepa/separate_parts push -u origin master
```

---

### 4. 待修复问题

#### 测试数据集 (data/test_dataset.json)
- [ ] 问题5：Anti-Hero 的 expected_keywords 中的 "blue eyes" 实际在 Tim McGraw 里
  - 建议改为：["Midnights", "self-hatred", "anti-hero"]

---

## 项目结构

```
separate_parts/
├── code/
│   ├── main.py              # RAG 主程序
│   ├── evaluate.py          # 批量测试脚本 (含评估指标)
│   ├── rag/                 # RAG 核心模块
│   │   ├── config.py        # 配置管理
│   │   ├── logger.py        # 日志系统
│   │   ├── components/      # 组件
│   │   │   ├── data_load.py
│   │   │   └── generate_answer.py
│   │   └── retrieval/        # 检索模块
│   │       ├── retrieval_search.py
│   │       ├── query_rewrite.py
│   │       ├── context_compressor.py
│   │       ├── reranker.py
│   │       └── pipeline.py
│   └── scripts/             # 脚本
│       ├── generate_lyrics_index.py
│       └── generate_track_index.py
├── data/
│   ├── test_dataset.json    # 测试数据集
│   └── Taylor_Swift_Genius/  # 歌词数据
├── index/                   # FAISS 索引 (不上传)
├── .gitignore
├── .env.example
└── README
```

---

## 运行指令

### 启动 RAG 系统
```bash
cd d:/AIGC_PROJECT/sepa/separate_parts/code
python main.py
```

### 运行批量测试
```bash
cd d:/AIGC_PROJECT/sepa/separate_parts/code
python evaluate.py
```

### 重新生成索引
```bash
cd d:/AIGC_PROJECT/sepa/separate_parts/code
python -m scripts.generate_lyrics_index
python -m scripts.generate_track_index
```
