# CS336 作业1（基础篇）：从零搭建Transformer语言模型
版本：26.0.3
授课团队：斯坦福CS336教研组
学期：2026年春季

## 1 作业概述
本次作业要求你从零搭建训练标准Transformer语言模型（LM）所需的全部组件，并完成模型训练实验。

### 你需要亲手实现的模块
1. 字节对编码（BPE）分词器（第2章节）
2. Transformer语言模型主体（第3章节）
3. 交叉熵损失函数、AdamW优化器（第4章节）
4. 训练循环，支持模型与优化器状态的保存、加载（第5章节）

### 你需要运行完成的实验流程
1. 在TinyStories数据集上训练BPE分词器
2. 使用训练好的分词器对数据集编码，将文本转换为整数Token ID序列
3. 在TinyStories数据集上训练Transformer语言模型
4. 基于训练完成的模型进行文本采样生成，并计算困惑度（Perplexity）完成评测
5. 在OpenWebText数据集上训练模型，将最终困惑度提交至课程排行榜

### 可用工具与代码约束
本次作业要求所有组件**从零手写实现**，有严格的PyTorch接口使用限制：
禁止直接调用 `torch.nn`、`torch.nn.functional`、`torch.optim` 内绝大多数预定义模块，仅允许使用以下内容：
- `torch.nn.Parameter`
- `torch.nn` 容器类（Module、ModuleList、Sequential等）
- `torch.optim.Optimizer` 优化器基类

其余PyTorch基础API均可自由使用。若不确定某个函数/类是否允许调用，可以在课程Slack群组提问；拿不准时，以贴合“从零实现”作业初衷为判断标准。

### AI工具使用规范
AI工具可以独立完成本次作业大部分代码编写，但过度依赖AI会导致你无法深入理解课程知识。
允许使用AI场景：
- 解答高层次概念问题
- 查询底层编程文档、函数签名、库API说明

严禁使用AI场景：
- 编写作业任意模块代码（包括Cursor智能代理、Claude Code、GitHub Copilot等代码补全工具）
使用AI对话工具时需要留存提示词记录；调用官方提供的AGENTS.md规则文件时必须遵循文件要求。

强烈建议在代码编辑器中关闭AI自动补全（Cursor Tab、Copilot等），仅保留基础语法自动补全即可；往届学生反馈，关闭AI补全能大幅加深对底层原理的理解。完整AI使用细则请查阅课程官方文档。

### 作业代码仓库
作业源码与本文档全部内容托管在GitHub：
`github.com/stanford-cs336/assignment1-basics`
执行 `git clone` 拉取仓库；后续代码更新教研组会通知，通过 `git pull` 获取最新版本。

仓库目录结构说明：
1. `cs336_basics/*`：你的自定义代码编写目录，初始无任何代码，所有逻辑完全从零实现
2. `adapters.py`：适配层胶水代码，规定了所有模块对外统一调用接口。你只需在对应函数内调用自己写的代码，**不可在此文件编写核心业务逻辑**
3. `test_*.py`：单元测试文件，所有测试用例必须全部通过；禁止修改测试源码，测试会调用adapters中定义的钩子函数校验你的实现正确性

### 作业提交方式
1. 运行脚本 `make_submission.sh` 打包提交压缩包；若数据集、模型权重等大文件无需提交，可在脚本排除列表中添加路径
2. 前往Gradescope平台上传两份文件：
   - `writeup.pdf`：书面问答题作答文档，要求排版工整
   - `code.zip`：完整源码压缩包

如需将模型困惑度提交至课程排行榜，需向仓库 `github.com/stanford-cs336/assignment1-basics-leaderboard` 提交Pull Request，排行榜仓库README内有详细提交流程。

### 数据集获取
本次作业使用两份预处理文本数据集：
1. TinyStories（R. Eldan 等人，2023）
2. OpenWebText（A. Gokaslan 等人，2019）
两份数据集均为超大纯文本单文件。
- 在校选课学生：计算集群使用指南内有数据集下载教程
- 自学线上学习者：仓库README.md内提供下载命令

### 低配设备适配小贴士（Low-Resource Tip）
作业各章节都会提供无GPU/少量GPU资源下的调试建议：比如缩小数据集规模、调小模型尺寸、在Mac集成显卡/CPU上运行训练代码。所有适配提示均以蓝色框标注。
即便你可以使用斯坦福校内计算服务器，这些技巧也能帮你快速迭代调试、节省耗时，推荐通读。

#### 小贴士：Apple芯片/CPU运行作业1
使用官方参考实现代码时，搭载36GB内存的Apple M4 Max芯片：
- Metal GPU（MPS后端）：5分钟内即可训练出可生成流畅文本的语言模型
- CPU纯运算：约30分钟
只要你的笔记本配置不算老旧、代码实现高效正确，就可以训练小型语言模型，生成通顺的儿童短篇故事。后续章节会详细讲解CPU/MPS后端适配修改方案。

---

## 2 字节对编码（BPE）分词器
作业第一部分需要实现并训练**字节级BPE分词器**（R. Sennrich 2016；C. Wang 2019）：
将任意Unicode字符串转为字节序列，基于字节序列训练BPE；训练完成后，用该分词器把文本字符串编码为整数Token序列，供给语言模型训练使用。

### 2.1 Unicode标准基础
Unicode是字符编码规范，将每一个字符映射为唯一整数码点。截至2025年9月发布的Unicode 17.0，一共收录172种书写体系、总计159801个字符。
示例：
- 英文字母`s`：码点115，标准写法 `U+0073`
- 汉字`牛`：码点29275

Python内置函数：
- `ord(字符)`：将单个Unicode字符转为对应整数码点
- `chr(整数)`：将Unicode码点转回字符
```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

#### 习题（unicode1）：理解Unicode（1分）
(a) `chr(0)` 会返回什么Unicode字符？
提交要求：一句话作答
(b) 该字符的字符串内部表示（`__repr__()`）和控制台打印效果有何区别？
提交要求：一句话作答
(c) 该空字符出现在文本中会产生什么现象？可在Python交互环境运行下方代码验证：
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
提交要求：一句话作答

### 2.2 Unicode编码格式
Unicode仅定义字符与码点的映射，无法直接存储传输；直接基于十余万码点训练分词器词汇量过大、字符分布稀疏，完全不可行。因此需要借助Unicode编码格式，将字符转为字节流。
Unicode三大主流编码：UTF-8、UTF-16、UTF-32；**UTF-8是互联网通用编码（全网98%以上网页使用）**。

Python编码/解码常用接口：
- `字符串.encode("utf-8")`：Unicode文本编码为UTF-8字节对象
- `list(字节对象)`：提取每一字节对应的0~255整数
- `字节对象.decode("utf-8")`：字节流还原为Unicode字符串

示例：
```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> len(test_string)
13
>>> len(utf8_encoded)
23
>>> utf8_encoded.decode("utf-8")
hello! こんにちは!
```
核心结论：**一个Unicode字符不一定对应单个字节**。
将码点转为0~255字节序列后，基础词汇量固定为256个，极易管理；字节级分词不存在未登录词（OOV）问题：任意文本都可以拆解为0~255的字节整数序列。

#### 习题（unicode2）：Unicode编码辨析（3分）
(a) 相较于UTF-16、UTF-32，选择基于UTF-8字节训练分词器的优势有哪些？可对比不同文本在三种编码下的字节输出差异。
提交要求：1~2句话作答
(b) 下方函数意图将UTF-8字节串解码为Unicode字符串，但实现存在错误，请说明错误原因，并给出一段会输出错误结果的输入字节串。
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```
提交要求：给出错误输入示例+一句话原因解释
(c) 举出一段无法解码为合法Unicode字符的双字节序列，并一句话说明缘由。
提交要求：示例+一句话解释

### 2.3 子词分词原理
基于完整单词分词：词汇量巨大、生僻词存在OOV问题；
基于纯字节分词：序列长度过长，训练计算开销大、长距离依赖难以学习；
**子词分词是二者折中方案**。

字节级BPE初始词汇固定256个（全部基础字节），通过不断合并文本中出现频次最高的相邻字节对，生成新子词扩充词汇表：高频连续字节组合（如英文`the`）会被合并为单个Token，大幅压缩序列长度；同时依托基础字节兜底，永远不会出现未登录词。这套迭代合并字节对的压缩算法就是**字节对编码（BPE）**。

### 2.4 BPE分词器完整训练流程
BPE训练分为三大核心步骤：词汇初始化、预分词、迭代计算字节合并规则，额外需要处理特殊占位Token。

#### 1. 词汇初始化
分词器词汇表是「字节序列 ↔ 整数ID」双向映射；字节级BPE初始词汇就是全部256个一字节字节，初始词汇大小=256。

#### 2. 预分词（Pre-tokenization）
若直接遍历全文本统计所有相邻字节对频次，计算量极大；同时标点附着单词会拆分出大量语义相近但Token ID完全不同的单元（如`dog`、`dog!`）。
预分词会按照固定规则把文本粗切分为基础单元，仅在单元内部统计字节对频次、执行合并，单元边界禁止合并：
- 原版BPE：仅按空格切分单词（SentencePiece、LLaMA1/2分词器沿用该逻辑）
- GPT-2起现代主流分词器：基于正则表达式预分词
本次作业采用OpenAI tiktoken优化后的正则规则：
```regex
r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```
Python调用示例（需要安装`regex`库）：
```python
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re.findall(PAT, "some text that i'll pre-tokenize")
# 输出：['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```
代码中推荐使用 `re.finditer` 迭代遍历，避免一次性存储全部预分词结果占用内存。

#### 3. 计算BPE合并规则
将所有预分词单元转为UTF-8字节序列后，循环执行：
1. 统计所有相邻字节对出现总频次
2. 选取频次最高字节对；若多对频次一致，**字典序更大的字节对优先合并**
3. 将文本中该字节对全部替换为新Token，新增至词汇表
4. 重复迭代，直至词汇量达到预设上限

约束：**禁止跨预分词单元边界合并字节**。

#### 4. 特殊Token处理
`<|endoftext|>` 这类标记用于分隔文档边界、标识文本起止，必须整体作为单个Token，永远不可被拆分；需要提前加入词汇表并分配固定ID，训练时文档边界会被特殊Token切割，边界两侧字节绝不允许合并。

#### BPE训练极简示例
训练语料：
```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```
特殊Token：`<|endoftext|>`
初始词汇：256基础字节 + 结束符Token
预分词按空格切分，频次统计：`{low:5, lower:2, widest:3, newest:6}`
第一轮最高频字节对：`es`、`st`频次并列，字典序`st`更大优先合并；持续迭代多轮合并后，单词`newest`最终会被编码为 `[ne, west]`。

### 2.5 BPE分词器训练实验
需要在TinyStories完整数据集上训练字节级BPE分词器；预分词是性能瓶颈，推荐使用Python `multiprocessing` 多进程并行加速：
将语料分块时，块边界必须落在特殊Token两端，保证文档不会被拆分截断；官方仓库提供了分块参考代码可直接复用。

额外训练规范：
1. 训练前先用特殊Token切割全文本，分段独立预分词，文档边界不参与合并统计
2. 朴素逐轮全局统计字节对频次速度极慢，优化方案：缓存所有字节对频次，每轮合并后仅更新受影响相邻对的计数，大幅提速；合并步骤无法并行

#### 低配调试小贴士
不要直接在完整数据集调试：先用验证集（2.2万文档，远小于全集212万文档）训练小分词器排查Bug；调试集规模要兼顾：能复现全集性能瓶颈、运行耗时可控。

#### 习题（train_bpe）：实现BPE训练函数（15分）
编写训练函数，输入文本文件路径，输出词汇表与合并规则列表。
##### 入参要求
- `input_path: str`：训练文本文件路径
- `vocab_size: int`：最终词汇总量上限（包含初始256字节、合并生成子词、所有特殊Token）
- `special_tokens: list[str]`：自定义特殊Token列表；训练中作为硬性分割边界，不计入合并频次统计

##### 出参要求
- `vocab: dict[int, bytes]`：词汇表，Token ID → 字节序列映射
- `merges: list[tuple[bytes, bytes]]`：合并规则列表，按训练生成顺序存储每一组被合并的字节对

##### 测试方式
实现adapters层适配函数 `run_train_bpe`，执行命令：
`uv run pytest tests/test_train_bpe.py`
所有测试用例必须通过；进阶可选：用C++/Rust编写核心训练逻辑提升速度。

#### 习题（train_bpe_tinystories）：TinyStories数据集BPE训练（2分）
(a) 在TinyStories上训练词汇量上限10000的字节级BPE分词器，添加`<|endoftext|>`结束Token；将词汇表、合并规则序列化保存至本地。回答：训练耗时、内存占用是多少？词汇表中最长Token是什么？该结果是否符合预期？
资源限制：无GPU耗时≤30分钟，内存≤30GB；多进程优化后可压缩至2分钟内。
提交要求：1~2句话作答
(b) 对代码做性能剖析，分词器训练哪一步耗时占比最高？
提交要求：1~2句话作答

#### 习题（train_bpe_expts_owt）：OpenWebText数据集BPE训练（2分）
(a) 在OpenWebText上训练词汇量上限32000的BPE分词器，保存结果。词汇表最长Token是什么？是否合理？
资源限制：无GPU耗时≤12小时，内存≤100GB
提交要求：1~2句话作答
(b) 对比TinyStories、OpenWebText两套分词器的差异与特点。
提交要求：1~2句话作答

### 2.6 分词器编码与解码逻辑
训练完成词汇表、合并规则后，需要实现完整Tokenizer类：加载权重，完成「文本→Token ID序列（编码）」「Token ID序列→原始文本（解码）」双向转换。

#### 2.6.1 编码流程
1. 正则预分词，切分基础单元，每个单元转为UTF-8字节列表
2. 严格按照训练时合并顺序，在单元内部迭代应用所有合并规则，单元边界不合并
3. 查找词汇表映射，将最终子词转为整数ID

示例：
输入文本 `the cat ate`，预分词结果：`['the', ' cat', ' ate']`
依次应用合并规则后：
- `the` → 单个Token ID
- ` cat` → 两个Token ID
- ` ate` → 两个Token ID
最终输出整数序列：`[9, 7, 1, 5, 10, 3]`

特殊规则：
自定义特殊Token需要优先匹配，整体保留为单个Token；
超大文件逐行流式编码时，必须分块处理且Token不能跨越分块边界，保证内存占用恒定。

#### 2.6.2 解码流程
1. 根据ID查表获取对应字节序列，全部字节拼接为完整字节串
2. 字节串解码为Unicode字符串；若字节序列不合法、无法解码，统一替换为Unicode替换字符 `U+FFFD`（Python `bytes.decode(errors='replace')` 原生支持该逻辑）

#### 习题（tokenizer）：实现Tokenizer分词器类（15分）
基于词汇表、合并规则实现完整分词器类，支持自定义特殊Token，推荐接口规范：
```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """初始化分词器"""
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """从本地序列化文件加载分词器权重"""
    def encode(self, text: str) -> list[int]:
        """整段文本编码为ID列表"""
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """迭代器流式编码大文件，惰性产出ID，内存友好"""
    def decode(self, ids: list[int]) -> str:
        """ID序列解码回文本"""
```
实现adapters适配函数 `get_tokenizer`，执行 `uv run pytest tests/test_tokenizer.py` 通过全部测试。

### 2.7 分词器对比实验
#### 习题（tokenizer_experiments）：分词器消融实验（4分）
(a) 分别从TinyStories、OpenWebText抽取10篇文档，使用各自对应分词器编码；计算两套分词器的压缩率（字节数/Token数）。
提交要求：1~2句话作答
(b) 用TinyStories训练的10k词汇分词器编码OpenWebText样本，压缩率会如何变化？定性描述差异。
提交要求：1~2句话作答
(c) 估算你的分词器吞吐速度（字节/秒），计算完整编码825GB The Pile数据集需要多久。
提交要求：1~2句话作答
(d) 将两套数据集训练集、验证集全部编码为Token ID，存储为`uint16`类型NumPy数组；说明选用uint16的合理性。
提交要求：1~2句话作答

---

## 3 Transformer语言模型架构
语言模型输入：批量整数Token ID张量，形状 `(batch_size, 序列长度)`
模型输出：词汇表维度归一化概率分布张量，形状 `(batch_size, 序列长度, 词汇量)`，每一位预测当前位置之后下一个Token的概率。
训练阶段：通过下一词预测计算交叉熵损失；推理生成阶段：取序列最后一位概率分布采样下一个Token，循环拼接直至生成结束符。
本章需要从零搭建**解码器Only、Pre-Norm架构的Transformer语言模型**。

### 3.1 整体架构总览
完整流水线：
Token嵌入层 → N层Transformer模块堆叠 → 最终层归一化 → LM输出头（线性层）→ Softmax概率分布

单层Pre-Norm Transformer模块内部结构：
输入残差流 → RMSNorm归一化 → 带RoPE位置编码的因果多头自注意力 → 残差相加
→ RMSNorm归一化 → SwiGLU前馈网络 → 残差相加

### 3.2 工程技巧：批量运算、Einsum张量运算
Transformer全程大量批量并行计算：批次维度、序列长度维度、多头注意力维度均可视为批量维度。
PyTorch原生矩阵乘法可读性差、张量维度变换繁琐；推荐使用：
1. `torch.einsum`：爱因斯坦求和记号，直观描述任意张量缩并运算
2. `einops`：张量维度重组、变形工具，代码可读性极高
课程强烈要求掌握Einsum写法，后文示例会提供参考。

补充：数学公式多采用列向量写法，但PyTorch/NumPy默认行主序内存布局，矩阵乘法需要转置适配；使用Einsum可规避该差异问题。

### 3.3 基础模块：线性层、嵌入层实现
#### 参数初始化规范
- 线性层权重：截断正态分布，均值0，方差 $\displaystyle \frac{2}{d_{in}+d_{out}}$，截断区间 $[-3\sigma,3\sigma]$
- 嵌入层权重：截断正态分布，均值0，方差1，截断区间 $[-3,3]$
- RMSNorm增益参数：初始化为全1张量
调用 `torch.nn.init.trunc_normal_` 完成初始化。

#### 习题（linear）：自定义无偏置线性层（1分）
继承 `torch.nn.Module` 实现Linear类，无偏置项，接口对齐PyTorch原生nn.Linear：
```python
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```
约束：禁止调用nn.Linear、F.linear；权重存入nn.Parameter托管；按上述规则初始化。
测试：实现adapters适配函数，执行 `uv run pytest -k test_linear`

#### 习题（embedding）：自定义嵌入层（1分）
手写Embedding层，禁止调用nn.Embedding：根据Token ID索引嵌入矩阵，输出形状 `(batch, seq_len, d_model)`，遵循初始化规范。
测试：适配adapters后执行 `uv run pytest -k test_embedding`

### 3.4 Pre-Norm Transformer模块各子组件实现
#### 3.4.1 RMSNorm均方根层归一化
摒弃原版LayerNorm，使用LLaMA系列采用的RMSNorm，计算公式：
$$\text{RMSNorm}(a_i) = g_i \cdot \frac{a_i}{\sqrt{\frac{1}{d_{model}}\sum_{k=1}^{d_{model}}a_k^2 + \varepsilon}}$$
$g$：可学习增益向量；$\varepsilon=1\mathrm{e}{-5}$ 数值稳定项。
实现要点：输入张量先转float32计算防止平方溢出，计算完成转回原始精度。

#### 习题（rmsnorm）：实现RMSNorm（1分）
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入shape (B, L, d_model)，输出同shape
        pass
```
测试：`uv run pytest -k test_rmsnorm`

#### 3.4.2 SwiGLU门控前馈网络
现代LLM主流前馈结构，替代原始ReLU双层前馈：
由SiLU激活+GLU门控结构组成，公式：
$$\text{FFN}(x) = W_2\big( \text{SiLU}(W_1 x) \odot W_3 x \big)$$
- 内部维度 $d_{ff}=\frac{8}{3}d_{model}$，向上取整至64的倍数适配硬件加速
- 所有线性层无偏置项

SiLU(Swish)：$\text{SiLU}(x)=x\cdot\sigma(x)$

#### 习题（positionwise_feedforward）：实现SwiGLU前馈层（2分）
测试命令：`uv run pytest -k test_swiglu`

#### 3.4.3 RoPE旋转位置编码
绝对位置编码弊端明显，RoPE通过旋转矩阵将位置信息融入Query、Key向量，天然适配相对位置建模，LLaMA全系采用。
实现要点：
1. 预计算所有位置对应的正弦、余弦值，存入Buffer（非可学习参数）
2. 仅对Query、Key施加旋转，Value向量不做位置变换
3. 支持任意前置批量维度

#### 习题（rope）：实现RoPE模块（2分）
```python
class RotaryPositionalEmbedding:
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        pass
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (*, seq_len, d_k)
        pass
```
测试：`uv run pytest -k test_rope`

#### 3.4.4 缩放点积注意力 & 手写Softmax
##### 手写稳定版Softmax
为防止指数运算溢出，计算前逐维度减去该维度最大值：
$$\text{softmax}(x_i)=\frac{e^{x_i-\max(x)}}{\sum_j e^{x_j-\max(x)}}$$

#### 习题（softmax）：手写Softmax（1分）
函数入参：张量+归一化维度dim，输出同shape概率张量。
测试：`uv run pytest -k test_softmax_matches_pytorch`

##### 缩放点积注意力公式
$$\text{Attention}(Q,K,V)=\text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} + \text{掩码} \right)V$$
因果掩码：右上角Future位置填充负无穷，保证无法看到未来Token。

#### 习题（scaled_dot_product_attention）：实现缩放点积注意力（5分）
支持任意前置批量维度、布尔注意力掩码，兼容3维、4维张量输入。
测试命令：
`uv run pytest -k test_scaled_dot_product_attention`
`uv run pytest -k test_4d_scaled_dot_product_attention`

#### 3.4.5 因果多头自注意力
将QKV向量切分为多头独立计算注意力，结果拼接后经过输出投影层；整体流程：
1. 输入经过三个线性层得到Q/K/V
2. 分头，每头维度 $d_k=d_{model}/\text{头数}$
3. Q、K施加RoPE位置编码
4. 缩放点积注意力+因果掩码
5. 多头结果拼接，输出线性层融合

#### 习题（multihead_self_attention）：因果多头自注意力（5分）
继承nn.Module，入参：d_model、num_heads；严格均分每头维度。
测试：`uv run pytest -k test_multihead_self_attention`

### 3.5 完整Pre-Norm Transformer模块 & 整体语言模型
单层Transformer模块计算公式：
$$
\begin{align*}
z &= x + \text{MHA}(\text{RMSNorm}(x)) \\
y &= z + \text{SwiGLU}(\text{RMSNorm}(z))
\end{align*}
$$

#### 习题（transformer_block）：搭建单层Transformer块（3分）
入参：d_model、num_heads、d_ff
测试：`uv run pytest -k test_transformer_block`

#### 习题（transformer_lm）：组装完整Transformer语言模型（3分）
入参：词汇量、上下文长度、层数、d_model、头数、d_ff
整体流程：嵌入→N层Transformer→最终RMSNorm→LM线性头输出Logits
测试：`uv run pytest -k test_transformer_lm`

### 3.6 模型算力与参数量统计（资源核算）
矩阵乘法算力公式：矩阵 $A(m\times n) \cdot B(n\times p)$ 总浮点运算量 $\text{FLOPs}=2mnp$

#### 习题（transformer_accounting）：Transformer资源开销计算（5分）
给定GPT-2 XL配置：
词汇量50257、上下文长度1024、层数48、d_model=1600、头数25、d_ff=4288
(a) 计算总可训练参数量；FP32精度存储模型权重需要多少内存？
(b) 罗列前向传播所有矩阵乘法，计算总FLOPs
(c) 模型哪一部分算力消耗最高？
(d) 依次计算GPT2 Small/Medium/Large三套尺寸的算力占比；分析模型扩容时各模块算力占比变化趋势
(e) 将上下文长度提升至16384，总算力与各模块占比如何改变？

---

## 4 模型训练核心组件：损失、优化器、学习率调度
### 4.1 交叉熵损失 & 困惑度Perplexity
语言模型损失：负对数似然交叉熵，基于Logits计算，复用Softmax数值稳定技巧。
困惑度公式（序列平均损失指数）：
$$\text{PPL}= \exp\left( \frac{1}{m}\sum_{i=1}^m \ell_i \right)$$

#### 习题（cross_entropy）：手写交叉熵损失（1分）
入参：批量Logits张量、目标Token ID；返回批次平均损失。
测试：`uv run pytest -k test_cross_entropy`

### 4.2 SGD随机梯度下降（基础优化器铺垫）
PyTorch自定义优化器需要继承 `torch.optim.Optimizer`，必须实现`__init__()`与`step()`两个核心方法；通过`self.state`字典存储每个参数迭代状态。

#### 习题（learning_rate_tuning）：学习率调试对照实验（1分）
以衰减版SGD玩具代码为例，分别设置学习率10、100、1000迭代10轮，观察损失收敛/震荡/发散现象，对比差异。

### 4.3 AdamW权重衰减优化器
现代LLM标准优化器，解耦权重衰减与梯度更新，算法伪代码见作业原文；需要维护每个参数一阶动量m、二阶动量v。

#### 习题（adamw）：实现AdamW优化器（2分）
可配置超参：学习率α、β1、β2、权重衰减λ、ε；适配PyTorch优化器规范。
测试：`uv run pytest -k test_adamw`

#### 习题（adamw_accounting）：AdamW训练内存算力开销（2分）
(a) 拆分FP32训练峰值内存：参数、激活值、梯度、优化器动量状态，写出代数表达式
(b) 代入GPT2 XL参数，推导仅和batch_size相关的内存公式；计算80GB显存可容纳最大批次大小
(c) 单次优化器step总FLOPs计算公式
(d) H100 GPU FP32峰值算力495 TFLOPS，MFU=50%；前向FLOPs:反向FLOPs=1:2；计算GPT2 XL训练40万步、批次1024所需总时长（小时）

### 4.4 带预热的余弦退火学习率调度
LLM通用学习率策略：预热上升 → 余弦衰减 → 固定最小学习率。
分段公式：
1. 预热阶段 $t<T_w$：$\displaystyle \alpha_t = \frac{t}{T_w}\alpha_{\text{max}}$
2. 余弦退火 $T_w\le t\le T_c$：
$$\alpha_t=\alpha_{\text{min}}+\frac{1}{2}\big(1+\cos(\pi\cdot\frac{t-T_w}{T_c-T_w})\big)(\alpha_{\text{max}}-\alpha_{\text{min}})$$
3. 退火结束 $t>T_c$：$\alpha_t=\alpha_{\text{min}}$

#### 习题（learning_rate_schedule）：实现余弦学习率调度（1分）
测试：`uv run pytest -k test_get_lr_cosine_schedule`

### 4.5 梯度裁剪
约束全局梯度L2范数不超过阈值M，防止梯度爆炸：
$$g_{\text{new}} = g \cdot \frac{M}{\|g\|_2+\varepsilon},\quad \varepsilon=1\mathrm{e}{-6}$$

#### 习题（gradient_clipping）：梯度裁剪函数（1分）
原地修改参数梯度；测试：`uv run pytest -k test_gradient_clipping`

---

## 5 完整训练循环搭建
### 5.1 数据加载器
将一维超长Token ID数组，滑动截取固定上下文长度的输入序列与目标序列（输入右移一位即为标签）；批量组装张量并搬运至指定设备（CPU/CUDA/MPS）。
超大文件采用内存映射`np.memmap`按需读取，无需一次性载入全部数据。

#### 习题（data_loading）：批量数据加载函数（2分）
入参：Token ID数组、batch_size、上下文长度、设备；返回输入张量、目标张量。
测试：`uv run pytest -k test_get_batch`

### 5.2 断点续训检查点保存/加载
检查点必须存储：模型权重state_dict、优化器state_dict、当前迭代步数；支持中途停止后完整恢复训练。

#### 习题（checkpointing）：编写保存&加载检查点函数（1分）
```python
def save_checkpoint(model, optimizer, iteration, out):
    pass
def load_checkpoint(src, model, optimizer) -> int:
    # 返回读取到的迭代步数
    pass
```
测试：`uv run pytest -k test_checkpointing`

### 5.3 整合全套代码：主训练脚本
#### 习题（training_together）：完整训练入口脚本（4分）
需求：
1. 命令行参数配置所有模型、优化器超参
2. memmap加载超大训练/验证数据集
3. 定期保存断点
4. 控制台+W&B日志记录训练损失、验证损失、困惑度

---

## 6 文本生成解码算法
训练完成后实现推理采样，支持两大主流生成技巧：
1. 温度缩放（Temperature）：调整分布平滑度，τ越小输出越确定
$$p_i=\frac{\exp(v_i/\tau)}{\sum_j\exp(v_j/\tau)}$$
2. Top-p（核采样）：累积概率p阈值截断低概率Token，提升生成流畅度

#### 习题（decoding）：文本生成解码函数（3分）
功能需求：
- 输入提示词，循环采样生成文本至最大长度或`<|endoftext|>`
- 可配置温度、top-p参数

---

## 7 全套消融实验与排行榜提交
### 7.1 TinyStories小数据集基础调优
基础模型配置（约1700万参数）：
词汇10000、上下文256、d_model=512、d_ff=1344、4层16头、RoPE θ=10000
总处理Token数：327,680,000

#### 习题（learning_rate）：学习率消融调优（3分）
(a) 多组学习率扫参，绘制损失曲线；目标验证损失≤1.45
(b) 探索收敛临界点学习率与最优学习率的关系

#### 习题（batch_size_experiment）：批次大小消融（1分）
遍历多种batch_size，对比训练收敛速度、最终损失，总结批次大小影响规律。

#### 习题（generate）：模型文本生成样例（1分）
输出不少于256Token生成文本，点评流畅度，并说明影响生成质量的两个关键因素。

### 7.2 架构模块消融对照实验
1. **移除RMSNorm消融**：去掉所有归一化层训练，对比收敛情况、损失曲线
2. **Pre-Norm改为Post-Norm消融**：对比两种归一化排布训练效果差异
3. **移除位置编码（NoPE）消融**：无任何位置信息训练，对比RoPE基线
4. **SwiGLU替换为普通SiLU前馈**：参数量对齐前提下对比性能

### 7.3 OpenWebText通用网页语料训练
使用完全相同模型架构、总训练步数在OpenWebText上训练；网页文本复杂度远高于儿童故事，损失会显著更高。
#### 习题（main_experiment）：OWT数据集训练实验（2分）
绘制损失曲线，对比两份数据集损失差异；生成文本并分析流畅度变差的原因。

### 7.4 自定义架构改进 + 课程排行榜比拼
在45分钟B200算力时限内，基于OpenWebText数据集优化模型架构/超参，尽可能压低验证困惑度；提交PR至排行榜仓库，最终按照验证损失排名计分。
可选优化方向：权重绑定、优化器微调、归一化改进、参数初始化优化、学习率策略优化等。

#### 习题（leaderboard）：排行榜提交（6分）
提交内容：最终验证损失、带时钟时间轴的损失曲线、详细改进方案说明。

---

## 参考文献
（原文文末参考文献列表已完整对应翻译）