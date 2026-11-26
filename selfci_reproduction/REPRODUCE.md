# 复测入口

工作目录为本文件所在目录。基础模型从 ModelScope 下载到同名目录。运行脚本会将模型权重分片逐个与 Hugging Face 发布哈希核对。Qwen3-4B-Instruct-2507 的 HF 修订为 `cdbee75f17c01a7cc42f958dc650907174af0554`，Qwen2.5-7B-Instruct 为 `a09a35458c702b33eeacc393d103063234e8bc28`。作者仓库固定为 `9cf9b2a53242c69e5f46150bbc34aea0859b98e4`。用户提供的论文 SHA-256 为 `935384b27432033400729290600c5e65e38ac8959c4e18396523f734463b7ef6`。

## CI-RL 表 1 主结果

本次复测使用 vLLM 0.16.0、PyTorch 2.9.1+cu128 和 Transformers 4.57.6，完整版本见 `results/environment_vllm.txt`。在 GPU 机的实验目录运行以下命令。包装脚本只将公开适配器所记录的底座地址指向已核验的本地权重，然后调用固定提交中未经改写的 `upstream/run_test.py`。适配器合并、聊天模板、测试切分、五轮生成与关键词评分都在上游入口执行。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
/root/selfci-vllm/bin/python run_upstream_test.py 4b_initial 4b_selfci 4b_grpo 4b_ci_sft
/root/selfci-vllm/bin/python run_upstream_test.py 7b_initial 7b_selfci
/root/selfci-vllm/bin/python summarize_upstream.py results/upstream
```

每个方法输出 73 条提示 × 5 轮。每轮沿用作者实现的同一个采样种子 42。`results/upstream/audit.json` 根据原始回复核对各轮 prompt ID、命中计数、解析成功数及重复输出。额外底座的 SHA 和可用适配器记录在 `extended_files.json`。联网机可用 `stage_local_base.py` 逐文件校验、下载再传输，例如 `python stage_local_base.py qwen3 Qwen3-4B`。GPU 上可用 `/root/selfci-vllm/bin/python extended_table1.py --models qwen3 --base-dir /path/to/Qwen3-4B` 调用相同的官方入口。

## 五个不同种子的敏感性实验

联网机器运行 `python fetch_adapters.py` 可领取固定修订的学生适配器。各自的 source.json 包含源地址、修订和文件哈希。将 adapters、data、上游代码和脚本放入 Linux GPU 工作目录，在能读取本地模型的 Python 环境中运行。

```bash
env/bin/python prepare_data.py --model ./Qwen3-4B-Instruct-2507 --output verified_remote
ENGINE=vllm PYTHON_BIN=/root/selfci-vllm/bin/python bash run_table1.sh
env/bin/python summarize_ci.py results
ENGINE=vllm PYTHON_BIN=/root/selfci-vllm/bin/python bash run_7b.sh
```

run_table1.sh 的四个模型使用同一 73 条测试输入，分别用种子 42–46 生成回复。默认 Transformers、BF16、每批 24 条。7B 脚本每批 16 条。可在另一个结果目录运行 vLLM 后端；已绑定的结果目录会拒绝混入其他配置。metrics.json 保留单种子分数与种子间标准差。paired_ci.json 使用同一批测试提示进行配对自助抽样。分母为提示数乘以种子数，置信区间按提示聚类。

## KCI 统一多角色留出集

在本研究工作区运行 `python prepare_kci.py` 导出验证提示与评分标签。GPU 只需 prompts.jsonl.gz。两组模型分别按原生聊天模板作贪心生成，每条最多 384 个新 token。推理过程保留原始文本、结束原因与 token 数。评分可完全在本地 CPU 执行。

```bash
ENGINE=vllm PYTHON_BIN=/root/selfci-vllm/bin/python BATCH=128 bash run_kci.sh
python eval_kci.py compare --prompts kci_inputs/prompts.jsonl.gz \
    --labels kci_inputs/labels.jsonl.gz --base-run results/kci_initial \
    --selfci-run results/kci_selfci --out results/kci_paired
```

paired.jsonl.gz 根据 record_id 关联两组输出。summary.json 给出来源组、域、执行角色、任务类型与菜单类型的计数。存在多种可行动作的状态采用原有标签；合法但缺少标注的选择保留为单独类别。

7B 使用同一组角色提示与标签。将生成命令的环境增加 `MODEL=Qwen2.5-7B-Instruct ADAPTER=adapters/selfci_7b KCI_OUT_PREFIX=kci7`，评分命令中的三个结果目录分别换成 `kci7_initial`、`kci7_selfci` 和 `kci7_paired`。

另行检查输出上限时，对同一完整留出集设置 `MAX_TOKENS=2048 CONTEXT=8192`，结果前缀分别使用 `kci_long` 和 `kci7_long`。来源组差值和配对区间用 `python aggregate_pairs.py` 从已有结果重新计算，重采样种子为 20261125，重复 10,000 次。CI-RL 以测试提示为采样单位，KCI 以 40 个来源组为采样单位。

## PrivacyLens 文本动作与泄漏检测

`prepared/privacylens` 保存 493 个重建输入和原基准的开源检测代码。固定评审模型为 Mistral-7B-Instruct-v0.2。运行 `bash run_privacylens.sh` 可产生每个 4B 方法的五轮文本动作、逐敏感事实检测回答和 `lr.json`。动作没有交给真实工具执行。评分将无法解析的动作纳入总分母并按原规则计为未发现泄漏，应同时查看 `unparsed`。完整比较还需要论文的 Helpful 和 ALR 评审。

## 扩展模型来源

扩展 CI-RL 入口逐文件核对固定版本的底座及作者适配器，并保留每行结果的来源信息。Llama 3.1 的公开 ModelScope 分发同时提供权重、LICENSE 和 USE_POLICY。已核对公开清单四个分片的 SHA-256，以及 tokenizer、配置和索引与 Meta Hub 固定提交的 Git blob 身份；实际使用前再次核对本地字节。[ModelScope 分发](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct) · [Meta 模型卡](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659)
