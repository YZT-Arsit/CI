# CPU 真实Qwen tokenizer与预处理诊断

**结论：这些预处理检查可以在CPU完成，现已完成；不应笼统推迟到GPU阶段。** 当前结果仅覆盖固定官方tokenizer、模板、数据分界、mask/padding规则及合成log-probability归约，不包含真实模型forward/backward、训练器接入或GPU数值验证。

机器可读结果：`summary.json`；原18个worker直接生成的未增补结果保留为 `summary_raw.json`。新增资产比对与因果控制独立保存，没有改写原始boundary控制或主实验源代码。

## 资产与输入

- 官方资产：`Qwen/Qwen2.5-7B-Instruct`，固定revision `a09a35458c702b33eeacc393d103063234e8bc28`。只下载 [tokenizer.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer.json)（7,031,645字节）和 [tokenizer_config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer_config.json)（7,305字节），没有模型权重下载。
- tokenizer.json SHA256：`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`。
- tokenizer_config.json SHA256：`5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`。
- 隔离依赖位于 `dependencies/`：tokenizers 0.23.3、Jinja2 3.1.6、MarkupSafe 3.0.4。wheel版本、来源和hash见 `dependencies_manifest.json`；默认全局解释器仍未安装这些包。没有安装torch/transformers，也未启动Ollama。
- 冻结输入为当前 `dataset/preferences.jsonl`，**1,824条去重偏好**，SHA256 `76a502402833d2f4618ff50e8ce39ae904200b9ebb53d065a80e269bff60f1e3`。另逐条确认原始shard的 **2,016/2,016** 记录均映射到这些已验证Prompt/chosen/rejected，192条为重复文本对。
- 主脚本及冻结快照SHA256：`cc89495adb67a7e0ec02f5ac07d9c17d58dc1db14cf766a871af5e7aa55f07be`。完整输入快照、分片、编码后的token序列和日志均已保存。

## 结果

| 检查 | 实际覆盖 | 结果 |
|---|---:|---|
| 完整官方chat_template渲染后的Prompt token前缀 | 1,824/1,824对 | chosen/rejected共享完全相同前缀，未依赖“分别tokenize后必然可拼接”的假设 |
| 回复终止符与监督范围 | 3,648分支 | 每分支唯一EOS=151645纳入completion监督span；模板EOS后换行单独保留为输入但不计监督，共3,648个尾部token |
| prompt/padding mask、下一token位移、position_ids | 64条唯一序列×16配置=1,024次 | 左/右padding、batch 1/2/8/32，以及真实PAD=151643和故意PAD=EOS对照均通过 |
| 合成log-probability求和/平均归约一致性 | 同上 | 与单条未padding基准最大绝对差0；这不是实际Qwen logits测试 |
| 256长度上限 | 1,824对 | 全部拒绝；普通右截会让3,456分支没有任何回复监督，并让全部3,648分支丢失EOS |
| 512/1024/2048/4096长度上限 | 每上限1,824对 | 当前短toy样本全部完整保留；Prompt范围252–335 tokens，不能据此声称长历史训练已通过 |
| 回复内嵌特殊EOS字符串 | 1个负对照 | 终止语义有歧义，阻断该样本 |
| 序列求和与序列平均 | 1个明确反例 | token级log-ratio为chosen=[0.1,0.1]、rejected=[0.15]时，sum差+0.05而mean差−0.05，排序反转 |

mask按位置与attention有效性构造，不按“token ID等于PAD”一刀切；否则PAD与EOS共用ID时会删除真实EOS。`position_ids=max(cumsum(attention_mask)-1,0)`，logits位置`[:-1]`与labels`[1:]`对齐。Prompt、padding和模板终止后的尾部token标为−100。

本诊断保持既定DPO条件序列log-likelihood**求和**口径；平均值只是显示目标差异的反例，没有更换loss或进行参数训练。

## 截断与因果边界

原 `results/boundary.json` 使用一条已有K2000交付记录作为长文本控制。它证明左截断会丢失这条原有记录，**未证明该记录是当前CI标签的因果必要信息**。原字段名中的critical不能被解读为因果认证；该限制已加入summary，原始产物未改。

独立新增 `causal_boundary.json` 匹配了当前数据中真实K1000身份交付的CI偏好及完整Prompt：先让A发送 `K1000 corresponds to Person000.` 后，同一B的band消息为VIOLATION；先让A执行SKIP则为VALID。两分支B的本地事实和turn相同，因此这项受控历史差异在声明toy环境中确实改变CI标签。

随后仅在文本控制的task字段追加干扰填充，形成15,362-token序列；256/512/1024/2048/4096左截均删除该已证相关receipt。独立脚本、源快照、核心文件hash与0.527秒运行receipt已保存。**没有把截断后的字符串当成合法Observation重新判标签，也没有运行模型观察截断后行为。** 结论仅是原环境标签会随这条历史改变，且相应文本可被截断删除。

预处理采用保守规则：任一Prompt/历史、完整回复或EOS会丢失时，整对拒绝，不把旧标签静默附到截短上下文上。它是诊断中的保护规则，尚未集成到真实训练器。

## 本地GGUF比对限制

固定HF资产与本地Ollama metadata的EOS/PAD ID一致，add_bos=false。模板源码不完全相同：工具调用说明中的花括号文本存在差异；一个无tools声明的Prompt及完整回复渲染示例相同。**未逐项比较GGUF vocab/merges，不声称两个tokenizer完全一致。** 本次全部tokenization结论只针对上述固定官方HF资产。

## 时间与证据范围

15个数据分片worker、3个预处理诊断worker、资产比对和因果控制，共20个实验worker，均硬限制9.5秒；最长实际0.594秒。下载与隔离安装也分别硬限制9.5秒，最长3.787秒。没有为满足时长而空转。

仍需后续真实模型/训练器验证attention与padding的实际数值、不同batch及dtype误差、reference/policy快照、梯度和优化器、真实动作分布及学习效果。当前数据仍只有一个底层任务/规则家族；这些CPU检查不会提高数据独立性或建立研究方法优势。
