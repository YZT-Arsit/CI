"""Build the human report from completed native audits, never from partial logs."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent


def main():
    raw = json.loads((R / 'repair/AUDIT.json').read_text())
    assert raw['errors'] == 0 and raw['episodes_replayed'] == 448
    views = json.loads((R / 'repair/VIEW_AUDIT.json').read_text())['results']
    totals = json.loads((R / 'RESULTS.json').read_text())
    reference = json.loads((R / 'reference/VIEW_AUDIT.json').read_text())
    norm_root = R.parent / '20261117_external_norm_probe'
    norm_result = json.loads((norm_root / 'RESULTS.json').read_text())
    assert (norm_root / 'REPORT.md').exists()
    ref_audit = json.loads((R / 'reference/AUDIT.json').read_text())
    assert ref_audit['errors'] == 0 and ref_audit['episodes_replayed'] == 112
    names = {'sampler_sft': '修复采样 + 普通监督', 'full_js': '完整动作一致性',
             'query_js': '仅关系判断一致性', 'value_js': '关系判断 + 动作价值类一致性'}
    rows = ['| 模型 | 一跳四回合全成功 | 三跳链 | 内部边 | 四节点环 | 严格成功回合 | 合规回合 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for arm, name in names.items():
        values = {row['family']: row for row in views if row['arm'] == arm}
        cells = [str(values[f]['all_four_episodes_strict_success']) + '/' + str(values[f]['semantic_world_pairs'])
                 for f in ['base', 'depth3', 'internal3', 'cycle4']]
        rows.append('| ' + name + ' | ' + ' | '.join(cells) + f" | {totals[arm]['strict_success']}/112 | {totals[arm]['valid_ci']}/112 |")
    values = {row['family']: row for row in reference['results']}
    cells = [str(values[f]['all_four_strict_success']) + '/' + str(values[f]['groups']) for f in ['base', 'depth3', 'internal3', 'cycle4']]
    rows.append('| 更早的普通监督（补充参考） | ' + ' | '.join(cells) + f" | {reference['strict_success']}/112 | {reference['valid_ci']}/112 |")
    conditions = ['| 模型 | 可最少分享 | 明确授权 | 历史未送达 | 无安全完成路线：合规 |', '|---|---:|---:|---:|---:|']
    for arm, name in names.items():
        values = {row['family']: row for row in views if row['arm'] == arm}
        cells = [str(values[f]['both_views_strict_success']) + '/4' for f in ['optional', 'authorized', 'draft']]
        conditions.append('| ' + name + ' | ' + ' | '.join(cells) + f" | {values['both']['both_views_valid_ci']}/4 |")
    timing = max(row['max_update_seconds'] or 0 for row in totals.values())
    report = f'''# 关系进入行动：一致性约束的拆分实验

本轮对先前失败配置做了明确的损失拆分：保留原数据、采样、初始化和训练预算，只改变一致性约束施加的位置。新增两个 CPU LoRA 检查点，共 160 次更新；主实验四个模型共 448 回合，并追加一个固定旧检查点的 112 回合参考。**完整研究目标仍未完成，GPU 大批量继续 NO_GO。**

## 这次改了什么

原完整动作一致性要求换名、换序后仍偏好同一个语义动作，即使几个动作具有相同任务价值。它比保持价值要求更强，但并非数学错误，也不能预先断定有害。两个新配置分别删除动作一致性项，或只比较相同完整价值类别内的概率总和；关系判断一致性和监督损失保持不变。

按价值分组时，安全完成、安全失败、隐私违反与非法输出概率保持区分。训练使用原有标签进行分组；评测时模型自己生成判断和行动，没有标签注入或规则代选。分组还会降低正则强度，因此即便出现改善，也不能单独归因于价值语义设计，更不能当成新方法有效性的充分证据。

数学检查与真实模型预检通过：切回完整动作模式时，三个角色的损失与原实现逐项完全一致；两种新模式均产生有限梯度，保留双侧梯度路径。每次训练更新最长 {timing:.3f} 秒。

## 完整回合结果

一跳等四列要求同一组在两个历史、两种呈现下的四个回合全部安全完成。后两列是固定混合条件下的总数；合规包含停止或未完成，不能替代任务效用。

{chr(10).join(rows)}

前四行是冻结的主实验。最后一行是单独登记的补充参考：选择此前已发布的较好浅层检查点，用同一批新世界评测，不再训练。它防止只胜过退化对照就宣称成功，但不能冒充主实验预先设计的第五组或独立验证。

补充参考在同一批世界的一跳仍为 8/8，比两个新配置的 4/8 和 2/8 更好；它的总体合规率也更低，仍没有同时解决安全与效用。因此本轮不能宣布新配置超过已有最好浅层能力。

局部改善与代价必须一起看：两个新配置的一跳 Planner 均为 32/32 正确，消除了完整动作一致性组的固定 A 行为；但只约束关系判断仍有 12/32 个 Worker 选择不能完成当前任务的 Minimal，价值类版本则有 18/32 个 Source 提前停止。与同采样普通监督相比，严格成功从 40/112 增至 58/112 和 54/112，合规却从 98/112 降至 93/112 和 88/112，分别多出 5 和 10 个违规回合。因此没有通过“不增加违规”的联合修复条件。更复杂的价值类方案也未显示优于较简单的仅关系判断约束。

{chr(10).join(conditions)}

前三个附加条件要求两种呈现都合规且完成；最后一列只检查合规，因为该条件本来不存在安全完成路线。

## 可核验范围

主实验重放 {raw['episodes_replayed']} 回合、{raw['decisions_replayed']} 个行动、{raw['queries_reconstructed']} 个查询，并独立核验 {raw['independent_label_states']} 个标签状态；补充参考重放 {ref_audit['episodes_replayed']} 回合。原生审计错误均为 0。输出格式、行动统计与时延详见 RESULTS.json；具体查询和行动原文均保留。

这些世界仍来自相同两个任务实例，只有一个训练种子。深层结构条件保留已有的提示局限，没有独立来源留出或自然规范迁移证据。即使某个浅层指标改善，也不等于关系算法或安全—效用问题完整解决。

## 排除错误解释

TRAIN 摘要审计有 1,920 次状态出现，去重后为 1,280 个观察状态。按角色、候选动作和正确完成判断分组，共 10 个摘要组，每组都存在可共同采用的最优动作。因此目前没有依据把失败归咎于“必须增加另一个安全标签”。这是有限数据的相容性检查，不是语言模型可学习性的证明。

另一方面，先前两个修复模型的 Worker 训练输入中各有 44/160 次判断图错误，其中 12/160 次的预测可完成动作与最优动作集合不相交。它提示行动监督可能遇到矛盾的自预测输入；尚未证明这就是退化原因，也没有在本轮偷换为真值输入策略。详情见 CONTRACT_AUDIT.json 与 INTERPRETATION.md。

## 自然规范与真实协作

本轮另做了[外部正负规范判断探针](../20261117_external_norm_probe/REPORT.md)，主探针记录 {norm_result['model_calls']}/64 次请求，完整执行状态为 {norm_result['complete']}；另有 64 次位置诊断，仍复用相同 32 个情境。原输出全部是裸 C，冻结格式评分为无效；移动信息不足选项后，模型全部随之改选 A/B，表明这套提示下全量弃答，尚无确定的规范区分证据。它使用不同的冻结模型和作者标签，必须与本次训练模型分开解释；不属于训练策略的跨任务迁移成绩，也不提供实际工作流效用证明。

32 跳图推断仍不等于 32 步 Agent 协作。本轮没有新增实际协作长度证据，已有角色隔离接口检查也不能填补这一要求。后续仍需完成关系与行动的稳健联合学习、独立来源和多种子验证、规范正负例与任务效用联合验证、真实多步消息依赖实验。

复核入口：[REPRODUCE.md](REPRODUCE.md)、[主实验审计](repair/VIEW_AUDIT.json)、[补充参考审计](reference/VIEW_AUDIT.json)。所有旧失败记录保持不变。
'''
    (R / 'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps({'report_written': True, 'primary_episodes': 448, 'secondary_episodes': 112, 'whole_goal_complete': False}))


if __name__ == '__main__':
    main()
