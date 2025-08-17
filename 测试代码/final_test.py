
import pandas as pd
import requests
import re
import tqdm

# def calculate_mastery(easy_score, easy_total, medium_score, medium_total, hard_score, hard_total):
#     """
#     计算认知维度掌握率和动态加权的总体掌握度
    
#     参数:
#         easy_score: 简单题得分
#         easy_total: 简单题总分
#         medium_score: 中等题得分
#         medium_total: 中等题总分
#         hard_score: 困难题得分
#         hard_total: 困难题总分
        
#     返回:
#         dict: 包含认知维度掌握率和总体掌握度的字典
#     """
#     # 防止除零错误
#     if easy_total == 0:
#         knowledge = 0
#     else:
#         knowledge = round(easy_score / easy_total, 2)
        
#     if medium_total == 0:
#         comprehension = 0
#     else:
#         comprehension = round(medium_score / medium_total, 2)
        
#     if hard_total == 0:
#         application = 0
#     else:
#         application = round(hard_score / hard_total, 2)
    
#     # 计算动态权重
#     total_rate = knowledge + comprehension + application
#     w_knowledge = knowledge / total_rate if total_rate != 0 else 0
#     w_comprehension = comprehension / total_rate if total_rate != 0 else 0
#     w_application = application / total_rate if total_rate != 0 else 0
    
#     # 计算总体掌握度
#     overall_mastery = round(
#         knowledge * w_knowledge + 
#         comprehension * w_comprehension + 
#         application * w_application, 
#         2
#     )
#     master = 1 if overall_mastery >= 0.7 and knowledge >= 0.4 and comprehension >= 0.4 and application >= 0.4 else 0
#     return {
#         "knowledge": knowledge,
#         "comprehension": comprehension,
#         "application": application,
#         "overall_mastery": overall_mastery,
#         "master": master
#     }

def calculate_mastery(easy_score, easy_total, medium_score, medium_total, hard_score, hard_total):
    """
    计算认知维度掌握率和动态加权的总体掌握度
    
    参数:
        easy_score: 简单题得分
        easy_total: 简单题总分
        medium_score: 中等题得分
        medium_total: 中等题总分
        hard_score: 困难题得分
        hard_total: 困难题总分
        
    返回:
        dict: 包含认知维度掌握率和总体掌握度的字典
    """
    # 防止除零错误
    if easy_total == 0:
        knowledge = 0
    else:
        knowledge = round(easy_score / easy_total, 2)
        
    if medium_total == 0:
        comprehension = 0
    else:
        comprehension = round(medium_score / medium_total, 2)
        
    if hard_total == 0:
        application = 0
    else:
        application = round(hard_score / hard_total, 2)

    # 固定权重和惩罚因子
    fixed_weights = [0.3, 0.3, 0.4]  # 简单、中等、困难
    penalty_factor = 0.5

    # 初始动态权重计算
    total_rate = knowledge + comprehension + application
    w_knowledge = knowledge / total_rate if total_rate != 0 else 0
    w_comprehension = comprehension / total_rate if total_rate != 0 else 0
    w_application = application / total_rate if total_rate != 0 else 0

    # 检查是否有权重为0的情况并应用惩罚因子
    adjusted_weights = []
    for i in range(3):
        if [w_knowledge, w_comprehension, w_application][i] == 0:
            # 替换为固定权重并应用惩罚因子
            adjusted_weights.append(fixed_weights[i] * penalty_factor)
        else:
            adjusted_weights.append([w_knowledge, w_comprehension, w_application][i])

    # 归一化调整后的权重
    total_adjusted = sum(adjusted_weights)
    if total_adjusted == 0:
        # 所有调整后的权重为0时，默认使用固定权重
        w_knowledge, w_comprehension, w_application = fixed_weights
    else:
        w_knowledge = adjusted_weights[0] / total_adjusted
        w_comprehension = adjusted_weights[1] / total_adjusted
        w_application = adjusted_weights[2] / total_adjusted

    # 计算总体掌握度
    overall_mastery = round(
        knowledge * w_knowledge + 
        comprehension * w_comprehension + 
        application * w_application, 
        2
    )

    # 修改条件判断：至少两个维度 >= 0.4
    valid_dims = sum(1 for rate in [knowledge, comprehension, application] if rate >= 0.4)
    master = 1 if overall_mastery >= 0.6 and valid_dims >= 2 else 0

    return {
        "knowledge": knowledge,
        "comprehension": comprehension,
        "application": application,
        "overall_mastery": overall_mastery,
        "master": master
    }

def is_valid_correct(row):
    """
    判断题目是否为有效正确答案
    根据时间、尝试次数、提示使用等条件判断
    """
    # 首先必须是correct=1
    if row['correct'] != 1:
        return False
    
    # 根据不同难度判断时间范围
    if row['question_level'] == 0:  # 简单题
        if  row['cost_time'] > 90:
            if row['attempt_count'] >= 3  or row['hint_count'] >= 2 or row['bottom_hint'] == 1:
                return False
    elif row['question_level'] == 1:  # 中等题
        if  row['cost_time'] > 150:
            if row['attempt_count'] >= 3 or row['hint_count'] >= 2 or row['bottom_hint'] == 1:
                return False
    elif row['question_level'] == 2:  # 困难题
        if  row['cost_time'] > 300:
            if row['attempt_count'] >= 3 or row['hint_count'] >= 2 or row['bottom_hint'] == 1:
                return False

    # 所有条件都满足，认为是有效正确
    return True

# def build_prompt(skill, template, easy_total, medium_total, hard_total, 
#                  easy_correct, medium_correct, hard_correct, group_data):
#     """
#     构建发送给LLM的提示词
#     """
#     # 难度配置
#     easy = 5
#     middle = 3
#     hard = 2
#     if template == 15:
#         easy = 7
#         middle = 5
#         hard = 3
#     elif template == 20:
#         easy = 12
#         middle = 6
#         hard = 2

#     prompt1 = '''你是一位经验丰富的数学认知诊断专家，请根据以下学生的做题记录，评估该学生在【{}】知识点的掌握情况。

# 【学生做题记录如下】：

# ### 一、题目难度分布与正确率：
# - 简单题：共 {} 道，做对 {} 道，正确率 {}%。
# - 中等题：共 {} 道，做对 {} 道，正确率 {}%。
# - 困难题：共 {} 道，做对 {} 道，正确率 {}%。'''
    
#     prompt1 = prompt1.format(
#         skill, 
#         easy, easy_correct, round(easy_correct/easy*100, 2) if easy > 0 else 0,
#         middle, medium_correct, round(medium_correct/middle*100, 2) if middle > 0 else 0,
#         hard, hard_correct, round(hard_correct/hard*100, 2) if hard > 0 else 0
#     )
    
#     prompt2 = '''### 二、做题时间统计（单位：秒）：
#     - 平均每道题耗时：{} 秒。
#     - 简单题平均耗时：{} 秒。
#     - 中等题平均耗时：{} 秒。
#     - 困难题平均耗时：{} 秒。
#     '''
    
#     prompt3 = '''### 三、提示使用情况：
#     - 总共查看提示次数：{} 次。
#     - 每道题平均提示次数：{} 次。'''
    
#     prompt = '''### 四、具体做题情况：\n'''

#     prompt11 = '''   第{}题，题目难度{}，
#        -做题结果 {}
#        - 做题时间：{} 秒
#        - 使用提示次数：{}
#        - 尝试次数：{}
#        - 查看所有提示：{}
#     '''
    
#     sum_time = 0
#     easy_time = 0
#     middle_time = 0
#     hard_time = 0
#     sum_count = 0
#     sum_attempt = 0
#     sum_bottom = 0
    
#     easy_count = 0
#     middle_count = 0
#     hard_count = 0
    
#     for idx, (index, row) in enumerate(group_data.iterrows()):
#         sum_count += row['hint_count']
#         sum_attempt += row['attempt_count']
#         sum_bottom += row['bottom_hint']
#         question_level = ""
#         if row['question_level'] == 0:
#             easy_time += row['cost_time']
#             easy_count += 1
#             question_level = "简单"
#         elif row['question_level'] == 1:
#             middle_time += row['cost_time']
#             middle_count += 1
#             question_level = "中等"
#         else:
#             hard_time += row['cost_time']
#             hard_count += 1
#             question_level = "困难"
#         sum_time += row['cost_time']
#         prompt += prompt11.format(
#             idx + 1, 
#             question_level, 
#             "正确" if row['correct'] == 1 else "错误", 
#             row['cost_time'], 
#             row['hint_count'],
#             row['attempt_count'],
#             "是" if row['bottom_hint'] == 1 else "否"
#         )
#         prompt += "\n"
    
#     avg_time = round(sum_time/template, 2) if template > 0 else 0
#     avg_easy_time = round(easy_time/easy_count, 2) if easy_count > 0 else 0
#     avg_middle_time = round(middle_time/middle_count, 2) if middle_count > 0 else 0
#     avg_hard_time = round(hard_time/hard_count, 2) if hard_count > 0 else 0
    
#     prompt2 = prompt2.format(avg_time, avg_easy_time, avg_middle_time, avg_hard_time)
#     prompt3 = prompt3.format(sum_count, round(sum_count/template, 2) if template > 0 else 0)
    
#     prompt4='''
#         ## 分析方法

#         ### 1. 整体正确率分析
#         - 总正确率反映基础掌握情况

#         ### 2. 时间-正确率关系分析
#         根据不同场景下的时间表现判断掌握情况：
#         **对于答对的题目：**
#         - **短时间答对**（<30秒）：表示熟练掌握，概念清晰
#         - **中等时间答对**（30-300秒）：表示理解正确但需要思考时间
#         - **长时间答对**（>300秒）：表示掌握较差，需要大量思考或试错

#         **对于答错的题目：**
#         - **短时间答错**（<2秒）：可能是完全不会直接猜答案，或者过于粗心
#         - **长时间答错**（>300秒）：表示有一定理解但方法不当或概念混淆


#         ### 3. 异常表现关注
#         - **用时过短**（<=1秒）：高度怀疑是猜答案
#         - **用时过长**（>300秒）：可能遇到真正的知识盲点
#         - **频繁使用提示**：表示缺乏独立解题能力

#         ## 评估标准
#         请按照下方的指标进行计算：
#         未掌握：正确率极低，尝试次数多，频繁使用提示，尤其是简单题目也做错。
#         掌握较差：正确率较低，但可能在中等或困难题目上表现稍好，但仍有较多错误，尝试次数和提示次数较多。
#         掌握良好：正确率中等或较高，尝试次数和提示次数较少，能在中等难度题目上表现较好。
#         掌握优秀：正确率高，尝试次数少，几乎不用提示，能在困难题目上表现良好。
#         # 输出格式要求：
#          请返回以下 JSON 格式的内容，其中 master只能是（未掌握、掌握较差、掌握良好、掌握优秀），不要添加任何额外说明或解释：
#          {
#            "master": ""
#          }
    
#     '''
    
#     final_prompt = prompt1 + "\n" + prompt2 + "\n" + prompt3 + "\n" + prompt + "\n" + prompt4
#     return final_prompt



def build_prompt(skill, template, easy_total, medium_total, hard_total, 
                 easy_correct, medium_correct, hard_correct, group_data):
    """
    构建发送给LLM的提示词
    """
    # 难度配置
    easy = 5
    middle = 3
    hard = 2
    if template == 15:
        easy = 7
        middle = 5
        hard = 3
    elif template == 20:
        easy = 12
        middle = 6
        hard = 2

    prompt1 = '''你是一位经验丰富的数学认知诊断专家，请根据以下学生的做题记录，评估该学生在【{}】知识点的掌握情况。

【学生做题记录如下】：

### 一、题目难度分布与正确率：
- 简单题：共 {} 道，做对 {} 道，正确率 {}%。
- 中等题：共 {} 道，做对 {} 道，正确率 {}%。
- 困难题：共 {} 道，做对 {} 道，正确率 {}%。'''
    
    prompt1 = prompt1.format(
        skill, 
        easy, easy_correct, round(easy_correct/easy*100, 2) if easy > 0 else 0,
        middle, medium_correct, round(medium_correct/middle*100, 2) if middle > 0 else 0,
        hard, hard_correct, round(hard_correct/hard*100, 2) if hard > 0 else 0
    )
    
    prompt2 = '''### 二、做题时间统计（单位：秒）：
    - 平均每道题耗时：{} 秒。
    - 简单题平均耗时：{} 秒。
    - 中等题平均耗时：{} 秒。
    - 困难题平均耗时：{} 秒。
    '''
    
    prompt3 = '''### 三、提示使用情况：
    - 总共查看提示次数：{} 次。
    - 每道题平均提示次数：{} 次。'''
    
    prompt = '''### 四、具体做题情况：\n'''

    prompt11 = '''   第{}题，题目难度{}，
       -做题结果 {}
       - 做题时间：{} 秒
       - 使用提示次数：{}
       - 尝试次数：{}
      - 查看所有提示：{}
    '''
    
    sum_time = 0
    easy_time = 0
    middle_time = 0
    hard_time = 0
    sum_count = 0
    
    easy_count = 0
    middle_count = 0
    hard_count = 0
    
    for idx, (index, row) in enumerate(group_data.iterrows()):
        sum_count += row['hint_count']
        question_level = ""
        if row['question_level'] == 0:
            easy_time += row['cost_time']
            easy_count += 1
            question_level = "简单"
        elif row['question_level'] == 1:
            middle_time += row['cost_time']
            middle_count += 1
            question_level = "中等"
        else:
            hard_time += row['cost_time']
            hard_count += 1
            question_level = "困难"
        sum_time += row['cost_time']
        prompt += prompt11.format(
            idx + 1, 
            question_level, 
            "正确" if row['correct'] == 1 else "错误", 
            row['cost_time'], 
            row['hint_count'],
            row['attempt_count'],
            "是" if row['bottom_hint'] == 1 else "否",
            
        )
        prompt += "\n"
    
    avg_time = round(sum_time/template, 2) if template > 0 else 0
    avg_easy_time = round(easy_time/easy_count, 2) if easy_count > 0 else 0
    avg_middle_time = round(middle_time/middle_count, 2) if middle_count > 0 else 0
    avg_hard_time = round(hard_time/hard_count, 2) if hard_count > 0 else 0
    
    prompt2 = prompt2.format(avg_time, avg_easy_time, avg_middle_time, avg_hard_time)
    prompt3 = prompt3.format(sum_count, round(sum_count/template, 2) if template > 0 else 0)
    
#     prompt4 = '''\n\n请你从以下几个方面进行分析：
# 1. **知识掌握程度评估**：该学生是否掌握了该知识点？在哪个难度层级上存在薄弱环节？
# 2. **解题能力分析**：是否存在解题速度慢、依赖提示等问题？反映出哪些思维或技巧方面的不足？

# 综合以上要求分析，学生对该知识点的掌握程度(如果用户需要重新学习该知识点，那就是未掌握)
# ## 输出格式要求：
# 请返回以下 JSON 格式的内容，其中 master只能是（未掌握、掌握较差、掌握良好、掌握优秀），不要添加任何额外说明或解释：
# {
#   "master": ""
# }
# '''
    prompt4='''
        ## 分析方法

        ### 1. 整体正确率分析
        - 总正确率反映基础掌握情况
        - 难题表现掌握程度
        - 通类型题目解题时长波动大于50%需要重新考虑

        ### 2. 时间-正确率关系分析
        根据不同场景下的时间表现判断掌握情况：
        **对于答对的题目：**
        - **短时间答对**（<30秒）：表示熟练掌握，概念清晰
        - **中等时间答对**（30-300秒）：表示理解正确但需要思考时间
        - **长时间答对**（>300秒）：表示掌握较差，需要大量思考或试错

        **对于答错的题目：**
        - **短时间答错**（<2秒）：可能是完全不会直接猜答案，或者过于粗心
        - **长时间答错**（>300秒）：表示有一定理解但方法不当或概念混淆

        ### 3. 做题模式识别
        - 计算时间分布的离散程度（是否有极端值）
        - 分析提示使用模式
        - 识别基于实际表现的难度分层

        ### 4. 异常表现关注
        - **用时过短**（<2秒）：高度怀疑是猜答案
        - **用时过长**（>300秒）：可能遇到真正的知识盲点
        - **频繁使用提示**：表示缺乏独立解题能力

        ## 评估标准
        **掌握优秀**：
        - 总体正确率≥90% 且其中2种题型准确率都不低于40%
        - 大部分题目能在合理时间内完成
        - 很少依赖提示
        - 时间分布稳定，无明显异常

        **掌握良好**：
        - 正确率≥70% 且其中2种题型准确率都不低于40%
        - 多数题目能独立完成
        - 偶有题目需要较长思考时间但最终能答对
        - 提示使用适度

        **掌握较差**：
        - 正确率≥40% 且其中2种题型准确率都不低于40%
        - 虽然正确率较高，但简单题目用时过长
        - 经常依赖提示才能完成

        **未掌握**（满足以下任一条件）：

        条件1：总体正确率<40%
        条件2：存在3道及以上题目用时<2秒且查看提示（明显在猜答案）
        条件3：简单题平均用时>90秒且正确率<60%（基础不扎实）
        条件4：时间分布极度不合理：简单题用时超过中等题平均时间的1.5倍以上
        条件5：简单题和中等题准确率<40%
        条件6：当前题型查看提示次数超过该题型题目数的60%
        条件7：做题查看所有提示且尝试次数超过3次的题目数量超过总数量的60%
        条件8：单题做题时长波动很大
        
        # 输出格式要求：
         请返回以下 JSON 格式的内容，其中 master只能是（未掌握、掌握较差、掌握良好、掌握优秀），不要添加任何额外说明或解释：
         {
           "master": ""
         }
    
    '''
    
    final_prompt = prompt1 + "\n" + prompt2 + "\n" + prompt3 + "\n" + prompt + "\n" + prompt4
    
    return final_prompt


def get_llm_master(prompt):
    """
    调用LLM获取掌握度判断
    """
    # Ollama API 地址
    url = "http://localhost:11434/api/generate"

    # 请求数据
    # data = {
    #     "model": "MAGIC_final_deepseek_r1_7b:latest",
    #     "prompt": prompt,
    #     "stream": False
    # }
    # 请求数据
    data = {
        "model": "deepseek-r1:14b",
        "prompt": prompt,
        "stream": False
    }

    try:
        # 发送 POST 请求
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            cleaned_res = re.sub(r"\<think\>.*?\<\/think\>", "", result.get("response", ""), flags=re.DOTALL).strip()
            
            overall_mastery = 0
            if "未掌握" in cleaned_res:
                overall_mastery = 0
                print("未掌握")
            elif "掌握较差" in cleaned_res:
                overall_mastery = 0.4
                print("掌握较差")
            elif "掌握良好" in cleaned_res:
                overall_mastery = 0.7
                print("掌握良好")
            else:
                overall_mastery = 1
                print("已掌握")
            
            master = 1 if overall_mastery >= 0.4 else 0
            return {
                "inference": result.get("response", ""),
                "overall_mastery": overall_mastery,
                "master": master
            }
        else:
            print(f"请求失败，状态码：{response.status_code}")
            return {"overall_mastery": 0, "master": 0}
    except Exception as e:
        print(f"LLM请求失败: {e}")
        return {"overall_mastery": 0, "master": 0}

def initialize_output_files(output_prefix):
    """
    初始化输出文件，写入表头
    """
    # 初始化统计掌握度结果文件
    statistic_columns = ['user_id', 'skill_id', 'skill', 'total_time', 'template', 
                        'easy_correct', 'middle_correct', 'hard_correct', 'knowledge', 
                        'comprehension', 'application', 'overall_mastery', 'master', 'date']
    pd.DataFrame(columns=statistic_columns).to_csv(
        f"{output_prefix}_statistic_mastery_data.csv", index=False)
    
    # 初始化LLM掌握度结果文件
    llm_columns = ['user_id', 'skill_id', 'skill', 'total_time', 'template', 
                  'easy_correct', 'middle_correct', 'hard_correct', 'knowledge', 
                  'comprehension', 'application', 'overall_mastery', 'master', 'date']
    pd.DataFrame(columns=llm_columns).to_csv(
        f"{output_prefix}_llm_mastery_data.csv", index=False)
    
    # 初始化比较结果文件
    comparison_columns = ['user_id', 'skill_id', 'skill', 'statistic_master', 'llm_master', 'correct', 'date']
    pd.DataFrame(columns=comparison_columns).to_csv(
        f"{output_prefix}_comparison_data.csv", index=False)
    
    # 初始化用户知识档案文件
    profile_columns = ['user_id', 'accquired_knowledges', 'weak_knowledges', 'current_knowledge', 'preference', 'current_path']
    pd.DataFrame(columns=profile_columns).to_csv(
        f"{output_prefix}_user_knowledge_profile.csv", index=False)

def append_to_csv(data, filename):
    """
    追加数据到CSV文件
    """
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
    df.to_csv(filename, mode='a', header=False, index=False)

def process_user_data(input_csv, output_prefix, batch_size=10):
    # 初始化输出文件
    initialize_output_files(output_prefix)
    
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 按user_id, skill_id, date分组
    grouped = df.groupby(['user_id', 'skill_id', 'date'])
    
    # 用户知识档案数据（内存中维护）
    user_profiles = {}
    
    # 计数器
    processed_count = 0
    
    # 处理每组数据
    for (user_id, skill_id, date), group in tqdm.tqdm(grouped):
        # 获取skill信息
        skill = group['skill'].iloc[0] if not group['skill'].empty else ''
        
        # 计算总时间
        total_time = group['cost_time'].sum()
        
        # 计算题目总数（模板数量）
        template = len(group)
        
        # 统计LLM结果使用的正确数量（使用新的判断逻辑）
        llm_easy_correct = len(group[
            (group['question_level'] == 0) & 
            group.apply(is_valid_correct, axis=1)
        ])
        llm_middle_correct = len(group[
            (group['question_level'] == 1) & 
            group.apply(is_valid_correct, axis=1)
        ])
        llm_hard_correct = len(group[
            (group['question_level'] == 2) & 
            group.apply(is_valid_correct, axis=1)
        ])
        
        # 统计Statistic结果使用的正确数量（同样使用新的判断逻辑）
        statistic_easy_correct = len(group[
            (group['question_level'] == 0) & 
            group.apply(is_valid_correct, axis=1)
        ])
        statistic_middle_correct = len(group[
            (group['question_level'] == 1) & 
            group.apply(is_valid_correct, axis=1)
        ])
        statistic_hard_correct = len(group[
            (group['question_level'] == 2) & 
            group.apply(is_valid_correct, axis=1)
        ])
        
        # 计算各难度等级总数
        easy_total = len(group[group['question_level'] == 0])
        medium_total = len(group[group['question_level'] == 1])
        hard_total = len(group[group['question_level'] == 2])
        
        # 计算统计掌握度
        statistic_result = calculate_mastery(statistic_easy_correct, easy_total, statistic_middle_correct, medium_total, statistic_hard_correct, hard_total)
        
        # 构建提示词并获取LLM掌握度
        prompt = build_prompt(skill, template, easy_total, medium_total, hard_total,
                            llm_easy_correct, llm_middle_correct, llm_hard_correct, group)
        llm_result = get_llm_master(prompt)
        
        # 构建要追加的数据
        statistic_row = {
            'user_id': user_id,
            'skill_id': skill_id,
            'skill': skill,
            'total_time': total_time,
            'template': template,
            'easy_correct': statistic_easy_correct,
            'middle_correct': statistic_middle_correct,
            'hard_correct': statistic_hard_correct,
            'knowledge': statistic_result['knowledge'],
            'comprehension': statistic_result['comprehension'],
            'application': statistic_result['application'],
            'overall_mastery': statistic_result['overall_mastery'],
            'master': statistic_result['master'],
            'date': date
        }
        
        llm_row = {
            'user_id': user_id,
            'skill_id': skill_id,
            'skill': skill,
            'total_time': total_time,
            'template': template,
            'easy_correct': llm_easy_correct,
            'middle_correct': llm_middle_correct,
            'hard_correct': llm_hard_correct,
            'knowledge': statistic_result['knowledge'],
            'comprehension': statistic_result['comprehension'],
            'application': statistic_result['application'],
            'overall_mastery': llm_result['overall_mastery'],
            'master': llm_result['master'],
            "inference": llm_result['inference'],
            'date': date
        }
        
        comparison_row = {
            'user_id': user_id,
            'skill_id': skill_id,
            'skill': skill,
            'statistic_master': statistic_result['master'],
            'llm_master': llm_result['master'],
            'correct': 1 if statistic_result['master'] == llm_result['master'] else 0,
            'date': date
        }
        
        # 追加到对应的CSV文件
        append_to_csv(statistic_row, f"{output_prefix}_statistic_mastery_data.csv")
        append_to_csv(llm_row, f"{output_prefix}_llm_mastery_data.csv")
        append_to_csv(comparison_row, f"{output_prefix}_comparison_data.csv")
        
        # 构建用户知识档案数据
        if user_id not in user_profiles:
            user_profiles[user_id] = {
                'user_id': user_id,
                'accquired_knowledges': [],
                'weak_knowledges': [],
                'current_knowledge': '',
                'preference': '',
                'current_path': ''
            }
        
        knowledge_info = {
            "skill": skill,
            "skill_id": skill_id,
            "overall_mastery": statistic_result['overall_mastery']*0.6+llm_result['overall_mastery']*0.4
        }
        
        # 根据掌握情况分类
        if llm_result['master'] == 1:  # 已掌握
            user_profiles[user_id]['accquired_knowledges'].append(knowledge_info)
        else:  # 未掌握
            user_profiles[user_id]['weak_knowledges'].append(knowledge_info)
        
        processed_count += 1
        
        # 每处理 batch_size 条数据保存一次用户知识档案
        if processed_count % batch_size == 0:
            print(f"已处理 {processed_count} 条数据")
            save_user_knowledge_profile(user_profiles, output_prefix)
    
    # 保存最终的用户知识档案
    print(f"处理完成，共处理 {processed_count} 条数据，正在保存最终用户知识档案...")
    save_user_knowledge_profile(user_profiles, output_prefix)
    
    return True

def save_user_knowledge_profile(user_profiles, output_prefix):
    """
    保存用户知识档案到CSV文件（覆盖方式）
    """
    # 准备用户知识档案数据
    profile_data = list(user_profiles.values())
    profile_df = pd.DataFrame(profile_data)
    
    # 将JSON数组转换为字符串格式保存
    profile_df['accquired_knowledges'] = profile_df['accquired_knowledges'].apply(lambda x: str(x) if x else '[]')
    profile_df['weak_knowledges'] = profile_df['weak_knowledges'].apply(lambda x: str(x) if x else '[]')
    
    # 保存用户知识档案（覆盖之前的文件）
    profile_output_file = f"{output_prefix}_user_knowledge_profile.csv"
    
    # 重新写入表头和数据
    profile_columns = ['user_id', 'accquired_knowledges', 'weak_knowledges', 'current_knowledge', 'preference', 'current_path']
    pd.DataFrame(columns=profile_columns).to_csv(profile_output_file, index=False)
    if not profile_df.empty:
        profile_df.to_csv(profile_output_file, mode='a', header=False, index=False)
    
    print(f"用户知识档案已更新到 {profile_output_file}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    
    # l=["user_middle_test_log","new_user_one_log","new_user_repeat_log","user_beginner_test_log","user_senior_test_log"]
    l=["user_middle_test_log","new_user_one_log","user_beginner_test_log","user_senior_test_log"]
    # l=["new_user_repeat_log"]
    
    for t in l:
        input_file = "/home/Data/final_test_data/"+t +".csv" # 输入CSV文件路径
        output_prefix = "/home/lms/project/dify-backend/data/test_data/"+t+"/new_prompt4_"+t # 输出文件前缀
        print(output_prefix)
        # 每处理10条数据更新一次用户知识档案
        process_user_data(input_file, output_prefix, batch_size=10)
    # f="user_middle_test_log"
    # input_file = "/home/Data/final_test_data/"+f +".csv" # 输入CSV文件路径
    # output_prefix = "/home/lms/project/dify-backend/data/test_data/output3_"+f # 输出文件前缀
    
    # # 每处理10条数据更新一次用户知识档案
    # process_user_data(input_file, output_prefix, batch_size=10)
    
    
    
    
    
    
