import requests
import sys
import argparse
import pymysql
import pandas as pd
import json

# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser(description="接收 Node.js 传入的参数")
parser.add_argument('--questions_detail', type=str, help='题目详情JSON字符串')
parser.add_argument('--skill', type=str, help='知识点名称')
parser.add_argument('--user_id', type=str, help='用户id')
args = parser.parse_args()

# 解析传入的JSON字符串
questions_detail = json.loads(args.questions_detail) if args.questions_detail else []
user_id = args.user_id
skill = args.skill

# print(questions_detail)
# 创建DataFrame
df = pd.DataFrame(questions_detail)

# 添加question_level列（根据题目索引确定难度）
# 难度配置
difficultyConfig = [
    {
        "total": 10,
        "simple": {"count": 5},
        "medium": {"count": 3},
        "hard": {"count": 2}
    },
    {
        "total": 15,
        "simple": {"count": 7},
        "medium": {"count": 5},
        "hard": {"count": 3}
    },
    {
        "total": 20,
        "simple": {"count": 12},
        "medium": {"count": 6},
        "hard": {"count": 2}
    }
]

# 根据题目总数匹配配置
template = len(df)
config = next((c for c in difficultyConfig if c["total"] == template), difficultyConfig[0])

# 添加question_level列
def get_question_level(index):
    if index < config["simple"]["count"]:
        return 0  # 简单
    elif index < config["simple"]["count"] + config["medium"]["count"]:
        return 1  # 中等
    else:
        return 2  # 困难

df['question_level'] = [get_question_level(i) for i in range(len(df))]

# 添加其他必要的列
df['user_id'] = user_id
df['skill'] = skill
df['cost_time'] = df['time_spent']
df['hint_count'] = 0  # 默认为0，因为传入数据中没有提示次数
df['correct'] = df['is_correct'].astype(int)

# 计算各难度正确题数
ec = df.loc[(df['question_level'] == 0) & (df['correct'] == 1)].shape[0]
mc = df.loc[(df['question_level'] == 1) & (df['correct'] == 1)].shape[0]
hc = df.loc[(df['question_level'] == 2) & (df['correct'] == 1)].shape[0]

# 获取各难度题数
easy = config["simple"]["count"]
middle = config["medium"]["count"]
hard = config["hard"]["count"]

def main(arg1, skill, template, easy_correct, middle_correct, hard_correct) -> dict:
    prompt1='''你是一位经验丰富的数学教师，请根据以下学生的做题记录，对该学生在【{}】这一知识点上的掌握情况进行全面分析，并给出针对性的学习建议。

【学生做题记录如下】：

### 一、题目难度分布与正确率：
- 简单题：共 {} 道，做对 {} 道，正确率 {}%。
- 中等题：共 {} 道，做对 {} 道，正确率 {}%。
- 困难题：共 {} 道，做对 {} 道，正确率 {}%。'''
    
    prompt1=prompt1.format(skill,easy,easy_correct,round(easy_correct/easy*100, 2) if easy > 0 else 0,
                           middle,middle_correct,round(middle_correct/middle*100, 2) if middle > 0 else 0,
                           hard,hard_correct,round(hard_correct/hard*100, 2) if hard > 0 else 0)
    
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
    '''
    
    sum_time=0
    easy_time=0
    middle_time=0
    hard_time=0
    sum_count=0
    
    for data in arg1.itertuples(index=True, name='Pandas'):
        sum_count += data.hint_count
        question_level=""
        if data.question_level == 0:
            easy_time += data.cost_time
            question_level = "简单"
        elif data.question_level == 1:
            middle_time += data.cost_time
            question_level = "中等"
        else:
            hard_time += data.cost_time
            question_level = "困难"
        sum_time += data.cost_time
        prompt += prompt11.format(data.Index+1, question_level, 
                                  "正确" if data.correct==1 else "错误", 
                                  data.cost_time, data.hint_count)
        prompt += "\n"
    
    avg_time = round(sum_time/template, 2) if template > 0 else 0
    avg_easy_time = round(easy_time/easy, 2) if easy > 0 else 0
    avg_middle_time = round(middle_time/middle, 2) if middle > 0 else 0
    avg_hard_time = round(hard_time/hard, 2) if hard > 0 else 0
    
    prompt2 = prompt2.format(avg_time, avg_easy_time, avg_middle_time, avg_hard_time)
    prompt3 = prompt3.format(sum_count, round(sum_count/template, 2) if template > 0 else 0)
    
    prompt4='''\n\n请你从以下几个方面进行分析：
1. **知识掌握程度评估**：该学生是否掌握了该知识点？在哪个难度层级上存在薄弱环节？
2. **解题能力分析**：是否存在解题速度慢、依赖提示等问题？反映出哪些思维或技巧方面的不足？
并在最后给出（未掌握、掌握较差、掌握良好、掌握优秀）中的任何一个
综合以上要求分析，学生对该知识点的掌握程度(如果用户需要重新学习该知识点，那就是未掌握)

'''
    final_prompt = prompt1 + "\n" + prompt2 + "\n" + prompt3 + "\n" + prompt + "\n" + prompt4
    
    return final_prompt

res = main(df, skill, len(df), ec, mc, hc)

def calculate_mastery(easy_score, easy_total, medium_score, medium_total, hard_score, hard_total):
    """
    计算认知维度掌握率和动态加权的总体掌握度
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
    
    # 计算动态权重
    total_rate = knowledge + comprehension + application
    w_knowledge = knowledge / total_rate if total_rate != 0 else 0
    w_comprehension = comprehension / total_rate if total_rate != 0 else 0
    w_application = application / total_rate if total_rate != 0 else 0
    
    # 计算总体掌握度
    overall_mastery = round(
        knowledge * w_knowledge + 
        comprehension * w_comprehension + 
        application * w_application, 
        2
    )
    master = 1 if overall_mastery >= 0.7 and knowledge >= 0.4 and comprehension >= 0.4 and application >= 0.4 else 0
    return {
        "knowledge": knowledge,
        "comprehention": comprehension,
        "application": application,
        "overall_mastery": overall_mastery,
        "master": master
    }

import re
# Ollama API 地址
url = "http://localhost:11434/api/generate"

# 请求数据
data = {
    "model": "MAGIC_final_deepseek_r1_7b:latest",  # 替换为你本地已加载的模型名称，如 llama3、mistral 等
    "prompt": res,
    "stream": False  # 设置为 False 表示不使用流式输出
}

# 发送 POST 请求
response = requests.post(url, json=data)

overall_mastery_statistic = calculate_mastery(ec, easy, mc, middle, hc, hard)

# 插入数据
try:
    # 检查响应状态
    if response.status_code == 200:
        result = response.json()
        
        cleaned_res = re.sub(r"<think>.*?</think>", "", result.get("response", ""), flags=re.DOTALL).strip()
        
        overall_mastery = 0
        if "未掌握" in cleaned_res:
            overall_mastery = 0
        elif "掌握较差" in cleaned_res:
            overall_mastery = 0.4
        elif "掌握良好" in cleaned_res:
            overall_mastery = 0.7
        else:
            overall_mastery = 1
           
        master = 1 if overall_mastery >= 0.4 else 0
        
        # 打印结果以便调试
        print(cleaned_res)
        
    else:
        print("请求失败，状态码：", response.status_code)
        print("错误信息：", response.text)
except Exception as e:
    print(f"请求失败: {e}")