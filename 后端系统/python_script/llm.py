import requests
import sys
import argparse
import pymysql
import pandas as pd
from datetime import datetime
import json
# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser(description="接收 Node.js 传入的参数")
parser.add_argument('--first_id', type=int, help='ID')
parser.add_argument('--rows', type=int, help='rows')
parser.add_argument('--path_id', type=int, help='path_id')


args = parser.parse_args()

id=args.first_id
row=args.rows
path_id=args.path_id


# 连接数据库
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    database='dify'
)

# 读取原始数据（比如只处理今天的数据）
query = f"SELECT * FROM user_skill_logs WHERE id >= {id} limit {row}"
# print(query)
df = pd.read_sql(query, conn)
# print(type(df))
# print(df)

ec=df.loc[(df['question_level'] == 0) & (df['correct'] == 1)].shape[0]
mc=df.loc[(df['question_level'] == 1) & (df['correct'] == 1)].shape[0]
hc=df.loc[(df['question_level'] == 2) & (df['correct'] == 1)].shape[0]
# print(ec,mc,hc)
template=len(df)

# for row in df.itertuples(index=True, name='Pandas'):
#     print(f"Index: {row.Index}, skill: {row.skill}, level: {row.question_level}, cost_time: {row.cost_time}, hint_cont: {row.hint_count}")
easy=5
middle=3
hard=2
if template==15:
    easy = 7
    middle = 5
    hard = 3
elif template==20:
    easy = 12
    middle = 6
    hard = 2

def main(arg1, skill,template,easy_correct,middle_correct,hard_correct) -> dict:
    prompt1='''你是一位经验丰富的数学教师，请根据以下学生的做题记录，对该学生在【{}】这一知识点上的掌握情况进行全面分析，并给出针对性的学习建议。

【学生做题记录如下】：

### 一、题目难度分布与正确率：
- 简单题：共 {} 道，做对 {} 道，正确率 {}%。
- 中等题：共 {} 道，做对 {} 道，正确率 {}%。
- 困难题：共 {} 道，做对 {} 道，正确率 {}%。'''
    
    prompt1=prompt1.format(skill,easy,easy_correct,easy_correct/easy*100,middle,middle_correct,middle_correct/middle*100,hard,hard_correct,hard_correct/hard*100)
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
        sum_count+=data.hint_count
        question_level=""
        if data.question_level == 0:
            easy_time+=data.cost_time
            question_level = "简单"
        elif data.question_level == 1:
            middle_time+=data.cost_time
            question_level = "中等"
        else:
            hard_time+=data.cost_time
            question_level = "困难"
        sum_time+=data.cost_time
        prompt+=prompt11.format(data.Index+1,question_level,"正确" if data.correct==1 else "错误",data.cost_time,data.hint_count)
        prompt+="\n"
    prompt2=prompt2.format(round(sum_time/template,2),round(easy_time/easy,2),round(middle_time/middle,2),round(hard_time/hard,2))
    prompt3=prompt3.format(sum_count,round(sum_count/template,2))
    prompt4='''\n\n请你从以下几个方面进行分析：
1. **知识掌握程度评估**：该学生是否掌握了该知识点？在哪个难度层级上存在薄弱环节？
2. **解题能力分析**：是否存在解题速度慢、依赖提示等问题？反映出哪些思维或技巧方面的不足？

综合以上要求分析，学生对该知识点的掌握程度(如果用户需要重新学习该知识点，那就是未掌握)
## 输出格式要求：
请返回以下 JSON 格式的内容，其中 master只能是（未掌握、掌握较差、掌握良好、掌握优秀），不要添加任何额外说明或解释：
{
  "master": ""
}
'''
    final_prompt=prompt1+"\n"+prompt2+"\n"+prompt3+"\n"+prompt+"\n"+prompt4
    
    return final_prompt



res=main(df,df.loc[0]["skill"],len(df),ec,mc,hc)
# print(res)

# def calculate_mastery():
#     return 0
#

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
    # 计算各维度掌握率
    knowledge = round(easy_score / easy_total, 2)
    comprehension = round(medium_score / medium_total, 2)
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
    master = 1 if overall_mastery >= 0.7 and knowledge>=0.4 and comprehension>=0.4 and application >=0.4 else 0
    return {
        "knowledge":knowledge,
        "comprehention":comprehension,
        "application":application,
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

overall_mastery_statistic=calculate_mastery(ec,easy,mc,middle,hc,hard)
# print(overall_mastery_statistic)
# 插入数据
try:
    data_to_insert = {
        'user_id': df.loc[0]["user_id"],
        'skill_id': df.loc[0]["skill_id"],
        'path_id':path_id,
        'skill': df.loc[0]["skill"],
        'template': len(df),
        'easy_correct': ec,
        'middle_correct': mc,
        'hard_correct': hc,
        'knowledge':overall_mastery_statistic["knowledge"],
        'comprehention':overall_mastery_statistic["comprehention"],
        'application':overall_mastery_statistic["application"],
        'overall_mastery': overall_mastery_statistic["overall_mastery"],
        'master': overall_mastery_statistic["master"]
    }
    with conn.cursor() as cursor:
        sql = """
           INSERT INTO user_skill_mastery_statistic (user_id,path_id, skill_id, skill, template, easy_correct, middle_correct, hard_correct, knowledge, comprehension, application, overall_mastery, master, count)
            VALUES 
            (%(user_id)s,%(path_id)s, %(skill_id)s, %(skill)s, %(template)s, %(easy_correct)s, %(middle_correct)s, %(hard_correct)s, %(knowledge)s,  %(comprehention)s, %(application)s, %(overall_mastery)s, %(master)s,1)
            ON DUPLICATE KEY UPDATE
            path_id= VALUES(path_id),
            template = VALUES(template),
            easy_correct = VALUES(easy_correct),
            middle_correct = VALUES(middle_correct),
            hard_correct = VALUES(hard_correct),
            knowledge = VALUES(knowledge),
            comprehension = VALUES(comprehension),
            application = VALUES(application),
            overall_mastery = VALUES(overall_mastery),
            master = VALUES(master),
            count= count+1
            """
        # print(sql)
        cursor.execute(sql, data_to_insert)
        conn.commit()
        # print("✅ statistic插入成功！")
except Exception as e:
    print(f"❌ 插入失败: {e}")
# 检查响应状态
if response.status_code == 200:
    result = response.json()
    
    cleaned_res=re.sub(r"<think>.*?</think>", "",result.get("response", ""), flags=re.DOTALL).strip()
    
    # print(cleaned_res)
    # 存入新表（如果没有就创建）
    cursor = conn.cursor()

    overall_mastery=0
    if "未掌握" in cleaned_res:
        overall_mastery=0
        print("未掌握")
    elif "掌握较差" in cleaned_res:
        overall_mastery=0.4
        print("掌握较差")
    elif "掌握良好" in cleaned_res:
        overall_mastery=0.7
        print("掌握良好")
    else:
        overall_mastery=1
        print("已掌握")
    master=1 if overall_mastery>=0.4 else 0
    
    # print("已掌握" if master==1 else "未掌握")
    # 要插入的数据
    data_to_insert = {
        'user_id': df.loc[0]["user_id"],
        'path_id':path_id,
        'skill_id': df.loc[0]["skill_id"],
        'skill': df.loc[0]["skill"],
        'template': len(df),
        'inference': result.get("response", ""),
        'overall_mastery': overall_mastery,
        'master': master,
        'correct': 1 if overall_mastery_statistic["master"]==master else 0,
    }
    # 在 # 插入数据 部分之后添加以下代码

    # 更新用户知识档案
    try:
        with conn.cursor() as cursor:
            # 查询用户当前的知识档案
            query_profile = "SELECT accquired_knowledges, weak_knowledges FROM user_knowledge_profile WHERE user_id = %s"
            cursor.execute(query_profile, (data_to_insert['user_id'],))
            profile_result = cursor.fetchone()
            
            # 构建新的知识点对象
            skill_obj = {
                "skill": data_to_insert['skill'],
                "skill_id": int(data_to_insert['skill_id']),
                "overall_mastery": round(data_to_insert['overall_mastery'],2)
            }
            
            # 根据掌握情况决定存储位置
            if data_to_insert['master'] == 1:  # 掌握了
                # 处理已掌握知识点
                if profile_result:
                    # 如果用户档案存在
                    accquired = profile_result[0] if profile_result[0] else []
                    weak = profile_result[1] if profile_result[1] else []
                    
                    # 从弱知识点中移除（如果存在）
                    weak = [s for s in weak if s['skill'] != skill_obj['skill']]
                    
                    # 检查是否已存在于已掌握知识点中
                    skill_exists = False
                    for i, s in enumerate(accquired):
                        if s['skill'] == skill_obj['skill']:
                            # 更新已存在的知识点
                            accquired[i] = skill_obj
                            skill_exists = True
                            break
                    
                    # 如果不存在，则添加
                    if not skill_exists:
                        accquired.append(skill_obj)
                else:
                    # 如果用户档案不存在，创建新的
                    accquired = [skill_obj]
                    weak = []
                
                # 更新数据库
                upsert_query = """
                    INSERT INTO user_knowledge_profile 
                    (user_id, accquired_knowledges, weak_knowledges)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    accquired_knowledges = %s,
                    weak_knowledges = %s
                """
                cursor.execute(upsert_query, (
                    int(data_to_insert['user_id']), 
                    json.dumps(accquired, ensure_ascii=False), 
                    json.dumps(weak, ensure_ascii=False),
                    json.dumps(accquired, ensure_ascii=False), 
                    json.dumps(weak, ensure_ascii=False)
                ))
            else:  # 未掌握
                # 处理未掌握知识点
                if profile_result:
                    # 如果用户档案存在
                    accquired = profile_result[0] if profile_result[0] else []
                    weak = profile_result[1] if profile_result[1] else []
                    
                    # 从已掌握知识点中移除（如果存在）
                    accquired = [s for s in accquired if s['skill'] != skill_obj['skill']]
                    
                    # 检查是否已存在于未掌握知识点中
                    skill_exists = False
                    for i, s in enumerate(weak):
                        if s['skill'] == skill_obj['skill']:
                            # 更新已存在的知识点
                            weak[i] = skill_obj
                            skill_exists = True
                            break
                    
                    # 如果不存在，则添加
                    if not skill_exists:
                        weak.append(skill_obj)
                else:
                    # 如果用户档案不存在，创建新的
                    accquired = []
                    weak = [skill_obj]
                
                # 更新数据库
                upsert_query = """
                    INSERT INTO user_knowledge_profile 
                    (user_id, accquired_knowledges, weak_knowledges)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    accquired_knowledges = %s,
                    weak_knowledges = %s
                """
    
                cursor.execute(upsert_query, (
                    int(data_to_insert['user_id']), 
                    json.dumps(accquired, ensure_ascii=False), 
                    json.dumps(weak, ensure_ascii=False),
                    json.dumps(accquired, ensure_ascii=False), 
                    json.dumps(weak, ensure_ascii=False)
                ))
            
            conn.commit()
            
            
    except Exception as e:
        # print(f"❌ 更新用户知识档案失败: {e}")
        print("")
    
    
    
    # 插入数据
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO user_skill_mastery_llm 
            (user_id, path_id, skill_id, skill, template, inference, overall_mastery, master, correct)
            VALUES (%(user_id)s, %(path_id)s,%(skill_id)s, %(skill)s, %(template)s, %(inference)s, %(overall_mastery)s, %(master)s, %(correct)s)
            ON DUPLICATE KEY UPDATE
            path_id=%(path_id)s,
            inference=%(inference)s,
            overall_mastery = %(overall_mastery)s,
            master = %(master)s,
            correct =%(correct)s
            """
            cursor.execute(sql, data_to_insert)
            conn.commit()
            # print("✅ llm插入成功！")
            
            sql="""
                INSERT INTO final_user_skill_mastery 
                (user_id, path_id, skill_id, skill, overall_mastery, master)
                VALUES 
                (%(user_id)s, %(path_id)s, %(skill_id)s, %(skill)s, %(overall_mastery)s, %(master)s)
                ON DUPLICATE KEY UPDATE
                path_id = VALUES(path_id),
                overall_mastery = VALUES(overall_mastery),
                master = VALUES(master)
            """
            data_to_insert = {
                'user_id': df.loc[0]["user_id"],
                'path_id':path_id,
                'skill_id': df.loc[0]["skill_id"],
                'skill': df.loc[0]["skill"],
                'overall_mastery': overall_mastery,
                'master': master,
            }
            cursor.execute(sql, data_to_insert)
            # print("✅ final插入成功！")
        
    except Exception as e:
        print(f"❌ 插入失败: {e}")
    finally:
        conn.close()

    # print("数据处理完成并已存入新表。")
else:
    print("请求失败，状态码：", response.status_code)
    print("错误信息：", response.text)
    
    


