import os
import json
import pandas as pd
import ast
df = pd.read_csv("/home/lms/project/dify-backend/data/test_data/new_user_one_log/output3_new_user_one_log_user_knowledge_profile.csv")
# 定义一个函数，用于处理每个字段：去重并保留最后一个skill
def remove_duplicate_skills(skill_list_str):
    try:
        # 将字符串解析为列表
        skill_list = ast.literal_eval(skill_list_str)
    except (ValueError, SyntaxError):
        # 如果解析失败，返回原值或空列表
        print(f"无法解析: {skill_list_str}")
        return skill_list_str  # 或者返回 [] 或 None

    if not isinstance(skill_list, list):
        return skill_list_str

    # 使用字典来去重，key为skill名称，保留最后一个
    seen_skills = {}
    for item in skill_list:
        if isinstance(item, dict) and 'skill' in item:
            # 用 skill 名作为 key，覆盖之前的值，自然保留最后一个
            seen_skills[item['skill']] = item
        else:
            # 如果 item 不是有效字典，跳过或保留？
            continue

    # 提取去重后的列表（顺序按原列表中最后一次出现的顺序）
    # 为了保持原始顺序（最后一次出现的顺序），我们重新遍历原列表
    result = []
    added_skills = set()
    # 倒序遍历，确保最后出现的在前
    for item in reversed(skill_list):
        if isinstance(item, dict) and 'skill' in item:
            skill_name = item['skill']
            if skill_name not in added_skills:
                result.append(item)
                added_skills.add(skill_name)
    # 再倒回来，保证顺序是“最后出现”的顺序在原顺序中的位置
    result.reverse()
    return result
def file_process(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 应用函数到两列
    df['accquired_knowledges'] = df['accquired_knowledges'].apply(remove_duplicate_skills)
    df['weak_knowledges'] = df['weak_knowledges'].apply(remove_duplicate_skills)
    
    # 设置current_knowledge为acquired_knowledges或weak_knowledges的第一个skill
    def get_first_skill(row):
        acquired = row['accquired_knowledges']
        weak = row['weak_knowledges']
        # acquired=ast.literal_eval(acquired)
        # weak=ast.literal_eval(weak)
        # 尝试从acquired_knowledges获取第一个skill
        if isinstance(acquired, list) and len(acquired) > 0:
            first_item = acquired[0]
            if isinstance(first_item, dict) and 'skill' in first_item:
                return first_item['skill']
        # 如果acquired_knowledges中没有，则从weak_knowledges获取
        if isinstance(weak, list) and len(weak) > 0:
            first_item = weak[0]
            if isinstance(first_item, dict) and 'skill' in first_item:
                return first_item['skill']
        
        # 如果都没有，则返回默认值
        return ""
    
    df['current_knowledge'] = df.apply(get_first_skill, axis=1)
    df['query'] = df.apply(lambda row: f"我想学习{row['current_knowledge']},帮我规划一条学习路径", axis=1)
    df.to_csv(file_path, index=False)

file_process('/home/lms/project/dify-backend/data/test_data/user_middle_test_log/new_prompt4_user_middle_test_log_user_knowledge_profile.csv')
        