import pandas as pd
import requests
import json
import re
import time
import argparse
import tqdm
from json_repair import repair_json
from thefuzz import process,fuzz
import os
import ast
from typing import Dict, List, Tuple, Any, Optional

# Ollama API 地址
OLLAMA_URL = "http://localhost:11434/api/generate"

# 模型名称
MODEL_NAME = "MAGIC_final_deepseek_r1_7b:latest"
EVALUATOR_MODEL = "deepseek-r1:32b"

# 推理过程保存目录
REASONING_DIR = "reasoning_logs"
if not os.path.exists(REASONING_DIR):
    os.makedirs(REASONING_DIR)

skill2id={"微积分":"54","向量空间":"86","拓扑方法":"1","矩阵乘法":"13","坐标轴":"346","组合数":"277","素数":"279","几何学":"48","正实数":"280","初始相位":"312","切线":"325","向量投影":"371","对称性":"18","投影":"70","幂运算":"79","二项式定理":"276","平面图":"338","上界":"83","椭圆曲线":"310","圆锥":"311","分形":"22","未知数":"50","阶乘":"278","因式":"75","二次收敛":"305","数学物理":"61","位似变换":"340","分部积分法":"322","基底":"27","无限循环小数":"40","锥曲面与球面":"49","QR分解":"363","系数":"34","幂次":"42","函数复合":"67","平方根":"4","同构":"81","二次方程":"77","定义域":"82","标准差":"358","因式分解":"47","零点":"46","齐次函数":"296","参数方程":"324","四面体":"334","三角形":"336","开区间":"350","极值点":"309","向量内积":"378","孤点":"103","群论":"65","标量λ":"15","梯度":"326","内积":"366","几何-调和平均数":"356","弱分离公理":"95","微分方程":"25","函数":"51","正交群":"84","阶数":"26","进位制":"39","矩阵逆":"294","线性函数":"297","反函数":"301","共轭转置":"368","固定点":"21","指数函数":"295","多项式":"8","数的分割":"36","子广群":"80","求和符号":"287","角频率":"314","拉格朗日":"331","分数指数":"302","最佳多项式":"307","柯西":"333","凹凸性":"392","变换":"17","矩阵":"12","余弦函数":"11","向量":"16","一次函数":"74","多元实函数":"298","邻域":"106","无穷级数":"240","复数空间":"299","三维正交坐标系":"217","一阶导数":"308","闭开集":"166","二项式展开":"317","二项式系数的求和公式":"288","有向曲面":"523","极限集合":"303","数学集合理论":"230","连续函数的闭支撑":"354","整数加法":"35","收敛速度":"306","古典力学":"58","双曲线":"202","连续可导函数":"323","乘法":"78","不动点理论":"304","拓扑空间":"110","直角坐标系":"362","二阶导数":"316","判别式":"85","逆矩阵":"69","代数":"5","调和平均数":"360","正交矩阵":"365","容积":"393","单位根":"2","指数运算":"9","常数项":"10","可导性":"319","科学计数法":"339","单位圆":"361","平面几何":"391","数域包含关系":"231","模运算":"6","线性":"24","三角函数":"284","向量加法":"281","子群":"63","三线坐标":"282","乘积法则":"321","线性外代数":"32","共线":"283","正实数集合":"224","狭义相对论":"62","复数域":"222","圆锥坐标系":"204","级数的项":"238","圆球面":"209","鞍点":"172","几何体":"6868","实数":"10466","无穷小量":"925","极限":"504","等式":"10467","等比级数":"3996","无穷":"1543","有理数":"490","实数系":"10468","收敛定理":"425","闭区间":"5368","基数":"5298","拓扑学":"10469","位置数字系统":"10470","标准实数系统":"2585","小数展开式":"10471","数论":"464","10进数":"9950","小数点":"10472","算术":"4019","柯西序列":"2548","戴德金分割":"10473","不定式":"10474","无限小数":"1627","排列组合":"5348","空函数":"3951","集合论中的指数运算":"10475","幂次法则":"10476","零次幂":"10477","常数函数":"10478","零函数":"4983","零次函数":"10479","空集":"3954","水平线":"3492","一元运算":"10480","超越数":"4730","多项式函数":"10481","泰勒级数":"559","微分规则":"10482","逐项微分":"10483","级数":"10484","几何级数":"6916","切比雪夫多项式":"7252","首项系数":"5088","泰勒展开":"560","误差余项":"10485","e^x":"10486","复数":"5299","无理数":"575","代数数":"5207","映射(函数)":"1777","转置":"10487","集合":"4974","合成关系":"8647","几何代数":"6867","关系合成":"6737","整数集(Z)":"1307","子集":"5316","双曲几何":"8145","直线":"10488","互自切点(tacnode)":"5772","等边五边形":"10489","射线":"10490","凸五边形镶嵌":"6946","线段":"10491","法线":"557","矢":"10492","最值点":"1914","角平分线":"4497","圆":"5287","摆线":"1011","垂直平分线":"9407","同界角":"8745","五边形镶嵌":"5841","平行":"10493","四维柱体柱":"10494","旋转":"1450","可展曲面":"8450","球面几何":"4173","三维空间":"5160","二维空间":"5655","直纹曲面":"10495","一维空间":"10496","体积":"6120","长度":"5003","周长":"8925","对称轴":"10497","正方形":"3351","曲线":"581","立方体":"3962","垂直":"9402","克莱因瓶":"6458","相交":"10498","相切":"10499","相离":"10500","镜像":"10501","反演":"8296","表面积":"10502","挠率":"10503","角度":"4500","面积":"10504","离心率":"10505","三角函数表":"10506","正弦曲线":"3321","cis":"10507","三分之一角公式":"10508","三角函数恒等式":"10509","三角多项式":"10510","双曲三角函数":"8135","反三角函数":"5266","诱导公式":"10511","半正矢公式":"7714","三角函数精确值":"10512","高斯函数":"5098","三角函数积分表":"10513","双曲函数":"5270","三角函数线":"10514","有理函数":"491","无理函数":"677","正弦平方":"3319","割圆八线":"7393","余弦定理":"5214","圆心角":"9260","周期性":"8909","余函数恒等式":"6137","半正矢定理":"7715","精确值":"10515","余切":"6138","根号":"515","tan":"10516","近似作图":"10517","比例":"554","正弦":"3311","π":"10518","绝对误差":"10519","克莱姆法则":"6462","协方差矩阵":"7751","基":"5296","正交":"546","特征值":"5352","标量":"2624","向量子空间":"8819","特征向量":"5353","线性方程组":"4227","对偶空间":"10520","列空间":"7297","线性投影":"4222","矩阵中的项":"10521","双曲余弦函数":"8143","矩阵中的Q元素":"10522","偏导数":"5221","双曲正弦函数":"8161","Lax对":"10523","正弦函数":"544","复共轭矩阵":"9637","高维Lax对":"10524","本征值":"2311","李代数":"499","本征向量":"2312","费马小定理":"4606","幺正矩阵":"10525","极分解":"2417","方块矩阵":"10526","奇异值分解":"10527","二重向量":"5683","克利福德代数":"10528","幂零矩阵":"10529","LU分解":"10530","稀疏矩阵":"3936","行列式":"4419","格拉姆-施密特正交化":"2733","单位上三角矩阵":"7763","非奇异方阵":"10531","三角矩阵":"5154","伴随矩阵":"6040","反对称矩阵":"8263","可逆矩阵":"5277","埃尔米特矩阵":"9441","秩":"3910","外积":"10532","核":"2681","迹":"4938","线性空间":"4105","四元数":"9043","多重积分":"10533","左反函数":"10534","二阶可导的凸函数":"5691","一元函数":"5143","凸函数":"5245","二次函数":"5194","一元可微函数":"10535","二次可微函数":"5551","导数":"5329","凸集":"5246","绝对值函数":"4351","严格凸函数":"10536","半正定":"7711","极小值":"507","黑塞矩阵":"5056","函数限制":"7077","一对一函数":"10537","可测函数":"5275","光滑函数":"6404","复合函数":"9670","空间变换":"3958","非满射函数":"10538","方程式根":"1386","偏函数":"6260","恒等函数":"10539","连续函数":"4920"}

entity_list={
  "拓扑方法": "拓扑方法_entities_list.json",
  "单位根": "单位根_entities_list.json",
  "平方根": "平方根_entities_list.json",
  "代数": "代数_entities_list.json",
  "模运算": "模运算_entities_list.json",
  "多项式": "多项式_entities_list.json",
  "指数运算": "指数运算_entities_list.json",
  "常数项": "常数项_entities_list.json",
  "余弦函数": "余弦函数_entities_list.json",
  "矩阵": "矩阵_entities_list.json",
  "矩阵乘法": "矩阵乘法_entities_list.json",
  "标量λ": "标量λ_entities_list.json",
  "向量": "向量_entities_list.json",
  "变换": "变换_entities_list.json",
  "对称性": "对称性_entities_list.json",
  "固定点": "固定点_entities_list.json",
  "分形": "分形_entities_list.json",
  "线性": "线性_entities_list.json",
  "微分方程": "微分方程_entities_list.json",
  "阶数": "阶数_entities_list.json",
  "基底": "基底_entities_list.json",
  "线性外代数": "线性外代数_entities_list.json",
  "系数": "系数_entities_list.json",
  "整数加法": "整数加法_entities_list.json",
  "数的分割": "数的分割_entities_list.json",
  "进位制": "进位制_entities_list.json",
  "无限循环小数": "无限循环小数_entities_list.json",
  "零点": "零点_entities_list.json",
  "因式分解": "因式分解_entities_list.json",
  "几何学": "几何学_entities_list.json",
  "锥曲面与球面": "锥曲面与球面_entities_list.json",
  "未知数": "未知数_entities_list.json",
  "函数": "函数_entities_list.json",
  "微积分": "微积分_entities_list.json",
  "古典力学": "古典力学_entities_list.json",
  "数学物理": "数学物理_entities_list.json",
  "狭义相对论": "狭义相对论_entities_list.json",
  "子群": "子群_entities_list.json",
  "群论": "群论_entities_list.json",
  "函数复合": "函数复合_entities_list.json",
  "逆矩阵": "逆矩阵_entities_list.json",
  "投影": "投影_entities_list.json",
  "一次函数": "一次函数_entities_list.json",
  "因式": "因式_entities_list.json",
  "二次方程": "二次方程_entities_list.json",
  "乘法": "乘法_entities_list.json",
  "幂运算": "幂运算_entities_list.json",
  "子广群": "子广群_entities_list.json",
  "同构": "同构_entities_list.json",
  "定义域": "定义域_entities_list.json",
  "上界": "上界_entities_list.json",
  "正交群": "正交群_entities_list.json",
  "判别式": "判别式_entities_list.json",
  "向量空间": "向量空间_entities_list.json",
  "弱分离公理": "弱分离公理_entities_list.json",
  "孤点": "孤点_entities_list.json",
  "邻域": "邻域_entities_list.json",
  "拓扑空间": "拓扑空间_entities_list.json",
  "闭开集": "闭开集_entities_list.json",
  "鞍点": "鞍点_entities_list.json",
  "双曲线": "双曲线_entities_list.json",
  "圆锥坐标系": "圆锥坐标系_entities_list.json",
  "圆球面": "圆球面_entities_list.json",
  "三维正交坐标系": "三维正交坐标系_entities_list.json",
  "复数域": "复数域_entities_list.json",
  "正实数集合": "正实数集合_entities_list.json",
  "数学集合理论": "数学集合理论_entities_list.json",
  "数域包含关系": "数域包含关系_entities_list.json",
  "级数的项": "级数的项_entities_list.json",
  "无穷级数": "无穷级数_entities_list.json",
  "二项式定理": "二项式定理_entities_list.json",
  "组合数": "组合数_entities_list.json",
  "阶乘": "阶乘_entities_list.json",
  "素数": "素数_entities_list.json",
  "正实数": "正实数_entities_list.json",
  "向量加法": "向量加法_entities_list.json",
  "三线坐标": "三线坐标_entities_list.json",
  "共线": "共线_entities_list.json",
  "三角函数": "三角函数_entities_list.json",
  "求和符号": "求和符号_entities_list.json",
  "二项式系数的求和公式": "二项式系数的求和公式_entities_list.json",
  "幂次": "幂次_entities_list.json",
  "矩阵逆": "矩阵逆_entities_list.json",
  "指数函数": "指数函数_entities_list.json",
  "齐次函数": "齐次函数_entities_list.json",
  "线性函数": "线性函数_entities_list.json",
  "多元实函数": "多元实函数_entities_list.json",
  "复数空间": "复数空间_entities_list.json",
  "反函数": "反函数_entities_list.json",
  "分数指数": "分数指数_entities_list.json",
  "极限集合": "极限集合_entities_list.json",
  "不动点理论": "不动点理论_entities_list.json",
  "二次收敛": "二次收敛_entities_list.json",
  "收敛速度": "收敛速度_entities_list.json",
  "最佳多项式": "最佳多项式_entities_list.json",
  "一阶导数": "一阶导数_entities_list.json",
  "极值点": "极值点_entities_list.json",
  "椭圆曲线": "椭圆曲线_entities_list.json",
  "圆锥": "圆锥_entities_list.json",
  "初始相位": "初始相位_entities_list.json",
  "角频率": "角频率_entities_list.json",
  "二阶导数": "二阶导数_entities_list.json",
  "二项式展开": "二项式展开_entities_list.json",
  "可导性": "可导性_entities_list.json",
  "乘积法则": "乘积法则_entities_list.json",
  "分部积分法": "分部积分法_entities_list.json",
  "连续可导函数": "连续可导函数_entities_list.json",
  "参数方程": "参数方程_entities_list.json",
  "切线": "切线_entities_list.json",
  "梯度": "梯度_entities_list.json",
  "拉格朗日": "拉格朗日_entities_list.json",
  "柯西": "柯西_entities_list.json",
  "四面体": "四面体_entities_list.json",
  "三角形": "三角形_entities_list.json",
  "平面图": "平面图_entities_list.json",
  "科学计数法": "科学计数法_entities_list.json",
  "位似变换": "位似变换_entities_list.json",
  "坐标轴": "坐标轴_entities_list.json",
  "开区间": "开区间_entities_list.json",
  "连续函数的闭支撑": "连续函数的闭支撑_entities_list.json",
  "几何-调和平均数": "几何-调和平均数_entities_list.json",
  "标准差": "标准差_entities_list.json",
  "调和平均数": "调和平均数_entities_list.json",
  "单位圆": "单位圆_entities_list.json",
  "直角坐标系": "直角坐标系_entities_list.json",
  "QR分解": "QR分解_entities_list.json",
  "正交矩阵": "正交矩阵_entities_list.json",
  "内积": "内积_entities_list.json",
  "共轭转置": "共轭转置_entities_list.json",
  "向量投影": "向量投影_entities_list.json",
  "向量内积": "向量内积_entities_list.json",
  "平面几何": "平面几何_entities_list.json",
  "凹凸性": "凹凸性_entities_list.json",
  "容积": "容积_entities_list.json",
  "有向曲面": "有向曲面_entities_list.json",
  "几何体": "几何体_entities_list.json"
}

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
    # 再倒回来，保证顺序是"最后出现"的顺序在原顺序中的位置
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

def extract_json_objects(text):
    json_objects = []

    # 1. 提取 JSON 片段
    matches = re.findall(r'```json\s*([\s\S]*?)```', text, re.DOTALL)
    if len(matches) == 0:
        repaired = repair_json(text)
        parsed = json.loads(repaired)  # 转为 Python 对象
        json_objects.append(parsed)
        return json_objects

    # 2. 修复并解析每个 JSON 片段
    for match in matches:
        try:
            # 使用 json_repair 修复 JSON 字符串
            repaired = repair_json(match.strip())
            parsed = json.loads(repaired)  # 转为 Python 对象
            json_objects.append(parsed)
        except json.JSONDecodeError as e:
            print(f"修复后仍解析失败: {e}")
            print("原始内容:\n", match)
            print("修复后内容:\n", repaired)
        except Exception as ex:
            print(f"未知错误: {ex}")

    return json_objects[0]

def call_llm(prompt, model=MODEL_NAME, save_reasoning=False, reasoning_id=None):
    """
    调用大语言模型
    """
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            full_response = result.get("response", "")
            
            # 保存推理过程
            if save_reasoning and reasoning_id:
                reasoning_file = os.path.join(REASONING_DIR, f"{reasoning_id}_reasoning.txt")
                with open(reasoning_file, 'w', encoding='utf-8') as f:
                    f.write(f"Model: {model}\n")
                    f.write(f"Prompt:\n{prompt}\n")
                    f.write(f"Full Response:\n{full_response}\n")
                    f.write("="*80 + "\n")
            
            # 去除<think>标签内容
            cleaned_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
            return cleaned_response
        else:
            print(f"模型调用失败，状态码: {response.status_code}")
            return None
    except Exception as e:
        print(f"模型调用异常: {e}")
        return None

def evaluate_reasoning_quality(reasoning_id, user_query, generated_path):
    """
    评估推理合理性
    """
    reasoning_file = os.path.join(REASONING_DIR, f"{reasoning_id}_reasoning.txt")
    
    if not os.path.exists(reasoning_file):
        print(f"推理文件不存在: {reasoning_file}")
        return 0.5  # 默认分数
    
    # 读取推理过程
    with open(reasoning_file, 'r', encoding='utf-8') as f:
        reasoning_content = f.read()
    
    # 构建评估prompt
    evaluation_prompt = f"""
你是一个专业的学习路径评估专家，需要对AI模型生成学习路径时的推理过程进行评分。

用户查询：{user_query}
生成的学习路径：{generated_path}

以下是AI模型的完整推理过程：
{reasoning_content}

请从以下维度评估推理质量：
1. 逻辑一致性：推理过程是否逻辑清晰、前后一致
2. 专业准确性：对知识点的理解是否准确，依赖关系是否正确
3. 个性化适配：是否充分考虑用户的具体需求和背景
4. 路径合理性：生成的路径是否符合学习规律和认知科学原理
5. 完整性：推理过程是否全面，考虑因素是否充分

请给出0-1之间的分数（保留2位小数），其中：
- 0.9-1.0：推理极其优秀，逻辑严密，专业准确
- 0.8-0.9：推理优秀，逻辑清晰，基本准确
- 0.7-0.8：推理良好，有一定逻辑性
- 0.6-0.7：推理一般，存在一些问题
- 0.5-0.6：推理较差，逻辑不清或错误较多
- 0.0-0.5：推理很差，严重缺乏逻辑或专业性

请只返回分数，格式如：0.85
"""
    
    try:
        response = call_llm(evaluation_prompt, model=EVALUATOR_MODEL)
        if response:
            # 提取分数
            score_match = re.search(r'(\d+\.?\d*)', response.strip())
            if score_match:
                score = float(score_match.group(1))
                # 确保分数在0-1范围内
                score = max(0.0, min(1.0, score))
                print(f"推理质量评分: {score}")
                return score
            else:
                print(f"无法解析评分结果: {response}")
                return 0.5
        else:
            print("评估模型调用失败")
            return 0.5
    except Exception as e:
        print(f"推理质量评估失败: {e}")
        return 0.5

def user_profile_analysis(user_query, history):
    """
    用户画像分析
    """
    prompt = f"""
你是一位智能分析助手，任务是根据用户输入和历史数据库信息，提取其学习需求与用户偏好。

- 当前用户输入（必须作为主要依据）：
{user_query}
- 用户历史偏好（仅可用于背景补充，不可主导判断）：
{history}

**你的任务：**
你负责提取用户信息，完成以下任务：
1. 分析从数据库加载的用户历史偏好
2. 提取当前问题的核心需求
3. 严格按照以下 JSON 格式输出结果，不要任何额外文本或说明。

请完成以下四项任务，并**严格以标准 JSON 格式返回，不要输出任何额外说明、注释或标点**：

1. **提取用户当前最核心的学习目标**、该信息**必须仅依据用户输入**得出；
2. **结合用户输入与数据库内容**分析用户知识背景，即先将将知识点归类到数学领域类型，再根据掌握度判断类型强弱，生成知识画像，描述用户在不同数学领域的掌握程度，指出可较差应用的强项领域和重点需要补充的弱项领域，禁止出现具体的知识点名称和掌握度数值；
3. **判断用户当前最合适的学习资源类型**，从以下选项中按需选择一项：
 - 视频资源
 - 课件资源
 - 论文资源
 - 练习资源
4. **判断用户的学习能力**，从用户掌握的知识点数和薄弱知识点数进行评估，从以下选项中按需选择一项：
 - 基础
 - 进阶
5. **判断用户的时间偏好**，从以下选项中按需选择一项：
        "速成"：几天到2周
        "短期"：2周-1个月
        "中期"：1-3个月
        "长期"：3个月以上
        "未指定"

**输出格式（仅限以下 JSON，不得有其他输出）：**
{{
"learning": "视频资源|课件资源|论文资源|练习资源" 中的一项，根据用户当前需求判断",
"profile": "包含要素为用户知识背景、用户偏好、兴趣主题等，均用一段文字描述",
"goal": "包含要素为用户的目标水平、应用场景"，
"time":"包含要素为用户的时间偏好"，
"ability":"基础或者进阶",
"question": "用户当前输入中明确表达的核心学习目标或问题"
}}

特别要求：
question 字段必须精准反映用户输入的目标，不受数据库内容影响；
profile 字段应体现用户的整体学习偏好、风格与知识背景等（可参考数据库），需要考虑强项和弱项；
goal字段应体现用户的目标水平、应用场景
time字段只能从指定的五项中选择，不可添加或改写；
ability字段只能从指定的两项中选择，不可添加或改写；
learning 字段只能从指定四项中选择，不可添加或改写；
最终输出必须是纯 JSON，无任何解释性语言。

请根据用户输入提取关键信息，并以标准 JSON 格式返回结果。
    """
    
    response = call_llm(prompt)
    return response
def path_planning(profile_analysis,reasoning_id=None):
    """
    路径规划
    """
    prompt=f"""
        你是一个学习路径规划专家，基于用户个人背景分析（{profile_analysis}）和具体需求，生成符合认知逻辑的个性化学习路径。请遵循以下原则：

        #### 核心原则
        1. **粒度控制**  
        - 路径长度：仅包含 **4-7个宏观知识模块**（每个模块为章节级内容，学习时间≥3小时）  
        - 模块要求：整合同一领域的相关概念，禁止碎片化（如将"函数基础"而非"正实数"作为模块）  
        - 宏观知识块优先：以章节级别的知识模块为单位，而非具体的小概念
        - 路径长度限制：整个学习路径应控制在4-7个主要知识块内
        - 避免过度细分：不要将同一知识领域的细节概念拆分成多个步骤
        - 示例：  
            ❌ 错误路径：正实数 → 反函数 → 乘法 → 除法...  
            ✅ 正确路径：函数基础 → 极限 → 定积分 → 不定积分 → 微分方程 

        2. **图谱结构与认知逻辑** 
        路径需分层递进，顺序严格遵循：
            1）前置知识：描述当前知识点的基本概念
            2）包含内容：描述当前知识点的核心概念和子知识点
            3）所属领域：描述知识点在数学体系中的定位及其与其他分支的关联
            4）相关概念：当前知识点和其他数学概念的关联
            5）后置应用：描述的是当前知识点的实际应用或者在高阶理论中的应用
        顺序：前置知识 -> 包含内容 -> 所属领域 -> 相关概念 -> 后置应用

        #### 路径生成步骤
        1. **领域定位**：识别对应的学科领域及基础框架  
        2. **逆向分析**：  
        - 上位概念 → 目标概念 → 下位概念  
        - 必需的前置知识模块（最多保留2个关键前置）  
        3. **分层压缩**：将认知层次（基础/构成/定义/应用）压缩为4-7个宏观模块  
        4. **逻辑校验**：  
        - 检查模块间依赖关系是否闭环  
        - 删除与目标无关的冗余模块  
        - 始终以用户的学习目标为中心
        - 前置知识筛选：只包含对达成学习目标真正必要的前置知识领域
        - 依赖关系检查：确保没有知识模块在其前置要求之前出现
        - 逻辑验证：生成路径后进行自检，确保学习顺序符合知识体系逻辑
        - 个性化调整：基于用户的薄弱知识点、学习目标以及学习兴趣进行针对性调整
        """
    p2="""
        #### 输出要求
        返回严格JSON格式，`plan` 字段为字符串数组,不得输出其他内容：
        json

        {

        "plan": ["宏观模块1", "宏观模块2", "宏观模块3"] // 4-7个模块

        }
    
    """
    
    response = call_llm(prompt+p2, save_reasoning=True, reasoning_id=reasoning_id)
    return response

def plan_review(plan,profile):
    """
    路径规划评审
    """
    prompt = f"""
        你是严格的质量评审员，负责评审为用户生成的学习计划。请根据以下要求进行评审，并**确保输出为标准的 JSON 格式**，不要任何额外文本。
    用户的学习计划为：
    {plan}
    评审内容：
    1. 评估学习计划是否匹配用户 profile：{profile}。`
    2. 检查时间安排是否合理。
    3. 确保知识递进关系正确。
    4. 确保学习路径中知识点的数量在4-7之间

    评审要点：
    - 是否符合用户时间投入？
    - 是否弥补用户薄弱环节？
    - 是否有明确里程碑？

    输出格式必须如下：\n
    """
    p2="""输出要求：
    {
    "status": "APPROVED 或 REJECTED",
    "message": "简要说明评审结论",
    "suggestions": ["建议1", "建议2", ...]
    }

    请严格按照上述 JSON 格式输出评审结果，不使用任何 Markdown 或其他格式。
    """
    
    response = call_llm(prompt+p2)
    return response


def plan_repeat(plan,critic_message):
    prompt=f"""
        这是路径规划师指定的学习路径：{plan}
        - 其中的plan是知识点的学习顺序，detail_plan字段涵盖了每个知识点的具体学习情况，现在不符合学生要要求
        - 请根据质量评审员提出的修改建议重新规划，修改建议为:{critic_message}
        - 仔细阅读质量评审员的反馈
        - 修改计划以解决指出的问题
        #### 核心原则
        1. **粒度控制**  
        - 路径长度：仅包含 **4-7个宏观知识模块**（每个模块为章节级内容，学习时间≥3小时）  
        - 模块要求：整合同一领域的相关概念，禁止碎片化（如将"函数基础"而非"正实数"作为模块）  
        - 宏观知识块优先：以章节级别的知识模块为单位，而非具体的小概念
        - 路径长度限制：整个学习路径应控制在4-7个主要知识块内
        - 避免过度细分：不要将同一知识领域的细节概念拆分成多个步骤
        - 示例：  
            ❌ 错误路径：正实数 → 反函数 → 乘法 → 除法...  
            ✅ 正确路径：函数基础 → 极限 → 定积分 → 不定积分 → 微分方程 

        2. **图谱结构与认知逻辑** 
        路径需分层递进，顺序严格遵循：
            1）前置知识：描述当前知识点的基本概念
            2）包含内容：描述当前知识点的核心概念和子知识点
            3）所属领域：描述知识点在数学体系中的定位及其与其他分支的关联
            4）相关概念：当前知识点和其他数学概念的关联
            5）后置应用：描述的是当前知识点的实际应用或者在高阶理论中的应用
        顺序：前置知识 -> 包含内容 -> 所属领域 -> 相关概念 -> 后置应用

        #### 路径生成步骤
        1. **领域定位**：识别对应的学科领域及基础框架  
        2. **逆向分析**：  
        - 上位概念 → 目标概念 → 下位概念  
        - 必需的前置知识模块（最多保留2个关键前置）  
        3. **分层压缩**：将认知层次（基础/构成/定义/应用）压缩为4-7个宏观模块  
        4. **逻辑校验**：  
        - 检查模块间依赖关系是否闭环  
        - 删除与目标无关的冗余模块  
        - 始终以用户的学习目标为中心
        - 前置知识筛选：只包含对达成学习目标真正必要的前置知识领域
        - 依赖关系检查：确保没有知识模块在其前置要求之前出现
        - 逻辑验证：生成路径后进行自检，确保学习顺序符合知识体系逻辑
        - 个性化调整：基于用户的薄弱知识点、学习目标以及学习兴趣进行针对性调整
        ## 输出格式要求：\n
            
        """
    p2="""请严格返回以下 JSON 格式的内容，不要添加任何额外说明或解释：
            {
            "plan": ["知识点1", "知识点2", ...],// 4-7 个知识点
            }
    
    """
    response = call_llm(prompt+p2)
    return response

class KnowledgePathFinder:
    def __init__(self, json_file_path: str):
        """初始化知识图谱路径查找器"""
        self.graph = self.load_knowledge_graph(json_file_path)
    
    def load_knowledge_graph(self, file_path: str) -> Dict:
        """加载知识图谱JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"文件 {file_path} 不存在")
            return {}
    
    def find_node_location(self, graph: Dict, target_node: str, current_path: List[str] = None) -> List[Tuple[List[str], Dict]]:
        """找到节点在图中的位置，返回(到达该节点的路径, 该节点的子图)"""
        if current_path is None:
            current_path = []
        
        results = []
        
        for node, children in graph.items():
            new_path = current_path + [node]
            
            if node == target_node:
                subgraph = children if isinstance(children, dict) else {}
                results.append((new_path, subgraph))
            
            if isinstance(children, dict) and children:
                sub_results = self.find_node_location(children, target_node, new_path)
                results.extend(sub_results)
        
        return results
    
    def can_reach_target_from_learned_path(self, target_node: str, learned_path: List[str]) -> Tuple[bool, List[str]]:
        """检查是否可以从已学习路径到达目标节点，返回(是否可达, 中间路径)"""
        target_locations = self.find_node_location(self.graph, target_node)
        
        for target_path, _ in target_locations:
            # 检查是否存在一条从已学习路径到目标节点的路径
            for learned_node in learned_path:
                learned_locations = self.find_node_location(self.graph, learned_node)
                
                for learned_path_full, _ in learned_locations:
                    # 检查target_path是否可以从learned_path_full到达
                    if self._can_path_reach_target(learned_path_full, target_path):
                        # 提取从learned_node到target_node的路径
                        intermediate_path = target_path[len(learned_path_full):]
                        return True, intermediate_path
        
        return False, []
    
    def _can_path_reach_target(self, learned_path_full: List[str], target_path: List[str]) -> bool:
        """检查学习路径是否可以到达目标路径"""
        if len(learned_path_full) >= len(target_path):
            return False
        
        # 检查learned_path_full是否是target_path的前缀
        for i, node in enumerate(learned_path_full):
            if i >= len(target_path) or target_path[i] != node:
                return False
        
        return True
    
    def get_siblings_and_children(self, node_path: List[str]) -> Tuple[List[str], List[str]]:
        """获取节点的同级节点和子节点"""
        if not node_path:
            return [], []
        
        # 获取同级节点（父节点的所有子节点，除了当前节点）
        siblings = []
        if len(node_path) > 1:
            parent_path = node_path[:-1]
            current_node = node_path[-1]
            
            # 导航到父节点
            current = self.graph
            for node in parent_path:
                if isinstance(current, dict) and node in current:
                    current = current[node]
                else:
                    current = {}
                    break
            
            if isinstance(current, dict):
                siblings = [node for node in current.keys() if node != current_node]
        
        # 获取子节点
        children = []
        current = self.graph
        for node in node_path:
            if isinstance(current, dict) and node in current:
                current = current[node]
            else:
                current = {}
                break
        
        if isinstance(current, dict):
            children = list(current.keys())
        
        return siblings, children
    
    def check_node_exists(self, node: str) -> bool:
        """检查节点是否在知识图谱中存在"""
        def search_recursive(graph: Dict) -> bool:
            for n, children in graph.items():
                if n == node:
                    return True
                if isinstance(children, dict) and search_recursive(children):
                    return True
            return False
        
        return search_recursive(self.graph)
    
    def find_best_path(self, start_node: str, target_sequence: List[str]) -> Tuple[List[str], float, int]:
        """找到包含最多目标节点的路径并计算匹配度"""
        
        # 规则1: 过滤出存在的目标节点
        valid_targets = [node for node in target_sequence if self.check_node_exists(node)]
        print(f"原始目标序列: {target_sequence}")
        print(f"有效目标序列: {valid_targets}")
        
        if not valid_targets:
            return [], 0.0, 0
        
        # 如果起始节点不存在，使用第一个有效节点
        if not self.check_node_exists(start_node):
            print(f"起始节点 {start_node} 不存在，使用第一个有效节点")
            start_node = valid_targets[0]
        
        # 开始构建学习路径
        learned_path = [start_node]
        remaining_targets = [node for node in valid_targets if node != start_node]
        
        print(f"起始学习路径: {learned_path}")
        
        # 当前节点在图中的位置
        current_node_locations = self.find_node_location(self.graph, start_node)
        if not current_node_locations:
            return learned_path, 0.0, 0
        
        # 使用第一个找到的位置
        current_path, current_subgraph = current_node_locations[0]
        print(f"当前节点 '{start_node}' 的位置: {' -> '.join(current_path)}")
        
        # 继续寻找其他目标节点
        while remaining_targets:
            found_any = False
            best_target = None
            best_new_path = None
            
            # 按照目标序列的顺序寻找下一个节点
            next_target = remaining_targets[0]  # 总是尝试列表中的第一个目标节点
            
            print(f"\n尝试按顺序寻找目标节点 '{next_target}':")
            
            # 防止重复添加已经在路径中的节点
            if next_target in learned_path:
                print(f"  节点 '{next_target}' 已在学习路径中，跳过")
                remaining_targets.remove(next_target)
                continue
            
            # 从已学习路径的每个节点开始检查可访问性
            for i, learned_node in enumerate(learned_path):
                learned_node_locations = self.find_node_location(self.graph, learned_node)
                
                for learned_path_full, learned_subgraph in learned_node_locations:
                    # 获取从这个已学习节点可访问的节点
                    siblings, children = self.get_siblings_and_children(learned_path_full)
                    accessible_nodes = siblings + children
                    
                    if next_target in accessible_nodes:
                        print(f"  可从 '{learned_node}' 直接访问（位置: {' -> '.join(learned_path_full)}）")
                        print(f"    同级节点: {siblings}")
                        print(f"    子节点: {children}")
                        
                        best_target = next_target
                        
                        # 找到目标的正确位置
                        target_locations = self.find_node_location(self.graph, next_target)
                        if target_locations:
                            for target_path_full, target_subgraph in target_locations:
                                if (next_target in siblings and len(target_path_full) == len(learned_path_full)) or \
                                   (next_target in children and len(target_path_full) == len(learned_path_full) + 1):
                                    best_new_path = target_path_full
                                    break
                        
                        found_any = True
                        break
                
                if found_any:
                    break
            
            if found_any and best_target and best_target not in learned_path:
                learned_path.append(best_target)
                remaining_targets.remove(best_target)
                
                # 更新当前位置
                if best_new_path:
                    current_path = best_new_path
                else:
                    # 找到目标节点的位置
                    target_locations = self.find_node_location(self.graph, best_target)
                    if target_locations:
                        current_path = target_locations[0][0]
                
                print(f"成功添加节点 '{best_target}'，当前路径: {learned_path}")
                print(f"更新位置到: {' -> '.join(current_path)}")
                
                # 重置变量为下一轮
                best_target = None
                best_new_path = None
            elif not found_any:
                # 如果当前目标节点无法直接访问，尝试通过中间节点
                print(f"  无法直接访问 '{next_target}'，尝试寻找中间节点...")
                
                target_locations = self.find_node_location(self.graph, next_target)
                intermediate_found = False
                
                for target_path, _ in target_locations:
                    # 检查是否可以通过已学习的节点到达
                    for learned_node in learned_path:
                        learned_locations = self.find_node_location(self.graph, learned_node)
                        
                        for learned_path_full, _ in learned_locations:
                            # 检查target_path是否以learned_path_full为前缀
                            if len(target_path) > len(learned_path_full) and \
                               target_path[:len(learned_path_full)] == learned_path_full:
                                
                                # 找到需要的中间节点
                                next_node = target_path[len(learned_path_full)]
                                
                                if next_node not in learned_path:
                                    print(f"    需要先添加中间节点 '{next_node}' 来到达 '{next_target}'")
                                    print(f"    路径: {' -> '.join(learned_path_full)} -> {next_node} -> ... -> {next_target}")
                                    
                                    # 检查中间节点是否可以直接访问
                                    siblings, children = self.get_siblings_and_children(learned_path_full)
                                    if next_node in siblings + children:
                                        best_target = next_node
                                        intermediate_found = True
                                        print(f"    中间节点 '{next_node}' 可以直接访问")
                                        break
                        
                        if intermediate_found:
                            break
                    
                    if intermediate_found:
                        break
                
                # 如果找到了中间节点，添加它
                if intermediate_found and best_target and best_target not in learned_path:
                    learned_path.append(best_target)
                    
                    # 更新当前位置
                    target_locations = self.find_node_location(self.graph, best_target)
                    if target_locations:
                        current_path = target_locations[0][0]
                    
                    print(f"成功添加中间节点 '{best_target}'，当前路径: {learned_path}")
                    print(f"更新位置到: {' -> '.join(current_path)}")
                    
                    # 重置变量，但不移除next_target，下一轮继续尝试
                    best_target = None
                    best_new_path = None
                else:
                    # 无法找到路径到达当前目标节点，跳过它
                    print(f"  无法找到路径到达 '{next_target}'，跳过")
                    remaining_targets.remove(next_target)
                    
                    if not remaining_targets:
                        break
        
        # 使用path_matching中的算法计算正确匹配数
        correct_count = self._calculate_correct_matches(learned_path, valid_targets)
        
        return learned_path, correct_count, len(learned_path)
    
    def _calculate_correct_matches(self, learned_path: List[str], valid_targets: List[str]) -> int:
        """修复的算法：按父节点判断顺序正确性 + 模糊匹配，更严格的评估标准"""
        
        def match_node(node: str, candidates: List[str], threshold: int = 80):
            """在候选列表中找到最相似的节点（≥阈值）"""
            best_match = None
            best_score = 0
            for c in candidates:
                score = fuzz.ratio(node, c)
                if score > best_score:
                    best_score = score
                    best_match = c
            return best_match if best_score >= threshold else None
        
        print(f"\n=== 正确性计算（修复版，模糊匹配阈值 80%） ===")
        print(f"算法找到的路径: {learned_path}")
        print(f"目标序列: {valid_targets}")
        
        # 用模糊匹配找出算法路径中对应的目标节点
        algorithm_target_nodes = []
        for node in learned_path:
            match = match_node(node, valid_targets, threshold=80)
            if match:
                algorithm_target_nodes.append(match)
        
        print(f"算法路径中的目标节点(按算法顺序): {algorithm_target_nodes}")
        
        correct_count = 0
        
        for node in algorithm_target_nodes:
            idx = valid_targets.index(node)
            if idx == 0:
                correct_count += 1
                print(f"  '{node}'（起始节点）顺序正确 ✓")
                continue
            
            parent_node = valid_targets[idx - 1]
            parent_match = match_node(parent_node, algorithm_target_nodes, threshold=80)
            
            if parent_match:
                if algorithm_target_nodes.index(parent_match) < algorithm_target_nodes.index(node):
                    correct_count += 1
                    print(f"  '{node}' 顺序正确（父节点 '{parent_node}' 在前）✓")
                else:
                    print(f"  '{node}' 顺序错误（父节点 '{parent_node}' 没在它前面）✗")
            else:
                # 修复：父节点不在路径中，这是一个问题，不算正确
                print(f"  '{node}' 顺序错误（父节点 '{parent_node}' 不在路径中）✗")
        
        print(f"正确匹配数: {correct_count}")
        return correct_count

def calculate_comprehensive_match_rate(learned_path, target_sequence, reasoning_score,
                                     order_weight=0.6, hit_weight=0.2, reasoning_weight=0.2):
    """
    计算综合匹配度：路径顺序 + 节点命中数 + 推理合理性
    修复了算法逻辑，使用更严格的顺序评估
    返回: (综合分数, 顺序分数, 命中分数, 推理分数)
    """
    if not target_sequence:
        return 0.0, 0.0, 0.0, reasoning_score
    
    # 模糊匹配找出命中的节点
    def match_node(node: str, candidates: List[str], threshold: int = 80):
        best_match = None
        best_score = 0
        for c in candidates:
            score = fuzz.ratio(node, c)
            if score > best_score:
                best_score = score
                best_match = c
        return best_match if best_score >= threshold else None
    
    # 计算节点命中数和命中分数
    hit_nodes = []
    for node in learned_path:
        match = match_node(node, target_sequence, threshold=80)
        if match and match not in hit_nodes:
            hit_nodes.append(match)
    
    hit_count = len(hit_nodes)
    # 节点命中分数 = 命中数 / 目标序列总数
    hit_score = hit_count / len(target_sequence) if len(target_sequence) > 0 else 0.0
    
    # 修复的顺序正确性算法
    correct_count = 0
    total_evaluable = 0  # 可评估的节点数量
    
    print(f"\n=== 顺序正确性详细分析 ===")
    print(f"命中的节点序列: {hit_nodes}")
    print(f"目标序列: {target_sequence}")
    
    for node in hit_nodes:
        idx = target_sequence.index(node)
        total_evaluable += 1
        
        if idx == 0:
            # 第一个节点总是正确的
            correct_count += 1
            print(f"  '{node}' (第1个节点): 正确 ✓")
        else:
            # 检查前置节点是否在路径中且位置正确
            parent_node = target_sequence[idx - 1]
            parent_match = match_node(parent_node, hit_nodes, threshold=80)
            
            if parent_match:
                if hit_nodes.index(parent_match) < hit_nodes.index(node):
                    correct_count += 1
                    print(f"  '{node}': 正确 ✓ (前置节点 '{parent_node}' 在正确位置)")
                else:
                    print(f"  '{node}': 错误 ✗ (前置节点 '{parent_node}' 位置错误)")
            else:
                # 修复：前置节点不在路径中，这是一个问题，不算正确
                print(f"  '{node}': 错误 ✗ (前置节点 '{parent_node}' 缺失)")
    
    order_score = correct_count / total_evaluable if total_evaluable > 0 else 0.0
    
    # 综合匹配度计算
    comprehensive_score = (order_score * order_weight +
                          hit_score * hit_weight +
                          reasoning_score * reasoning_weight)
    
    print(f"\n=== 综合匹配度计算 ===")
    print(f"顺序正确数: {correct_count}/{total_evaluable}")
    print(f"路径顺序分数: {order_score:.3f} (权重: {order_weight})")
    print(f"节点命中分数: {hit_score:.3f} (命中{hit_count}/{len(target_sequence)}) (权重: {hit_weight})")
    print(f"推理质量分数: {reasoning_score:.3f} (权重: {reasoning_weight})")
    print(f"综合匹配度: {comprehensive_score:.3f}")
    print(f"匹配成功: {'是' if comprehensive_score >= 0.7 else '否'} (阈值: 0.7)")
    
    return comprehensive_score, order_score, hit_score, reasoning_score

def process_user_learning_path(row, kg_path):
    """
    处理单个用户的学习路径规划
    """
    user_id = row.get('user_id', 'unknown')
    reasoning_id = f"user_{user_id}_{int(time.time())}"
    
    accquired_skills = []
    weak_skills = []
    if "accquired_knowledges" in row:
        acquired = ast.literal_eval(row["accquired_knowledges"])
        accquired_skills = [
            f"{item['skill']}，掌握度为{item['overall_mastery']:.2f}"
            for item in acquired
        ]
        weaks = ast.literal_eval(row["weak_knowledges"])
        weak_skills = [
            f"{item['skill']}，掌握度为{item['overall_mastery']:.2f}"
            for item in weaks
        ]

    accquired_skills_str = '、'.join(accquired_skills)
    weak_skills_str = '、'.join(weak_skills)
    preference_str = row["preference"] if "preference" in row else ""
    query = row["query"]
    profile_input = f'用户已掌握的知识点的情况为${accquired_skills_str}，掌握较弱的知识点的情况为{weak_skills_str}，以及用户的学习偏好为{preference_str}'
    
    # 1. 用户画像分析
    print(f"正在分析用户 {user_id} 的画像...")
    profile_analysis = user_profile_analysis(query, profile_input)
    profile_analysis_json = extract_json_objects(profile_analysis)
    final_profile = ""
    try:
        final_profile = f'用户的偏好为：{profile_analysis_json["profile"]},用户的学习能力是：{profile_analysis_json["ability"]}，用户想达到的掌握程度是{profile_analysis_json["goal"]},用户的学习时间要求为：{profile_analysis_json["time"]}，用户想要学习的知识是：{profile_analysis_json["question"]}'
    except:
        final_profile = profile_input
    
    print(final_profile)
    if not profile_analysis:
        return {"error": "用户画像分析失败"}
    
    # 2. 路径规划与评审循环
    max_iterations = 1
    iteration = 0
    approved = False
    final_plan = None
    review_result = None
    
    while iteration < max_iterations and not approved:
        print(f"第 {iteration + 1} 次路径规划...")
        
        plan = path_planning(final_profile, reasoning_id=f"{reasoning_id}_iter_{iteration}")
        plan_json = extract_json_objects(plan)
        if not plan_json["plan"]:
            return {"error": "路径规划失败"}

        # 路径评审
        print("正在进行路径评审...")
        review_result_str = plan_review(plan, final_profile)
        if "APPROVED" in review_result_str:
            print("路径通过审核")
            approved = True
            review_result = review_result_str
            final_plan = plan_json["plan"]
            
        iteration += 1
        time.sleep(1)
    
    pl = None
    while not approved and iteration < 3:
        print("正在进行路径修改...")
        new_plan = plan_repeat(plan, review_result)
        plan_json = extract_json_objects(new_plan)
        pl = new_plan
        review_result_str = plan_review(new_plan, final_profile)
        if "APPROVED" in review_result_str:
            print("路径通过审核")
            approved = True
            review_result = review_result_str
            final_plan = plan_json["plan"]
        iteration += 1
        time.sleep(1)
        
    if not approved:
        final_plan = pl
    
    print(f"子图位置：{kg_path}")
    print(f"原始路径：{final_plan}")
    origin_path = final_plan.copy() if final_plan else []
    
    # 实体匹配
    entity = entity_list[row["current_knowledge"]]
    with open("/home/Data/entities_list_15jump/" + entity, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, p in enumerate(final_plan):
        matches = process.extract(p, data, limit=10)
        result = max(matches, key=lambda x: (x[1], len(x[0])))
        final_plan[i] = result[0]

    print(f"处理后的路径：{final_plan}")
    final_plan = list(dict.fromkeys(final_plan))
    
    # 知识图谱路径查找
    finder = KnowledgePathFinder(kg_path)
    start_node = final_plan[0]
    learned_path, correct_count, path_length = finder.find_best_path(start_node, final_plan)
    
    # 评估推理质量
    print("正在评估推理质量...")
    reasoning_score = evaluate_reasoning_quality(
        f"{reasoning_id}_iter_0",
        query,
        final_plan
    )
    
    # 计算综合匹配度（使用path_matching的算法）
    comprehensive_match_rate, order_score, hit_score, reasoning_score = calculate_comprehensive_match_rate(
        learned_path,
        final_plan,
        reasoning_score,
        order_weight=0.6,
        hit_weight=0.2,
        reasoning_weight=0.2
    )
    
    print(f"最终路径：{final_plan}")
    print(f"学习路径：{learned_path}")
    print(f"综合匹配度：{comprehensive_match_rate:.3f}")
    
    # 3. 返回最终结果
    return {
        "user_id": user_id,
        "origin_plan": origin_path,
        "final_plan": final_plan,
        "learned_path": learned_path,
        "order_score": round(order_score, 3),
        "hit_score": round(hit_score, 3),
        "reasoning_score": round(reasoning_score, 3),
        "comprehensive_match_rate": round(comprehensive_match_rate, 3),
        "is_match": 1 if comprehensive_match_rate >= 0.7 else 0,
        "reasoning_id": reasoning_id,
        "error": ""
    }

def main(input_csv, output_csv, start_index=0):
    """
    主函数：读取CSV数据，处理每个用户，保存结果到输出CSV
    """
    try:
        df = pd.read_csv(input_csv)
        print(f"成功读取 {len(df)} 条用户数据")
        
        if start_index > 0:
            print(f"从索引 {start_index} 开始处理数据")
            df = df.iloc[start_index:]
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    results = []
    kg_paths = {
        "圆锥坐标系": "clean_圆锥坐标系_15jump_complete.json",
        "未知数": "clean_未知数_15jump_complete.json",
        "三线坐标": "clean_三线坐标_15jump_complete.json",
        "科学计数法": "clean_科学计数法_15jump_complete.json",
        "级数的项": "clean_级数的项_15jump_complete.json",
        "子广群": "clean_子广群_15jump_complete.json",
        "三角形": "clean_三角形_15jump_complete.json",
        "代数": "clean_代数_15jump_complete.json",
        "弱分离公理": "clean_弱分离公理_15jump_complete.json",
        "无穷级数": "clean_无穷级数_15jump_complete.json",
        "孤点": "clean_孤点_15jump_complete.json",
        "二项式系数的求和公式": "clean_二项式系数的求和公式_15jump_complete.json",
        "数学物理": "clean_数学物理_15jump_complete.json",
        "定义域": "clean_定义域_15jump_complete.json",
        "系数": "clean_系数_15jump_complete.json",
        "整数加法": "clean_整数加法_15jump_complete.json",
        "拓扑空间": "clean_拓扑空间_15jump_complete.json",
        "进位制": "clean_进位制_15jump_complete.json",
        "复数域": "clean_复数域_15jump_complete.json",
        "分形": "clean_分形_15jump_complete.json",
        "调和平均数": "clean_调和平均数_15jump_complete.json",
        "逆矩阵": "clean_逆矩阵_15jump_complete.json",
        "微积分": "clean_微积分_15jump_complete.json",
        "二项式定理": "clean_二项式定理_15jump_complete.json",
        "标准差": "clean_标准差_15jump_complete.json",
        "凹凸性": "clean_凹凸性_15jump_complete.json",
        "邻域": "clean_邻域_15jump_complete.json",
        "组合数": "clean_组合数_15jump_complete.json",
        "向量": "clean_向量_15jump_complete.json",
        "二次方程": "clean_二次方程_15jump_complete.json",
        "乘积法则": "clean_乘积法则_15jump_complete.json",
        "一阶导数": "clean_一阶导数_15jump_complete.json",
        "余弦函数": "clean_余弦函数_15jump_complete.json",
        "二次收敛": "clean_二次收敛_15jump_complete.json",
        "二阶导数": "clean_二阶导数_15jump_complete.json",
        "梯度": "clean_梯度_15jump_complete.json",
        "切线": "clean_切线_15jump_complete.json",
        "参数方程": "clean_参数方程_15jump_complete.json",
        "乘法": "clean_乘法_15jump_complete.json",
        "容积": "clean_容积_15jump_complete.json",
        "线性": "clean_线性_15jump_complete.json",
        "投影": "clean_投影_15jump_complete.json",
        "无限循环小数": "clean_无限循环小数_15jump_complete.json",
        "素数": "clean_素数_15jump_complete.json",
        "极值点": "clean_极值点_15jump_complete.json",
        "阶数": "clean_阶数_15jump_complete.json",
        "零点": "clean_零点_15jump_complete.json",
        "几何学": "clean_几何学_15jump_complete.json",
        "齐次函数": "clean_齐次函数_15jump_complete.json",
        "初始相位": "clean_初始相位_15jump_complete.json",
        "指数函数": "clean_指数函数_15jump_complete.json",
        "QR分解": "clean_QR分解_15jump_complete.json",
        "上界": "clean_上界_15jump_complete.json",
        "鞍点": "clean_鞍点_15jump_complete.json",
        "三维正交坐标系": "clean_三维正交坐标系_15jump_complete.json",
        "幂次": "clean_幂次_15jump_complete.json",
        "分部积分法": "clean_分部积分法_15jump_complete.json",
        "三角函数": "clean_三角函数_15jump_complete.json",
        "直角坐标系": "clean_直角坐标系_15jump_complete.json",
        "模运算": "clean_模运算_15jump_complete.json",
        "单位圆": "clean_单位圆_15jump_complete.json",
        "复数空间": "clean_复数空间_15jump_complete.json",
        "群论": "clean_群论_15jump_complete.json",
        "不动点理论": "clean_不动点理论_15jump_complete.json",
        "矩阵逆": "clean_矩阵逆_15jump_complete.json",
        "微分方程": "clean_微分方程_15jump_complete.json",
        "数域包含关系": "clean_数域包含关系_15jump_complete.json",
        "函数": "clean_函数_15jump_complete.json",
        "向量内积": "clean_向量内积_15jump_complete.json",
        "向量投影": "clean_向量投影_15jump_complete.json",
        "开区间": "clean_开区间_15jump_complete.json",
        "一次函数": "clean_一次函数_15jump_complete.json",
        "双曲线": "clean_双曲线_15jump_complete.json",
        "圆球面": "clean_圆球面_15jump_complete.json",
        "闭开集": "clean_闭开集_15jump_complete.json",
        "变换": "clean_变换_15jump_complete.json",
        "线性外代数": "clean_线性外代数_15jump_complete.json",
        "向量空间": "clean_向量空间_15jump_complete.json",
        "圆锥": "clean_圆锥_15jump_complete.json",
        "有向曲面": "clean_有向曲面_15jump_complete.json",
        "幂运算": "clean_幂运算_15jump_complete.json",
        "连续可导函数": "clean_连续可导函数_15jump_complete.json",
        "阶乘": "clean_阶乘_15jump_complete.json",
        "位似变换": "clean_位似变换_15jump_complete.json",
        "矩阵乘法": "clean_矩阵乘法_15jump_complete.json",
        "求和符号": "clean_求和符号_15jump_complete.json",
        "矩阵": "clean_矩阵_15jump_complete.json",
        "共轭转置": "clean_共轭转置_15jump_complete.json",
        "内积": "clean_内积_15jump_complete.json",
        "狭义相对论": "clean_狭义相对论_15jump_complete.json",
        "坐标轴": "clean_坐标轴_15jump_complete.json",
        "锥曲面与球面": "clean_锥曲面与球面_15jump_complete.json",
        "椭圆曲线": "clean_椭圆曲线_15jump_complete.json",
        "指数运算": "clean_指数运算_15jump_complete.json",
        "多项式": "clean_多项式_15jump_complete.json",
        "收敛速度": "clean_收敛速度_15jump_complete.json",
        "可导性": "clean_可导性_15jump_complete.json",
        "向量加法": "clean_向量加法_15jump_complete.json",
        "多元实函数": "clean_多元实函数_15jump_complete.json",
        "连续函数的闭支撑": "clean_连续函数的闭支撑_15jump_complete.json",
        "二项式展开": "clean_二项式展开_15jump_complete.json",
        "柯西": "clean_柯西_15jump_complete.json",
        "数学集合理论": "clean_数学集合理论_15jump_complete.json",
        "判别式": "clean_判别式_15jump_complete.json",
        "同构": "clean_同构_15jump_complete.json",
        "反函数": "clean_反函数_15jump_complete.json",
        "分数指数": "clean_分数指数_15jump_complete.json",
        "单位根": "clean_单位根_15jump_complete.json",
        "标量λ": "clean_标量λ_15jump_complete.json",
        "对称性": "clean_对称性_15jump_complete.json",
        "正交矩阵": "clean_正交矩阵_15jump_complete.json",
        "函数复合": "clean_函数复合_15jump_complete.json",
        "拓扑方法": "clean_拓扑方法_15jump_complete.json",
        "子群": "clean_子群_15jump_complete.json",
        "几何体": "clean_几何体_15jump_complete.json",
        "数的分割": "clean_数的分割_15jump_complete.json",
        "正交群": "clean_正交群_15jump_complete.json",
        "基底": "clean_基底_15jump_complete.json",
        "固定点": "clean_固定点_15jump_complete.json",
        "角频率": "clean_角频率_15jump_complete.json",
        "正实数集合": "clean_正实数集合_15jump_complete.json",
        "共线": "clean_共线_15jump_complete.json",
        "平方根": "clean_平方根_15jump_complete.json",
        "四面体": "clean_四面体_15jump_complete.json",
        "拉格朗日": "clean_拉格朗日_15jump_complete.json",
        "几何-调和平均数": "clean_几何-调和平均数_15jump_complete.json",
        "古典力学": "clean_古典力学_15jump_complete.json",
        "常数项": "clean_常数项_15jump_complete.json",
        "正实数": "clean_正实数_15jump_complete.json",
        "线性函数": "clean_线性函数_15jump_complete.json",
        "平面几何": "clean_平面几何_15jump_complete.json",
        "平面图": "clean_平面图_15jump_complete.json",
        "最佳多项式": "clean_最佳多项式_15jump_complete.json",
        "因式": "clean_因式_15jump_complete.json",
        "极限集合": "clean_极限集合_15jump_complete.json",
        "因式分解": "clean_因式分解_15jump_complete.json"
    }
    
    df = df.reset_index(drop=True)
    
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        actual_index = index + start_index
        print(f"\n处理第 {actual_index + 1}/{len(df) + start_index} 条数据...")
        
        try:
            result = process_user_learning_path(
                row,
                '/home/Data/subgraph_15jump/final/' + kg_paths[row["current_knowledge"]]
            )
            results.append(result)
        except Exception as e:
            print(f"处理第 {actual_index + 1} 条数据时出错: {e}")
            results.append({
                "user_id": row.get('user_id', 'unknown'),
                "origin_plan": [],
                "final_plan": [],
                "learned_path": [],
                "order_score": 0.0,
                "hit_score": 0.0,
                "reasoning_score": 0.0,
                "comprehensive_match_rate": 0.0,
                "is_match": 0,
                "reasoning_id": "",
                "error": str(e)
            })
        
        # 每处理5条数据就追加到输出文件中
        if (index + 1) % 5 == 0 or (index + 1) == len(df):
            try:
                results_df = pd.DataFrame(results)
                if index + 1 <= 5:
                    if start_index > 0:
                        results_df.to_csv(output_csv, index=False, mode='a', header=False)
                    else:
                        results_df.to_csv(output_csv, index=False, mode='w')
                else:
                    results_df.to_csv(output_csv, index=False, mode='a', header=False)
                print(f"已处理 {actual_index + 1} 条数据，结果已追加到 {output_csv}")
                results = []
            except Exception as e:
                print(f"保存结果失败: {e}")
        
        time.sleep(1)
    
    print(f"所有数据处理完成，结果已保存到 {output_csv}")
    print(f"推理过程文件保存在: {REASONING_DIR}/")

if __name__ == "__main__":
    input_csv = "/home/lms/project/dify-backend/data/test_data/user_middle_test_log/user_middle_test_log_user_knowledge_profile.csv"
    output_csv = "/home/lms/project/dify-backend/data/test_data/user_middle_test_log/comprehensive_match_output_user_middle_user_knowledge_profile.csv"
    
    main(input_csv, output_csv, 0)
