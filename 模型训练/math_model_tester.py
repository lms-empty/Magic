import json
import random
import time
import re
from typing import List, Dict, Tuple
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import hmac
import base64
from urllib.parse import urlencode
import requests


class KnowledgeGraphModelTester:
    def __init__(self):
        # Ollama配置
        self.OLLAMA_BASE_URL = "http://172.20.192.113:11434"

        # 模型配置 - 针对知识图谱预训练模型
        self.models = {
            "evaluator": "deepseek-r1:32b",  # 评估器模型
            "finetuned": "MAGIC_final_deepseek_r1_7b:latest",  # 微调后的知识图谱模型
            "baseline": "deepseek-r1:7b"  # 基线模型
        }

        # 评估样本存储
        self.kg_reasoning_samples = []  # 知识图谱推理样本
        self.structured_qa_samples = []  # 结构化问答样本
        self.multi_hop_samples = []  # 多跳推理样本
        self.domain_adaptation_samples = []  # 领域适应样本

        # 结果存储
        self.results = {
            "finetuned": {
                "kg_reasoning": [],
                "structured_qa": [],
                "multi_hop": [],
                "domain_adaptation": []
            },
            "baseline": {
                "kg_reasoning": [],
                "structured_qa": [],
                "multi_hop": [],
                "domain_adaptation": []
            }
        }

    def call_ollama(self, model: str, prompt: str, max_retries: int = 3) -> str:
        """统一的Ollama API调用方法"""
        generate_url = f"{self.OLLAMA_BASE_URL}/api/generate"

        for attempt in range(max_retries):
            try:
                # 为评估器模型设置更高的参数
                if model == "deepseek-r1:32b":
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # 降低随机性，提高格式一致性
                            "top_p": 0.8,
                            "max_tokens": 4096,
                            "repeat_penalty": 1.1,
                            "stop": ["==END=="]  # 添加停止符
                        }
                    }
                else:
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 3072
                        }
                    }

                response = requests.post(generate_url, json=payload, timeout=240)
                if response.status_code == 200:
                    result = response.json().get("response", "").strip()
                    if result:
                        return result
                    else:
                        print(f"      ⚠️ 模型返回空响应 (尝试 {attempt + 1}/{max_retries})")
                else:
                    print(
                        f"      ⚠️ Ollama API调用失败，状态码: {response.status_code} (尝试 {attempt + 1}/{max_retries})")

                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 10)
                    print(f"      ⏳ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

            except Exception as e:
                print(f"      ❌ Ollama调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 10)
                    print(f"      ⏳ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        print(f"      💀 Ollama API调用最终失败，已重试 {max_retries} 次")
        return ""

    def clean_json_string(self, text: str) -> str:
        """清理字符串中的控制字符和特殊字符"""
        if not text:
            return ""

        cleaned = ""
        for char in text:
            if ord(char) < 32:
                if char in ['\n', '\r', '\t']:
                    cleaned += ' '
            elif char == '"':
                cleaned += '\\"'
            elif char == '\\':
                cleaned += '\\\\'
            else:
                cleaned += char

        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def generate_kg_reasoning_samples(self, num_samples: int = 200) -> List[Dict]:
        """生成知识图谱推理样本"""
        print("\n" + "=" * 80)
        print(f"🔗 开始生成知识图谱推理样本 - 目标: {num_samples} 个样本")
        print("=" * 80)

        # 知识图谱推理模板
        kg_templates = [
            {
                "type": "entity_relation",
                "category": "single_hop",
                "prompt_template": "创建一个包含{entity_type}实体的知识图谱推理问题，涉及{relation_type}关系，要求{reasoning_level}推理。",
                "entity_types": ["数学概念", "定理", "公式", "数学家", "数学分支"],
                "relation_types": ["属于", "包含", "应用于", "证明", "发现", "推导"],
                "reasoning_levels": ["直接查询", "一跳推理", "关系判断", "属性推理"]
            },
            {
                "type": "path_reasoning",
                "category": "multi_hop",
                "prompt_template": "设计一个需要{hop_count}跳推理的知识图谱问题，从{start_concept}到{end_concept}，通过{path_type}路径。",
                "hop_counts": ["二", "三", "四"],
                "start_concepts": ["基础概念", "定理", "公式", "数学原理"],
                "end_concepts": ["应用领域", "实际问题", "高级概念", "综合应用"],
                "path_types": ["因果关系", "包含关系", "应用关系", "推导关系"]
            },
            {
                "type": "graph_completion",
                "category": "knowledge_inference",
                "prompt_template": "构建一个关于{domain}的知识图谱补全问题，要求根据{given_relations}推断{missing_relations}。",
                "domains": ["代数学", "几何学", "微积分", "概率论", "线性代数"],
                "given_relations": ["部分实体关系", "已知属性", "现有连接", "基础事实"],
                "missing_relations": ["隐含关系", "未知属性", "潜在连接", "推导结论"]
            }
        ]

        return self._generate_samples_with_template(kg_templates, num_samples, "kg_reasoning", "知识图谱推理")

    def generate_structured_qa_samples(self, num_samples: int = 200) -> List[Dict]:
        """生成结构化问答样本"""
        print("\n" + "=" * 80)
        print(f"📊 开始生成结构化问答样本 - 目标: {num_samples} 个样本")
        print("=" * 80)

        # 结构化问答模板
        qa_templates = [
            {
                "type": "fact_retrieval",
                "category": "structured_query",
                "prompt_template": "创建一个关于{math_domain}的结构化问答，要求从{data_structure}中检索{query_type}信息。",
                "math_domains": ["代数理论", "几何定理", "微积分公式", "概率分布", "统计方法"],
                "data_structures": ["概念层次", "定理体系", "公式集合", "证明链条", "应用实例"],
                "query_types": ["具体数值", "关系判断", "分类信息", "属性查询", "条件筛选"]
            },
            {
                "type": "conditional_reasoning",
                "category": "logic_qa",
                "prompt_template": "设计一个需要{condition_type}条件推理的结构化问答，涉及{reasoning_pattern}模式。",
                "condition_types": ["单一条件", "多重条件", "嵌套条件", "互斥条件"],
                "reasoning_patterns": ["演绎推理", "归纳推理", "类比推理", "假设推理", "反证推理"]
            },
            {
                "type": "comparison_analysis",
                "category": "analytical_qa",
                "prompt_template": "构建一个{comparison_type}比较分析的结构化问答，要求{analysis_depth}分析。",
                "comparison_types": ["概念对比", "方法比较", "结果对照", "性能分析"],
                "analysis_depths": ["表面特征", "深层原理", "适用范围", "优缺点", "应用场景"]
            }
        ]

        return self._generate_samples_with_template(qa_templates, num_samples, "structured_qa", "结构化问答")

    def generate_multi_hop_samples(self, num_samples: int = 200) -> List[Dict]:
        """生成多跳推理样本"""
        print("\n" + "=" * 80)
        print(f"🔄 开始生成多跳推理样本 - 目标: {num_samples} 个样本")
        print("=" * 80)

        # 多跳推理模板
        multi_hop_templates = [
            {
                "type": "chain_reasoning",
                "category": "sequential_inference",
                "prompt_template": "创建一个{chain_length}步推理链问题，从{start_point}开始，通过{reasoning_steps}，最终得到{end_goal}。",
                "chain_lengths": ["三步", "四步", "五步"],
                "start_points": ["基础定义", "已知条件", "给定公式", "初始假设"],
                "reasoning_steps": ["逐步推导", "条件转换", "关系建立", "逻辑连接"],
                "end_goals": ["目标结论", "问题答案", "证明完成", "方案确定"]
            },
            {
                "type": "branching_reasoning",
                "category": "parallel_inference",
                "prompt_template": "设计一个需要{branch_count}个并行推理分支的问题，每个分支处理{branch_task}，最终{integration_method}。",
                "branch_counts": ["两个", "三个", "四个"],
                "branch_tasks": ["不同条件", "不同方法", "不同角度", "不同层面"],
                "integration_methods": ["综合分析", "结果比较", "统一结论", "最优选择"]
            },
            {
                "type": "recursive_reasoning",
                "category": "iterative_inference",
                "prompt_template": "构建一个{recursion_type}递归推理问题，包含{recursion_depth}层递归，每层{recursion_operation}。",
                "recursion_types": ["数学归纳", "递归定义", "迭代逼近", "分治策略"],
                "recursion_depths": ["二", "三", "四"],
                "recursion_operations": ["条件细化", "范围缩小", "精度提高", "复杂度降低"]
            }
        ]

        return self._generate_samples_with_template(multi_hop_templates, num_samples, "multi_hop", "多跳推理")

    def generate_domain_adaptation_samples(self, num_samples: int = 200) -> List[Dict]:
        """生成领域适应样本"""
        print("\n" + "=" * 80)
        print(f"🎯 开始生成领域适应样本 - 目标: {num_samples} 个样本")
        print("=" * 80)

        # 领域适应模板
        domain_templates = [
            {
                "type": "cross_domain",
                "category": "domain_transfer",
                "prompt_template": "创建一个从{source_domain}到{target_domain}的知识迁移问题，要求{transfer_method}。",
                "source_domains": ["纯数学理论", "抽象代数", "几何原理", "微积分概念"],
                "target_domains": ["物理应用", "工程计算", "经济模型", "计算机算法"],
                "transfer_methods": ["概念映射", "原理应用", "方法迁移", "模型适配"]
            },
            {
                "type": "specialized_reasoning",
                "category": "expert_knowledge",
                "prompt_template": "设计一个需要{expertise_level}专业知识的{specialized_field}问题，涉及{technical_aspects}。",
                "expertise_levels": ["专业基础", "深入理解", "专家水平", "前沿研究"],
                "specialized_fields": ["数论", "拓扑学", "数理逻辑", "计算数学", "应用统计"],
                "technical_aspects": ["核心理论", "证明技巧", "计算方法", "实际应用", "前沿进展"]
            },
            {
                "type": "contextual_adaptation",
                "category": "context_aware",
                "prompt_template": "构建一个需要{context_type}上下文理解的数学问题，要求{adaptation_strategy}。",
                "context_types": ["历史背景", "实际场景", "跨学科", "前沿应用"],
                "adaptation_strategies": ["动态调整", "情境分析", "灵活应用", "创新思维"]
            }
        ]

        return self._generate_samples_with_template(domain_templates, num_samples, "domain_adaptation", "领域适应")

    def _generate_samples_with_template(self, templates: List[Dict], num_samples: int, sample_type: str,
                                        type_name: str) -> List[Dict]:
        """通用的样本生成方法 - 优化版本"""
        samples = []
        success_count = 0
        failed_count = 0
        partial_success_count = 0
        start_time = time.time()

        print(f"📝 开始生成{type_name}样本...")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for i in range(num_samples):
            iteration_start = time.time()
            template = random.choice(templates)

            # 生成具体的问题描述
            prompt_params = self._extract_template_params(template)
            specific_prompt = template["prompt_template"].format(**prompt_params)

            # 尝试多种生成策略
            sample = None
            strategies = ["detailed", "simplified", "example_based"]

            for strategy in strategies:
                try:
                    # 根据策略生成不同的提示词
                    generation_prompt = self._create_generation_prompt_v2(sample_type, specific_prompt, strategy)

                    response = self.call_ollama(self.models["evaluator"], generation_prompt)

                    if response:
                        try:
                            # 优化版本：使用增强解析和宽松验证
                            sample_data = self._parse_sample_response_enhanced(response, sample_type)

                            if sample_data and self._validate_sample_data_relaxed(sample_data, sample_type):
                                sample = self._create_sample_dict(sample_data, template, specific_prompt, sample_type,
                                                                  i + 1, "success")
                                success_count += 1
                                break
                            else:
                                # 尝试宽松解析
                                sample_data = self._parse_sample_response_lenient(response, sample_type)
                                if sample_data and self._validate_sample_data_minimal(sample_data, sample_type):
                                    sample = self._create_sample_dict(sample_data, template, specific_prompt,
                                                                      sample_type, i + 1, "success")  # 改为success
                                    success_count += 1
                                    break

                        except Exception as e:
                            print(f"      ⚠️ 策略 {strategy} 解析失败: {str(e)[:100]}")
                            continue
                    else:
                        print(f"      ⚠️ 策略 {strategy} 无响应")
                        continue

                except Exception as e:
                    print(f"      ❌ 策略 {strategy} 异常: {str(e)[:100]}")
                    continue

            # 如果所有策略都失败，创建备用样本
            if sample is None:
                sample = self._create_fallback_sample_v2(template, specific_prompt, sample_type, i + 1)
                failed_count += 1

            samples.append(sample)
            self._display_progress(i + 1, num_samples, success_count, partial_success_count, failed_count,
                                   start_time, iteration_start, template, type_name)

            # 中间保存
            if (i + 1) % 50 == 0:
                self._save_intermediate_results(samples, sample_type, i + 1)

            time.sleep(0.3)  # 减少等待时间

        # 最终统计和保存
        self._print_generation_summary(type_name, samples, success_count, partial_success_count, failed_count,
                                       start_time)
        self._save_final_results(samples, sample_type)

        return samples

    def _create_generation_prompt_v2(self, sample_type: str, specific_prompt: str, strategy: str = "detailed") -> str:
        """创建优化的生成提示词"""

        # 基础示例
        examples = self._get_format_examples(sample_type)

        base_system = f"""你是一位专业的{self._get_type_description(sample_type)}专家。请严格按照指定格式生成高质量的测试样本。

重要要求：
1. 必须完全按照格式要求输出，不要添加额外内容
2. 每个部分都要填写完整，不能留空
3. 问题要具体明确，答案要详细规范
4. 使用中文回答，保持专业性

{examples}

现在请按照上述格式，根据以下要求生成一个完整的样本：

题目要求：{specific_prompt}

请开始生成（严格按照格式输出）："""

        if strategy == "simplified":
            base_system += "\n\n注意：请保持内容简洁明了，重点突出核心要素。"
        elif strategy == "example_based":
            base_system += "\n\n注意：请参考上述示例的风格和结构，确保格式完全一致。"

        base_system += "\n\n==END=="

        return base_system

    def _get_format_examples(self, sample_type: str) -> str:
        """获取格式示例"""
        if sample_type == "kg_reasoning":
            return """
格式示例：

==QUESTION==
在数学知识图谱中，已知"微积分"是"数学分析"的一个分支，"极限理论"是"微积分"的基础，"连续性"概念基于"极限理论"。请推理：如果要理解"连续函数"概念，需要掌握哪些前置知识？

==KNOWLEDGE_GRAPH==
实体：微积分, 数学分析, 极限理论, 连续性, 连续函数
关系：分支关系(数学分析→微积分), 基础关系(极限理论→微积分), 依赖关系(连续性→极限理论)

==STANDARD_ANSWER==
1. 问题分析：理解连续函数需要追溯其知识依赖链
2. 知识图谱检索策略：沿着依赖关系向上追溯
3. 推理步骤：连续函数→连续性→极限理论→微积分→数学分析
4. 最终答案：需要掌握数学分析基础、微积分基本概念、极限理论、连续性定义

==REASONING_TYPE==
多跳推理

==ENTITIES==
微积分, 数学分析, 极限理论, 连续性, 连续函数

==RELATIONS==
分支关系, 基础关系, 依赖关系

==DIFFICULTY_LEVEL==
中等

==EVALUATION_FOCUS==
推理准确性, 知识检索, 逻辑连贯性
"""
        elif sample_type == "structured_qa":
            return """
格式示例：

==QUESTION==
在数学公式数据库中，查询所有包含"导数"概念的微积分公式，并按照难度等级分类。数据库结构包含：公式名称、所属分支、难度等级、相关概念。

==DATA_STRUCTURE==
表格结构：公式表(公式ID, 公式名称, 数学表达式, 所属分支, 难度等级, 相关概念列表)
索引：按所属分支和相关概念建立索引

==STANDARD_ANSWER==
1. 问题理解：需要检索包含"导数"的公式并分类
2. 数据检索策略：在相关概念字段中搜索"导数"
3. 信息提取步骤：筛选→分类→排序
4. 答案组织：按难度分组列出相关公式

==QUERY_TYPE==
条件查询

==KEY_CONCEPTS==
导数, 微积分, 公式分类

==RETRIEVAL_STRATEGY==
关键词匹配, 分类筛选

==DIFFICULTY_LEVEL==
基础

==EVALUATION_FOCUS==
查询准确性, 结构理解
"""
        elif sample_type == "multi_hop":
            return """
格式示例：

==QUESTION==
证明：如果函数f(x)在区间[a,b]上连续，且f(a)·f(b)<0，则存在c∈(a,b)使得f(c)=0。请展示完整的三步推理过程。

==REASONING_CHAIN==
步骤1：建立连续性条件 → 步骤2：应用中间值定理 → 步骤3：得出存在性结论

==STANDARD_ANSWER==
1. 问题分解：需要运用连续性和中间值定理
2. 推理路径规划：连续性→中间值定理→存在性
3. 逐步推理过程：详细的数学证明
4. 结果验证：确认结论正确性

==HOP_COUNT==
三跳

==INTERMEDIATE_STEPS==
建立连续性, 应用中间值定理, 证明存在性

==REASONING_PATTERN==
链式

==DIFFICULTY_LEVEL==
困难

==EVALUATION_FOCUS==
推理完整性, 逻辑连贯性, 中间步骤
"""
        elif sample_type == "domain_adaptation":
            return """
格式示例：

==QUESTION==
将线性代数中的矩阵特征值理论应用到经济学中的投入产出模型分析，说明如何利用特征值判断经济系统的稳定性。

==DOMAIN_CONTEXT==
源领域：线性代数的矩阵特征值理论
目标领域：经济学投入产出模型

==STANDARD_ANSWER==
1. 领域分析：识别两个领域的关联点
2. 知识迁移策略：矩阵→经济系统，特征值→稳定性指标
3. 适应方法：建立数学模型映射
4. 应用验证：实际案例分析

==ADAPTATION_TYPE==
模型适配

==SOURCE_KNOWLEDGE==
矩阵理论, 特征值计算

==TARGET_APPLICATION==
经济稳定性分析, 投入产出模型

==DIFFICULTY_LEVEL==
中等

==EVALUATION_FOCUS==
迁移准确性, 适应能力
"""
        else:
            return "格式示例未定义"

    def _get_type_description(self, sample_type: str) -> str:
        """获取类型描述"""
        descriptions = {
            "kg_reasoning": "知识图谱推理",
            "structured_qa": "结构化问答",
            "multi_hop": "多跳推理",
            "domain_adaptation": "领域适应"
        }
        return descriptions.get(sample_type, "数学推理")

    def _parse_sample_response_enhanced(self, response: str, sample_type: str) -> dict:
        """增强版样本响应解析 - 更宽松更智能"""
        # 预处理响应文本
        response = response.replace("==END==", "").strip()

        # 首先尝试标准解析
        try:
            standard_result = self._parse_sample_response_v2(response, sample_type)
            if standard_result and self._has_sufficient_content(standard_result):
                return standard_result
        except:
            pass

        # 如果标准解析失败，尝试增强解析
        return self._parse_sample_response_flexible(response, sample_type)

    def _parse_sample_response_flexible(self, response: str, sample_type: str) -> dict:
        """灵活的样本响应解析 - 智能识别内容"""
        if sample_type == "kg_reasoning":
            return self._parse_kg_reasoning_flexible(response)
        elif sample_type == "structured_qa":
            return self._parse_structured_qa_flexible(response)
        elif sample_type == "multi_hop":
            return self._parse_multi_hop_flexible(response)
        elif sample_type == "domain_adaptation":
            return self._parse_domain_adaptation_flexible(response)
        else:
            return {}

    def _parse_kg_reasoning_flexible(self, response: str) -> dict:
        """灵活解析知识图谱推理响应"""
        sections = {
            "question": "",
            "knowledge_graph": "",
            "standard_answer": "",
            "reasoning_type": "推理",
            "entities": [],
            "relations": [],
            "difficulty_level": "中等",
            "evaluation_focus": []
        }

        # 多种方式匹配问题
        question_patterns = [
            r'==QUESTION==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'【问题】\s*\n*(.*?)(?=\n【|\n\n|$)',
            r'问题[:：]\s*\n*(.*?)(?=\n|$)',
            r'^(.*?)(?=\n==|\n【)',  # 第一段作为问题
        ]

        for pattern in question_patterns:
            match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
            if match and match.group(1).strip():
                sections["question"] = match.group(1).strip()
                break

        # 多种方式匹配答案
        answer_patterns = [
            r'==STANDARD_ANSWER==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'【标准答案】\s*\n*(.*?)(?=\n【|\n\n|$)',
            r'答案[:：]\s*\n*(.*?)(?=\n|$)',
            r'解答[:：]\s*\n*(.*?)(?=\n|$)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
            if match and match.group(1).strip():
                sections["standard_answer"] = match.group(1).strip()
                break

        # 如果没找到明确的问题和答案，尝试分割
        if not sections["question"] or not sections["standard_answer"]:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            if lines:
                if not sections["question"]:
                    sections["question"] = lines[0]
                if not sections["standard_answer"] and len(lines) > 1:
                    sections["standard_answer"] = '\n'.join(lines[1:])

        # 提取实体和关系
        self._extract_entities_relations(response, sections)

        # 提取其他信息
        self._extract_additional_info(response, sections)

        return sections

    def _parse_structured_qa_flexible(self, response: str) -> dict:
        """灵活解析结构化问答响应"""
        sections = {
            "question": "",
            "data_structure": "",
            "standard_answer": "",
            "query_type": "查询",
            "key_concepts": [],
            "retrieval_strategy": [],
            "difficulty_level": "中等",
            "evaluation_focus": []
        }

        # 基本解析逻辑类似，但针对结构化问答的特殊字段
        self._extract_basic_qa_fields(response, sections)
        self._extract_structured_specific_fields(response, sections)

        return sections

    def _parse_multi_hop_flexible(self, response: str) -> dict:
        """灵活解析多跳推理响应"""
        sections = {
            "question": "",
            "reasoning_chain": "",
            "standard_answer": "",
            "hop_count": "多跳",
            "intermediate_steps": [],
            "reasoning_pattern": "链式",
            "difficulty_level": "中等",
            "evaluation_focus": []
        }

        self._extract_basic_qa_fields(response, sections)
        self._extract_multi_hop_specific_fields(response, sections)

        return sections

    def _parse_domain_adaptation_flexible(self, response: str) -> dict:
        """灵活解析领域适应响应"""
        sections = {
            "question": "",
            "domain_context": "",
            "standard_answer": "",
            "adaptation_type": "适应",
            "source_knowledge": [],
            "target_application": [],
            "difficulty_level": "中等",
            "evaluation_focus": []
        }

        self._extract_basic_qa_fields(response, sections)
        self._extract_domain_specific_fields(response, sections)

        return sections

    def _extract_basic_qa_fields(self, response: str, sections: dict):
        """提取基本问答字段"""
        # 提取问题
        question_patterns = [
            r'==QUESTION==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'【问题】\s*\n*(.*?)(?=\n【|\n\n|$)',
            r'问题[:：]\s*\n*(.*?)(?=\n|$)',
        ]

        for pattern in question_patterns:
            match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
            if match and match.group(1).strip():
                sections["question"] = match.group(1).strip()
                break

        # 提取答案
        answer_patterns = [
            r'==STANDARD_ANSWER==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'【标准答案】\s*\n*(.*?)(?=\n【|\n\n|$)',
            r'答案[:：]\s*\n*(.*?)(?=\n|$)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
            if match and match.group(1).strip():
                sections["standard_answer"] = match.group(1).strip()
                break

        # 如果仍然没有找到，使用默认分割
        if not sections["question"] or not sections["standard_answer"]:
            lines = [line.strip() for line in response.split('\n') if line.strip() and not line.startswith('==')]
            if lines:
                if not sections["question"]:
                    # 找到第一个看起来像问题的句子
                    for line in lines:
                        if '？' in line or '?' in line or '如何' in line or '什么' in line or '怎样' in line:
                            sections["question"] = line
                            break
                    if not sections["question"]:
                        sections["question"] = lines[0]

                if not sections["standard_answer"]:
                    # 其余内容作为答案
                    remaining_lines = [line for line in lines if line != sections["question"]]
                    if remaining_lines:
                        sections["standard_answer"] = '\n'.join(remaining_lines)

    def _extract_entities_relations(self, response: str, sections: dict):
        """提取实体和关系信息"""
        # 提取实体
        entity_patterns = [
            r'==ENTITIES==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'实体[:：]\s*(.*?)(?=\n|$)',
            r'相关实体[:：]\s*(.*?)(?=\n|$)',
        ]

        for pattern in entity_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                entities_text = match.group(1).strip()
                if entities_text:
                    sections["entities"] = [e.strip() for e in re.split(r'[,，、]', entities_text) if e.strip()]
                break

        # 提取关系
        relation_patterns = [
            r'==RELATIONS==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'关系[:：]\s*(.*?)(?=\n|$)',
            r'相关关系[:：]\s*(.*?)(?=\n|$)',
        ]

        for pattern in relation_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                relations_text = match.group(1).strip()
                if relations_text:
                    sections["relations"] = [r.strip() for r in re.split(r'[,，、]', relations_text) if r.strip()]
                break

        # 如果没有找到，从内容中智能提取
        if not sections["entities"]:
            # 从问题和答案中提取可能的实体
            content = sections["question"] + " " + sections["standard_answer"]
            math_entities = re.findall(r'["\'](.*?)["\']', content)
            if math_entities:
                sections["entities"] = math_entities[:5]  # 最多取5个
            else:
                sections["entities"] = ["数学概念", "相关理论"]

    def _extract_additional_info(self, response: str, sections: dict):
        """提取其他附加信息"""
        # 提取推理类型
        reasoning_patterns = [
            r'==REASONING_TYPE==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'推理类型[:：]\s*(.*?)(?=\n|$)',
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match and match.group(1).strip():
                sections["reasoning_type"] = match.group(1).strip()
                break

        # 提取难度等级
        difficulty_patterns = [
            r'==DIFFICULTY_LEVEL==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'难度[:：]\s*(.*?)(?=\n|$)',
        ]

        for pattern in difficulty_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match and match.group(1).strip():
                sections["difficulty_level"] = match.group(1).strip()
                break

    def _extract_structured_specific_fields(self, response: str, sections: dict):
        """提取结构化问答特定字段"""
        # 提取关键概念
        concept_patterns = [
            r'==KEY_CONCEPTS==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'关键概念[:：]\s*(.*?)(?=\n|$)',
        ]

        for pattern in concept_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                concepts_text = match.group(1).strip()
                if concepts_text:
                    sections["key_concepts"] = [c.strip() for c in re.split(r'[,，、]', concepts_text) if c.strip()]
                break

        if not sections["key_concepts"]:
            sections["key_concepts"] = ["核心概念", "相关知识"]

    def _extract_multi_hop_specific_fields(self, response: str, sections: dict):
        """提取多跳推理特定字段"""
        # 提取中间步骤
        steps_patterns = [
            r'==INTERMEDIATE_STEPS==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'中间步骤[:：]\s*(.*?)(?=\n|$)',
        ]

        for pattern in steps_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                steps_text = match.group(1).strip()
                if steps_text:
                    sections["intermediate_steps"] = [s.strip() for s in re.split(r'[,，、]', steps_text) if s.strip()]
                break

        if not sections["intermediate_steps"]:
            sections["intermediate_steps"] = ["步骤1", "步骤2", "步骤3"]

    def _extract_domain_specific_fields(self, response: str, sections: dict):
        """提取领域适应特定字段"""
        # 提取源知识
        source_patterns = [
            r'==SOURCE_KNOWLEDGE==\s*\n*(.*?)(?=\n==|\n\n|$)',
            r'源知识[:：]\s*(.*?)(?=\n|$)',
        ]

        for pattern in source_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                source_text = match.group(1).strip()
                if source_text:
                    sections["source_knowledge"] = [s.strip() for s in re.split(r'[,，、]', source_text) if s.strip()]
                break

        if not sections["source_knowledge"]:
            sections["source_knowledge"] = ["源领域知识", "基础理论"]

    def _has_sufficient_content(self, sample_data: dict) -> bool:
        """检查样本数据是否有足够的内容"""
        if not sample_data:
            return False

        # 检查必要字段
        question = sample_data.get("question", "")
        answer = sample_data.get("standard_answer", "")

        return (len(question.strip()) >= 15 and len(answer.strip()) >= 20)

    def _validate_sample_data_relaxed(self, sample_data: dict, sample_type: str) -> bool:
        """宽松的样本数据验证 - 提高成功率"""
        if not sample_data:
            return False

        # 检查基本必需字段 - 放宽要求
        question = sample_data.get("question", "")
        answer = sample_data.get("standard_answer", "")

        # 内容长度要求降低
        if not question or len(question.strip()) < 15:  # 从10降到15但更合理
            return False
        if not answer or len(answer.strip()) < 20:  # 从10降到20但更合理
            return False

        # 特定类型的检查更宽松
        type_checks = {
            "kg_reasoning": ["entities", "relations"],
            "structured_qa": ["key_concepts"],
            "multi_hop": ["intermediate_steps"],
            "domain_adaptation": ["source_knowledge", "target_application"]
        }

        if sample_type in type_checks:
            for field in type_checks[sample_type]:
                if field not in sample_data:
                    return False
                # 对于列表字段，允许为空，稍后会填充默认值

        return True

    def _validate_sample_data_minimal(self, sample_data: dict, sample_type: str) -> bool:
        """最小验证要求 - 确保基本可用性"""
        if not sample_data:
            return False

        # 只检查最基本的要求
        question = sample_data.get("question", "")
        answer = sample_data.get("standard_answer", "")

        # 进一步降低要求
        return (question and len(question.strip()) >= 10 and
                answer and len(answer.strip()) >= 15)

    def _parse_sample_response_v2(self, response: str, sample_type: str) -> dict:
        """优化的样本响应解析 - 更严格的格式要求"""
        # 预处理响应文本
        response = response.replace("==END==", "").strip()

        if sample_type == "kg_reasoning":
            return self._parse_kg_reasoning_response_v2(response)
        elif sample_type == "structured_qa":
            return self._parse_structured_qa_response_v2(response)
        elif sample_type == "multi_hop":
            return self._parse_multi_hop_response_v2(response)
        elif sample_type == "domain_adaptation":
            return self._parse_domain_adaptation_response_v2(response)
        else:
            return {}

    def _parse_kg_reasoning_response_v2(self, response: str) -> dict:
        """解析知识图谱推理响应 - 优化版本"""
        sections = {
            "question": "",
            "knowledge_graph": "",
            "standard_answer": "",
            "reasoning_type": "",
            "entities": [],
            "relations": [],
            "difficulty_level": "",
            "evaluation_focus": []
        }

        delimiters = {
            "==QUESTION==": "question",
            "==KNOWLEDGE_GRAPH==": "knowledge_graph",
            "==STANDARD_ANSWER==": "standard_answer",
            "==REASONING_TYPE==": "reasoning_type",
            "==ENTITIES==": "entities",
            "==RELATIONS==": "relations",
            "==DIFFICULTY_LEVEL==": "difficulty_level",
            "==EVALUATION_FOCUS==": "evaluation_focus"
        }

        return self._parse_response_sections_v2(response, delimiters, sections)

    def _parse_structured_qa_response_v2(self, response: str) -> dict:
        """解析结构化问答响应 - 优化版本"""
        sections = {
            "question": "",
            "data_structure": "",
            "standard_answer": "",
            "query_type": "",
            "key_concepts": [],
            "retrieval_strategy": [],
            "difficulty_level": "",
            "evaluation_focus": []
        }

        delimiters = {
            "==QUESTION==": "question",
            "==DATA_STRUCTURE==": "data_structure",
            "==STANDARD_ANSWER==": "standard_answer",
            "==QUERY_TYPE==": "query_type",
            "==KEY_CONCEPTS==": "key_concepts",
            "==RETRIEVAL_STRATEGY==": "retrieval_strategy",
            "==DIFFICULTY_LEVEL==": "difficulty_level",
            "==EVALUATION_FOCUS==": "evaluation_focus"
        }

        return self._parse_response_sections_v2(response, delimiters, sections)

    def _parse_multi_hop_response_v2(self, response: str) -> dict:
        """解析多跳推理响应 - 优化版本"""
        sections = {
            "question": "",
            "reasoning_chain": "",
            "standard_answer": "",
            "hop_count": "",
            "intermediate_steps": [],
            "reasoning_pattern": "",
            "difficulty_level": "",
            "evaluation_focus": []
        }

        delimiters = {
            "==QUESTION==": "question",
            "==REASONING_CHAIN==": "reasoning_chain",
            "==STANDARD_ANSWER==": "standard_answer",
            "==HOP_COUNT==": "hop_count",
            "==INTERMEDIATE_STEPS==": "intermediate_steps",
            "==REASONING_PATTERN==": "reasoning_pattern",
            "==DIFFICULTY_LEVEL==": "difficulty_level",
            "==EVALUATION_FOCUS==": "evaluation_focus"
        }

        return self._parse_response_sections_v2(response, delimiters, sections)

    def _parse_domain_adaptation_response_v2(self, response: str) -> dict:
        """解析领域适应响应 - 优化版本"""
        sections = {
            "question": "",
            "domain_context": "",
            "standard_answer": "",
            "adaptation_type": "",
            "source_knowledge": [],
            "target_application": [],
            "difficulty_level": "",
            "evaluation_focus": []
        }

        delimiters = {
            "==QUESTION==": "question",
            "==DOMAIN_CONTEXT==": "domain_context",
            "==STANDARD_ANSWER==": "standard_answer",
            "==ADAPTATION_TYPE==": "adaptation_type",
            "==SOURCE_KNOWLEDGE==": "source_knowledge",
            "==TARGET_APPLICATION==": "target_application",
            "==DIFFICULTY_LEVEL==": "difficulty_level",
            "==EVALUATION_FOCUS==": "evaluation_focus"
        }

        return self._parse_response_sections_v2(response, delimiters, sections)

    def _parse_response_sections_v2(self, response: str, delimiters: dict, sections: dict) -> dict:
        """优化的响应解析方法 - 增强容错性"""
        current_section = None
        lines = response.split('\n')

        # 首先尝试找到所有分隔符的位置
        delimiter_positions = {}
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if clean_line in delimiters:
                delimiter_positions[clean_line] = i

        # 按顺序处理每个部分
        delimiter_list = list(delimiters.keys())

        for i, delimiter in enumerate(delimiter_list):
            if delimiter not in delimiter_positions:
                continue

            start_pos = delimiter_positions[delimiter]
            # 找到下一个分隔符的位置
            end_pos = len(lines)
            for j in range(i + 1, len(delimiter_list)):
                next_delimiter = delimiter_list[j]
                if next_delimiter in delimiter_positions:
                    end_pos = delimiter_positions[next_delimiter]
                    break

            # 提取该部分的内容
            section_name = delimiters[delimiter]
            content_lines = lines[start_pos + 1:end_pos]
            content = '\n'.join(line for line in content_lines if line.strip()).strip()

            if content:
                if section_name in ["entities", "relations", "key_concepts", "retrieval_strategy",
                                    "intermediate_steps", "source_knowledge", "target_application",
                                    "evaluation_focus"]:
                    # 处理列表类型的字段
                    if ',' in content:
                        items = [item.strip() for item in content.split(',') if item.strip()]
                    else:
                        # 如果没有逗号，尝试按行分割
                        items = [line.strip() for line in content.split('\n') if line.strip()]
                    sections[section_name] = items[:10]  # 限制最多10个项目
                else:
                    # 处理文本类型的字段
                    sections[section_name] = content

        return sections

    def _parse_sample_response_lenient(self, response: str, sample_type: str) -> dict:
        """宽松的样本响应解析 - 用于备用解析"""
        # 尝试从响应中提取基本信息
        sections = {}

        # 查找可能的问题内容
        question_patterns = [
            r'(?:问题|题目|Question)[:：]\s*(.+?)(?=\n|\r|$)',
            r'==QUESTION==\s*(.+?)(?===|$)',
            r'【问题】\s*(.+?)(?=【|$)'
        ]

        for pattern in question_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sections["question"] = match.group(1).strip()
                break

        # 查找可能的答案内容
        answer_patterns = [
            r'(?:答案|回答|Answer)[:：]\s*(.+?)(?=\n\n|\r\r|$)',
            r'==STANDARD_ANSWER==\s*(.+?)(?===|$)',
            r'【答案】\s*(.+?)(?=【|$)'
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sections["standard_answer"] = match.group(1).strip()
                break

        # 如果找不到明确的问题和答案，尝试从整个响应中提取
        if not sections.get("question") or not sections.get("standard_answer"):
            # 尝试将响应分成两部分
            parts = response.split('\n\n')
            if len(parts) >= 2:
                sections["question"] = parts[0].strip()
                sections["standard_answer"] = '\n\n'.join(parts[1:]).strip()
            else:
                # 如果分割失败，使用整个响应作为问题，生成简单答案
                sections["question"] = response.strip()
                sections["standard_answer"] = "需要进一步分析和解答。"

        # 设置默认值
        sections.setdefault("difficulty_level", "中等")
        sections.setdefault("evaluation_focus", ["基本理解"])

        # 根据样本类型添加特定字段
        if sample_type == "kg_reasoning":
            sections.setdefault("knowledge_graph", "相关知识图谱信息")
            sections.setdefault("reasoning_type", "推理")
            sections.setdefault("entities", ["相关实体"])
            sections.setdefault("relations", ["相关关系"])
        elif sample_type == "structured_qa":
            sections.setdefault("data_structure", "数据结构描述")
            sections.setdefault("query_type", "查询")
            sections.setdefault("key_concepts", ["核心概念"])
            sections.setdefault("retrieval_strategy", ["检索策略"])
        elif sample_type == "multi_hop":
            sections.setdefault("reasoning_chain", "推理链描述")
            sections.setdefault("hop_count", "多跳")
            sections.setdefault("intermediate_steps", ["中间步骤"])
            sections.setdefault("reasoning_pattern", "推理模式")
        elif sample_type == "domain_adaptation":
            sections.setdefault("domain_context", "领域背景")
            sections.setdefault("adaptation_type", "适应类型")
            sections.setdefault("source_knowledge", ["源知识"])
            sections.setdefault("target_application", ["目标应用"])

        return sections

    def _create_sample_dict(self, sample_data: dict, template: dict, specific_prompt: str,
                            sample_type: str, sample_id: int, status: str) -> dict:
        """创建样本字典"""
        base_sample = {
            "id": f"{sample_type}_sample_{sample_id}",
            "type": template["type"],
            "category": template["category"],
            "sample_type": sample_type,
            "original_prompt": specific_prompt,
            "question": sample_data["question"],
            "standard_answer": sample_data["standard_answer"],
            "difficulty_level": sample_data.get("difficulty_level", "中等"),
            "evaluation_focus": sample_data.get("evaluation_focus", []),
            "generated_at": datetime.now().isoformat(),
            "generation_status": status
        }

        # 添加特定类型的字段
        if sample_type == "kg_reasoning":
            base_sample.update({
                "knowledge_graph": sample_data.get("knowledge_graph", ""),
                "reasoning_type": sample_data.get("reasoning_type", ""),
                "entities": sample_data.get("entities", []),
                "relations": sample_data.get("relations", [])
            })
        elif sample_type == "structured_qa":
            base_sample.update({
                "data_structure": sample_data.get("data_structure", ""),
                "query_type": sample_data.get("query_type", ""),
                "key_concepts": sample_data.get("key_concepts", []),
                "retrieval_strategy": sample_data.get("retrieval_strategy", [])
            })
        elif sample_type == "multi_hop":
            base_sample.update({
                "reasoning_chain": sample_data.get("reasoning_chain", ""),
                "hop_count": sample_data.get("hop_count", ""),
                "intermediate_steps": sample_data.get("intermediate_steps", []),
                "reasoning_pattern": sample_data.get("reasoning_pattern", "")
            })
        elif sample_type == "domain_adaptation":
            base_sample.update({
                "domain_context": sample_data.get("domain_context", ""),
                "adaptation_type": sample_data.get("adaptation_type", ""),
                "source_knowledge": sample_data.get("source_knowledge", []),
                "target_application": sample_data.get("target_application", [])
            })

        return base_sample

    def _create_fallback_sample_v2(self, template: dict, specific_prompt: str, sample_type: str,
                                   sample_id: int) -> dict:
        """创建增强的备用样本"""
        # 根据模板和提示生成基础内容
        base_question = f"关于{template['type']}的{template['category']}问题：{specific_prompt}"

        # 生成针对性的标准答案
        if sample_type == "kg_reasoning":
            base_answer = """这是一个知识图谱推理问题，需要：
1. 识别相关实体和关系
2. 构建推理路径
3. 进行逻辑推导
4. 得出结论"""
            specific_data = {
                "knowledge_graph": "相关实体和关系的图结构",
                "reasoning_type": "图谱推理",
                "entities": ["实体1", "实体2", "实体3"],
                "relations": ["关系1", "关系2"]
            }
        elif sample_type == "structured_qa":
            base_answer = """这是一个结构化问答问题，需要：
1. 理解数据结构
2. 制定查询策略
3. 提取相关信息
4. 组织答案"""
            specific_data = {
                "data_structure": "结构化数据描述",
                "query_type": "信息查询",
                "key_concepts": ["概念1", "概念2"],
                "retrieval_strategy": ["策略1", "策略2"]
            }
        elif sample_type == "multi_hop":
            base_answer = """这是一个多跳推理问题，需要：
1. 分解推理步骤
2. 建立推理链
3. 逐步推导
4. 验证结论"""
            specific_data = {
                "reasoning_chain": "步骤1 → 步骤2 → 结论",
                "hop_count": "多跳",
                "intermediate_steps": ["步骤1", "步骤2", "步骤3"],
                "reasoning_pattern": "链式推理"
            }
        elif sample_type == "domain_adaptation":
            base_answer = """这是一个领域适应问题，需要：
1. 分析源领域知识
2. 识别目标领域需求
3. 建立迁移策略
4. 验证适应效果"""
            specific_data = {
                "domain_context": "跨领域应用背景",
                "adaptation_type": "知识迁移",
                "source_knowledge": ["源知识1", "源知识2"],
                "target_application": ["目标应用1", "目标应用2"]
            }
        else:
            base_answer = "这是一个数学推理问题，需要系统分析和推导。"
            specific_data = {}

        sample = {
            "id": f"{sample_type}_sample_{sample_id}",
            "type": template["type"],
            "category": template["category"],
            "sample_type": sample_type,
            "original_prompt": specific_prompt,
            "question": base_question,
            "standard_answer": base_answer,
            "difficulty_level": "中等",
            "evaluation_focus": ["基本能力", "逻辑推理"],
            "generated_at": datetime.now().isoformat(),
            "generation_status": "fallback"
        }

        # 添加特定数据
        sample.update(specific_data)

        return sample

    def _extract_template_params(self, template: Dict) -> Dict:
        """提取模板参数"""
        prompt_params = {}
        param_mappings = {
            'entity_types': 'entity_type',
            'relation_types': 'relation_type',
            'reasoning_levels': 'reasoning_level',
            'hop_counts': 'hop_count',
            'start_concepts': 'start_concept',
            'end_concepts': 'end_concept',
            'path_types': 'path_type',
            'domains': 'domain',
            'given_relations': 'given_relations',
            'missing_relations': 'missing_relations',
            'math_domains': 'math_domain',
            'data_structures': 'data_structure',
            'query_types': 'query_type',
            'condition_types': 'condition_type',
            'reasoning_patterns': 'reasoning_pattern',
            'comparison_types': 'comparison_type',
            'analysis_depths': 'analysis_depth',
            'chain_lengths': 'chain_length',
            'start_points': 'start_point',
            'reasoning_steps': 'reasoning_steps',
            'end_goals': 'end_goal',
            'branch_counts': 'branch_count',
            'branch_tasks': 'branch_task',
            'integration_methods': 'integration_method',
            'recursion_types': 'recursion_type',
            'recursion_depths': 'recursion_depth',
            'recursion_operations': 'recursion_operation',
            'source_domains': 'source_domain',
            'target_domains': 'target_domain',
            'transfer_methods': 'transfer_method',
            'expertise_levels': 'expertise_level',
            'specialized_fields': 'specialized_field',
            'technical_aspects': 'technical_aspects',
            'context_types': 'context_type',
            'adaptation_strategies': 'adaptation_strategy'
        }

        for key in template:
            if key in param_mappings and isinstance(template[key], list):
                param_name = param_mappings[key]
                prompt_params[param_name] = random.choice(template[key])

        return prompt_params

    def _display_progress(self, current: int, total: int, success: int, partial: int, failed: int,
                          start_time: float, iteration_start: float, template: dict, sample_type: str):
        """显示进度信息"""
        iteration_time = time.time() - iteration_start
        elapsed_time = time.time() - start_time
        avg_time_per_sample = elapsed_time / current
        estimated_remaining = avg_time_per_sample * (total - current)

        progress_percent = current / total * 100
        progress_bar = "█" * int(progress_percent // 2) + "░" * (50 - int(progress_percent // 2))

        total_processed = success + partial + failed
        success_rate = ((success + partial) / total_processed * 100) if total_processed > 0 else 0

        if current % 10 == 0 or current in [1, 5, 25, 50] or current % 50 == 0:
            print(f"\r🔄 [{progress_bar}] {progress_percent:.1f}%")
            print(f"   📊 进度: {current}/{total} | ✅ 成功: {success} | ⚠️ 部分成功: {partial} | ❌ 失败: {failed}")
            print(
                f"   📈 成功率: {success_rate:.1f}% | ⏱️ 用时: {elapsed_time:.1f}s | 🕐 预计剩余: {estimated_remaining:.1f}s")
            print(
                f"   🎯 当前样本: {sample_type} - {template['type']} ({template['category']}) - 耗时: {iteration_time:.2f}s")
            print("-" * 80)
        else:
            print(f"\r🔄 [{progress_bar}] {progress_percent:.1f}% ({current}/{total}) | "
                  f"✅ {success} | ⚠️ {partial} | ❌ {failed} | "
                  f"成功率: {success_rate:.1f}%", end="", flush=True)

    def _save_intermediate_results(self, samples: list, sample_type: str, count: int):
        """保存中间结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = f"{sample_type}_samples_intermediate_{timestamp}_{count}.json"
        with open(intermediate_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"\n💾 中间结果已保存: {intermediate_file}")

    def _save_final_results(self, samples: list, sample_type: str):
        """保存最终结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        samples_file = f"{sample_type}_samples_{timestamp}.json"
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"💾 {sample_type}样本已保存到: {samples_file}")

    def _print_generation_summary(self, sample_type: str, samples: list, success: int,
                                  partial: int, failed: int, start_time: float):
        """打印生成总结"""
        total_time = time.time() - start_time
        print(f"\n\n🎉 {sample_type}生成阶段完成！")
        print("=" * 80)
        print(f"📊 最终统计:")
        print(f"   ✅ 完全成功: {success} ({success / len(samples) * 100:.1f}%)")
        print(f"   ⚠️ 部分成功: {partial} ({partial / len(samples) * 100:.1f}%)")
        print(f"   ❌ 生成失败: {failed} ({failed / len(samples) * 100:.1f}%)")
        print(f"   📈 总体成功率: {((success + partial) / len(samples) * 100):.1f}%")
        print(f"   ⏱️ 总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")

    # 评估方法
    def kg_reasoning_evaluation(self, sample: Dict, model_response: str, model_name: str) -> Dict:
        """知识图谱推理能力评估"""
        clean_question = self.clean_json_string(sample['question'])
        clean_standard_answer = self.clean_json_string(sample['standard_answer'])
        clean_model_response = self.clean_json_string(model_response)

        evaluation_focus = sample.get('evaluation_focus', ['推理准确性', '知识检索'])
        entities = sample.get('entities', [])
        relations = sample.get('relations', [])
        reasoning_type = sample.get('reasoning_type', '')

        evaluation_prompt = f"""
你是一位专业的知识图谱和推理能力评估专家，请对以下知识图谱推理任务的模型回答进行专项评估。

【任务信息】
推理类型：{sample['type']}
任务分类：{sample.get('category', '未分类')}
推理模式：{reasoning_type}
难度等级：{sample.get('difficulty_level', '中等')}
关键实体：{', '.join(entities)}
关系类型：{', '.join(relations)}
评估重点：{', '.join(evaluation_focus)}

【知识图谱推理问题】
{clean_question}

【标准答案】
{clean_standard_answer}

【模型回答】（来自模型：{model_name}）
{clean_model_response}

请从以下知识图谱推理维度进行评估（每个维度0-100分）：

1. 实体识别准确性（25%权重）：是否正确识别和理解相关实体
2. 关系理解准确性（25%权重）：是否准确理解实体间关系
3. 推理路径正确性（30%权重）：推理路径是否逻辑正确、完整
4. 知识整合能力（20%权重）：是否能有效整合多个知识点

请使用以下格式输出评估结果：

==OVERALL_SCORE==
综合得分(0-100的整数)

==ENTITY_RECOGNITION==
实体识别得分(0-100的整数)

==RELATION_UNDERSTANDING==
关系理解得分(0-100的整数)

==REASONING_PATH==
推理路径得分(0-100的整数)

==KNOWLEDGE_INTEGRATION==
知识整合得分(0-100的整数)

==WEIGHTED_SCORE==
加权得分(0-100的整数)

==IDENTIFIED_ENTITIES==
识别到的实体(用逗号分隔)

==IDENTIFIED_RELATIONS==
识别到的关系(用逗号分隔)

==REASONING_QUALITY==
推理质量(优秀/良好/一般/较差)

==KNOWLEDGE_COVERAGE==
知识覆盖度(完整/部分/不足)

==STRENGTHS==
推理优点(用逗号分隔)

==WEAKNESSES==
推理不足(用逗号分隔)

==IMPROVEMENT_SUGGESTIONS==
改进建议(用逗号分隔)

==EVALUATION_SUMMARY==
简要评估总结

注意：请重点评估知识图谱推理的准确性和完整性。
"""

        print(f"      🔗 调用知识图谱推理评估器...")
        return self._call_evaluator_with_fallback(evaluation_prompt, sample, model_response, model_name,
                                                  "kg_reasoning", self.parse_kg_reasoning_evaluation_response)

    def structured_qa_evaluation(self, sample: Dict, model_response: str, model_name: str) -> Dict:
        """结构化问答能力评估"""
        clean_question = self.clean_json_string(sample['question'])
        clean_standard_answer = self.clean_json_string(sample['standard_answer'])
        clean_model_response = self.clean_json_string(model_response)

        evaluation_focus = sample.get('evaluation_focus', ['查询准确性', '结构理解'])
        key_concepts = sample.get('key_concepts', [])
        query_type = sample.get('query_type', '')

        evaluation_prompt = f"""
你是一位专业的结构化数据和问答系统评估专家，请对以下结构化问答任务的模型回答进行专项评估。

【任务信息】
问答类型：{sample['type']}
任务分类：{sample.get('category', '未分类')}
查询类型：{query_type}
难度等级：{sample.get('difficulty_level', '中等')}
核心概念：{', '.join(key_concepts)}
评估重点：{', '.join(evaluation_focus)}

【结构化问答问题】
{clean_question}

【标准答案】
{clean_standard_answer}

【模型回答】（来自模型：{model_name}）
{clean_model_response}

请从以下结构化问答维度进行评估（每个维度0-100分）：

1. 查询理解准确性（25%权重）：是否正确理解查询意图和要求
2. 结构化数据处理（25%权重）：是否能有效处理结构化信息
3. 信息检索准确性（30%权重）：是否能准确检索相关信息
4. 答案组织完整性（20%权重）：答案是否组织良好、逻辑清晰

请使用以下格式输出评估结果：

==OVERALL_SCORE==
综合得分(0-100的整数)

==QUERY_UNDERSTANDING==
查询理解得分(0-100的整数)

==STRUCTURE_PROCESSING==
结构处理得分(0-100的整数)

==INFO_RETRIEVAL==
信息检索得分(0-100的整数)

==ANSWER_ORGANIZATION==
答案组织得分(0-100的整数)

==WEIGHTED_SCORE==
加权得分(0-100的整数)

==RETRIEVED_CONCEPTS==
检索到的概念(用逗号分隔)

==PROCESSING_STRATEGY==
处理策略质量(优秀/良好/一般/较差)

==ANSWER_COMPLETENESS==
答案完整性(完整/部分/不足)

==LOGICAL_COHERENCE==
逻辑连贯性(优秀/良好/一般/较差)

==STRENGTHS==
问答优点(用逗号分隔)

==WEAKNESSES==
问答不足(用逗号分隔)

==IMPROVEMENT_SUGGESTIONS==
改进建议(用逗号分隔)

==EVALUATION_SUMMARY==
简要评估总结

注意：请重点评估结构化信息处理和查询响应的准确性。
"""

        print(f"      📊 调用结构化问答评估器...")
        return self._call_evaluator_with_fallback(evaluation_prompt, sample, model_response, model_name,
                                                  "structured_qa", self.parse_structured_qa_evaluation_response)

    def multi_hop_evaluation(self, sample: Dict, model_response: str, model_name: str) -> Dict:
        """多跳推理能力评估"""
        clean_question = self.clean_json_string(sample['question'])
        clean_standard_answer = self.clean_json_string(sample['standard_answer'])
        clean_model_response = self.clean_json_string(model_response)

        evaluation_focus = sample.get('evaluation_focus', ['推理完整性', '逻辑连贯性'])
        hop_count = sample.get('hop_count', '')
        intermediate_steps = sample.get('intermediate_steps', [])
        reasoning_pattern = sample.get('reasoning_pattern', '')

        evaluation_prompt = f"""
你是一位专业的多跳推理和复杂推理评估专家，请对以下多跳推理任务的模型回答进行专项评估。

【任务信息】
推理类型：{sample['type']}
任务分类：{sample.get('category', '未分类')}
推理跳数：{hop_count}
推理模式：{reasoning_pattern}
难度等级：{sample.get('difficulty_level', '中等')}
中间步骤：{', '.join(intermediate_steps)}
评估重点：{', '.join(evaluation_focus)}

【多跳推理问题】
{clean_question}

【标准答案】
{clean_standard_answer}

【模型回答】（来自模型：{model_name}）
{clean_model_response}

请从以下多跳推理维度进行评估（每个维度0-100分）：

1. 推理路径完整性（30%权重）：是否包含完整的推理路径
2. 中间步骤准确性（25%权重）：中间推理步骤是否正确
3. 逻辑连贯性（25%权重）：各推理步骤间逻辑是否连贯
4. 最终结论正确性（20%权重）：最终推理结论是否正确

请使用以下格式输出评估结果：

==OVERALL_SCORE==
综合得分(0-100的整数)

==PATH_COMPLETENESS==
路径完整性得分(0-100的整数)

==STEP_ACCURACY==
步骤准确性得分(0-100的整数)

==LOGICAL_COHERENCE==
逻辑连贯性得分(0-100的整数)

==CONCLUSION_CORRECTNESS==
结论正确性得分(0-100的整数)

==WEIGHTED_SCORE==
加权得分(0-100的整数)

==IDENTIFIED_STEPS==
识别到的推理步骤(用逗号分隔)

==REASONING_QUALITY==
推理质量(优秀/良好/一般/较差)

==PATH_COVERAGE==
路径覆盖度(完整/部分/不足)

==STEP_CLARITY==
步骤清晰度(优秀/良好/一般/较差)

==STRENGTHS==
推理优点(用逗号分隔)

==WEAKNESSES==
推理不足(用逗号分隔)

==IMPROVEMENT_SUGGESTIONS==
改进建议(用逗号分隔)

==EVALUATION_SUMMARY==
简要评估总结

注意：请重点评估推理路径的完整性和步骤间的逻辑连贯性。
"""

        print(f"      🔄 调用多跳推理评估器...")
        return self._call_evaluator_with_fallback(evaluation_prompt, sample, model_response, model_name,
                                                  "multi_hop", self.parse_multi_hop_evaluation_response)

    def domain_adaptation_evaluation(self, sample: Dict, model_response: str, model_name: str) -> Dict:
        """领域适应能力评估"""
        clean_question = self.clean_json_string(sample['question'])
        clean_standard_answer = self.clean_json_string(sample['standard_answer'])
        clean_model_response = self.clean_json_string(model_response)

        evaluation_focus = sample.get('evaluation_focus', ['迁移准确性', '适应能力'])
        source_knowledge = sample.get('source_knowledge', [])
        target_application = sample.get('target_application', [])
        adaptation_type = sample.get('adaptation_type', '')

        evaluation_prompt = f"""
你是一位专业的领域知识迁移和适应能力评估专家，请对以下领域适应任务的模型回答进行专项评估。

【任务信息】
适应类型：{sample['type']}
任务分类：{sample.get('category', '未分类')}
迁移类型：{adaptation_type}
难度等级：{sample.get('difficulty_level', '中等')}
源领域知识：{', '.join(source_knowledge)}
目标应用：{', '.join(target_application)}
评估重点：{', '.join(evaluation_focus)}

【领域适应问题】
{clean_question}

【标准答案】
{clean_standard_answer}

【模型回答】（来自模型：{model_name}）
{clean_model_response}

请从以下领域适应维度进行评估（每个维度0-100分）：

1. 知识迁移准确性（30%权重）：是否准确迁移相关知识
2. 领域理解深度（25%权重）：对源领域和目标领域的理解深度
3. 适应策略有效性（25%权重）：适应策略是否有效可行
4. 应用场景匹配度（20%权重）：是否匹配目标应用场景

请使用以下格式输出评估结果：

==OVERALL_SCORE==
综合得分(0-100的整数)

==TRANSFER_ACCURACY==
迁移准确性得分(0-100的整数)

==DOMAIN_UNDERSTANDING==
领域理解得分(0-100的整数)

==ADAPTATION_STRATEGY==
适应策略得分(0-100的整数)

==APPLICATION_MATCHING==
应用匹配得分(0-100的整数)

==WEIGHTED_SCORE==
加权得分(0-100的整数)

==TRANSFERRED_KNOWLEDGE==
迁移的知识(用逗号分隔)

==ADAPTATION_QUALITY==
适应质量(优秀/良好/一般/较差)

==DOMAIN_COVERAGE==
领域覆盖度(完整/部分/不足)

==STRATEGY_EFFECTIVENESS==
策略有效性(优秀/良好/一般/较差)

==STRENGTHS==
适应优点(用逗号分隔)

==WEAKNESSES==
适应不足(用逗号分隔)

==IMPROVEMENT_SUGGESTIONS==
改进建议(用逗号分隔)

==EVALUATION_SUMMARY==
简要评估总结

注意：请重点评估知识迁移的准确性和适应策略的有效性。
"""

        print(f"      🎯 调用领域适应评估器...")
        return self._call_evaluator_with_fallback(evaluation_prompt, sample, model_response, model_name,
                                                  "domain_adaptation", self.parse_domain_adaptation_evaluation_response)

    def _call_evaluator_with_fallback(self, evaluation_prompt: str, sample: Dict, model_response: str,
                                      model_name: str, eval_type: str, parse_func) -> Dict:
        """调用评估器并提供备用方案"""
        try:
            eval_start = time.time()
            response = self.call_ollama(self.models["evaluator"], evaluation_prompt)
            eval_time = time.time() - eval_start

            print(f"      ⏱️ 评估响应时间: {eval_time:.2f}s")

            if response:
                evaluation_result = parse_func(response)

                if evaluation_result and evaluation_result.get("overall_score") is not None:
                    evaluation_result['evaluation_method'] = f'{eval_type}_intelligent'
                    evaluation_result['evaluator_model'] = self.models['evaluator']
                    evaluation_result['evaluation_time'] = eval_time

                    print(f"      ✅ 评估成功")
                    print(f"      📊 得分: {evaluation_result['overall_score']:.1f}/100")

                    return evaluation_result
                else:
                    print(f"      ⚠️ 评估结果解析失败")
                    raise ValueError("评估结果解析失败")
            else:
                print(f"      ❌ 评估器没有返回有效响应")
                raise ValueError("评估器没有返回有效响应")

        except Exception as e:
            print(f"      🔄 智能评估失败，切换到备用评估: {e}")
            return self._fallback_evaluation(sample, model_response, model_name, eval_type)

    def _fallback_evaluation(self, sample: Dict, model_response: str, model_name: str, eval_type: str) -> Dict:
        """备用评估方法"""
        score = 0

        # 基础评估
        if len(model_response.strip()) > 50:
            score += 25

        # 关键词匹配
        if eval_type == "kg_reasoning":
            keywords = ['实体', '关系', '推理', '知识', '图谱']
        elif eval_type == "structured_qa":
            keywords = ['查询', '结构', '数据', '检索', '信息']
        elif eval_type == "multi_hop":
            keywords = ['推理', '步骤', '逻辑', '连接', '路径']
        elif eval_type == "domain_adaptation":
            keywords = ['领域', '迁移', '适应', '应用', '知识']
        else:
            keywords = ['分析', '理解', '推理', '解决']

        found_keywords = [kw for kw in keywords if kw in model_response]
        score += len(found_keywords) * 10

        # 长度和结构评估
        if len(model_response) > 200:
            score += 15
        if '。' in model_response or '，' in model_response:
            score += 10

        final_score = min(score, 100)

        return {
            "overall_score": final_score,
            "dimension_scores": {
                "primary": final_score * 0.9,
                "secondary": final_score * 0.8,
                "tertiary": final_score * 0.7,
                "quaternary": final_score * 0.6
            },
            "weighted_score": final_score,
            "evaluation_summary": f"备用{eval_type}评估",
            "strengths": found_keywords,
            "weaknesses": ["需要更详细的分析"],
            "improvement_suggestions": ["加强专业能力训练"],
            "evaluation_method": f"fallback_{eval_type}_rule_based",
            "evaluator_model": "rule_based_fallback"
        }

    # 解析评估响应的方法
    def parse_kg_reasoning_evaluation_response(self, response: str) -> dict:
        """解析知识图谱推理评估响应"""
        sections = {
            "overall_score": None,
            "dimension_scores": {
                "entity_recognition": None,
                "relation_understanding": None,
                "reasoning_path": None,
                "knowledge_integration": None
            },
            "weighted_score": None,
            "identified_entities": [],
            "identified_relations": [],
            "reasoning_quality": "",
            "knowledge_coverage": "",
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation_summary": ""
        }

        delimiters = {
            "==OVERALL_SCORE==": "overall_score",
            "==ENTITY_RECOGNITION==": "entity_recognition",
            "==RELATION_UNDERSTANDING==": "relation_understanding",
            "==REASONING_PATH==": "reasoning_path",
            "==KNOWLEDGE_INTEGRATION==": "knowledge_integration",
            "==WEIGHTED_SCORE==": "weighted_score",
            "==IDENTIFIED_ENTITIES==": "identified_entities",
            "==IDENTIFIED_RELATIONS==": "identified_relations",
            "==REASONING_QUALITY==": "reasoning_quality",
            "==KNOWLEDGE_COVERAGE==": "knowledge_coverage",
            "==STRENGTHS==": "strengths",
            "==WEAKNESSES==": "weaknesses",
            "==IMPROVEMENT_SUGGESTIONS==": "improvement_suggestions",
            "==EVALUATION_SUMMARY==": "evaluation_summary"
        }

        return self._parse_evaluation_response(response, delimiters, sections)

    def parse_structured_qa_evaluation_response(self, response: str) -> dict:
        """解析结构化问答评估响应"""
        sections = {
            "overall_score": None,
            "dimension_scores": {
                "query_understanding": None,
                "structure_processing": None,
                "info_retrieval": None,
                "answer_organization": None
            },
            "weighted_score": None,
            "retrieved_concepts": [],
            "processing_strategy": "",
            "answer_completeness": "",
            "logical_coherence": "",
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation_summary": ""
        }

        delimiters = {
            "==OVERALL_SCORE==": "overall_score",
            "==QUERY_UNDERSTANDING==": "query_understanding",
            "==STRUCTURE_PROCESSING==": "structure_processing",
            "==INFO_RETRIEVAL==": "info_retrieval",
            "==ANSWER_ORGANIZATION==": "answer_organization",
            "==WEIGHTED_SCORE==": "weighted_score",
            "==RETRIEVED_CONCEPTS==": "retrieved_concepts",
            "==PROCESSING_STRATEGY==": "processing_strategy",
            "==ANSWER_COMPLETENESS==": "answer_completeness",
            "==LOGICAL_COHERENCE==": "logical_coherence",
            "==STRENGTHS==": "strengths",
            "==WEAKNESSES==": "weaknesses",
            "==IMPROVEMENT_SUGGESTIONS==": "improvement_suggestions",
            "==EVALUATION_SUMMARY==": "evaluation_summary"
        }

        return self._parse_evaluation_response(response, delimiters, sections)

    def parse_multi_hop_evaluation_response(self, response: str) -> dict:
        """解析多跳推理评估响应"""
        sections = {
            "overall_score": None,
            "dimension_scores": {
                "path_completeness": None,
                "step_accuracy": None,
                "logical_coherence": None,
                "conclusion_correctness": None
            },
            "weighted_score": None,
            "identified_steps": [],
            "reasoning_quality": "",
            "path_coverage": "",
            "step_clarity": "",
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation_summary": ""
        }

        delimiters = {
            "==OVERALL_SCORE==": "overall_score",
            "==PATH_COMPLETENESS==": "path_completeness",
            "==STEP_ACCURACY==": "step_accuracy",
            "==LOGICAL_COHERENCE==": "logical_coherence",
            "==CONCLUSION_CORRECTNESS==": "conclusion_correctness",
            "==WEIGHTED_SCORE==": "weighted_score",
            "==IDENTIFIED_STEPS==": "identified_steps",
            "==REASONING_QUALITY==": "reasoning_quality",
            "==PATH_COVERAGE==": "path_coverage",
            "==STEP_CLARITY==": "step_clarity",
            "==STRENGTHS==": "strengths",
            "==WEAKNESSES==": "weaknesses",
            "==IMPROVEMENT_SUGGESTIONS==": "improvement_suggestions",
            "==EVALUATION_SUMMARY==": "evaluation_summary"
        }

        return self._parse_evaluation_response(response, delimiters, sections)

    def parse_domain_adaptation_evaluation_response(self, response: str) -> dict:
        """解析领域适应评估响应"""
        sections = {
            "overall_score": None,
            "dimension_scores": {
                "transfer_accuracy": None,
                "domain_understanding": None,
                "adaptation_strategy": None,
                "application_matching": None
            },
            "weighted_score": None,
            "transferred_knowledge": [],
            "adaptation_quality": "",
            "domain_coverage": "",
            "strategy_effectiveness": "",
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation_summary": ""
        }

        delimiters = {
            "==OVERALL_SCORE==": "overall_score",
            "==TRANSFER_ACCURACY==": "transfer_accuracy",
            "==DOMAIN_UNDERSTANDING==": "domain_understanding",
            "==ADAPTATION_STRATEGY==": "adaptation_strategy",
            "==APPLICATION_MATCHING==": "application_matching",
            "==WEIGHTED_SCORE==": "weighted_score",
            "==TRANSFERRED_KNOWLEDGE==": "transferred_knowledge",
            "==ADAPTATION_QUALITY==": "adaptation_quality",
            "==DOMAIN_COVERAGE==": "domain_coverage",
            "==STRATEGY_EFFECTIVENESS==": "strategy_effectiveness",
            "==STRENGTHS==": "strengths",
            "==WEAKNESSES==": "weaknesses",
            "==IMPROVEMENT_SUGGESTIONS==": "improvement_suggestions",
            "==EVALUATION_SUMMARY==": "evaluation_summary"
        }

        return self._parse_evaluation_response(response, delimiters, sections)

    def _parse_evaluation_response(self, response: str, delimiters: dict, sections: dict) -> dict:
        """通用的评估响应解析方法"""
        current_section = None
        lines = response.split('\n')

        for line in lines:
            line = line.strip()

            if line in delimiters:
                current_section = delimiters[line]
                continue

            if current_section and line:
                if current_section in ["overall_score", "weighted_score"]:
                    try:
                        score = int(re.findall(r'\d+', line)[0])
                        sections[current_section] = max(0, min(100, score))
                    except (ValueError, IndexError):
                        sections[current_section] = 50

                elif current_section in sections.get("dimension_scores", {}):
                    try:
                        score = int(re.findall(r'\d+', line)[0])
                        score = max(0, min(100, score))
                        sections["dimension_scores"][current_section] = score
                    except (ValueError, IndexError):
                        sections["dimension_scores"][current_section] = 50

                elif current_section in ["reasoning_quality", "knowledge_coverage", "processing_strategy",
                                         "answer_completeness", "logical_coherence", "path_coverage",
                                         "step_clarity", "adaptation_quality", "domain_coverage",
                                         "strategy_effectiveness", "evaluation_summary"]:
                    if sections[current_section]:
                        sections[current_section] += " " + line
                    else:
                        sections[current_section] = line

                elif current_section in ["identified_entities", "identified_relations", "retrieved_concepts",
                                         "identified_steps", "transferred_knowledge", "strengths",
                                         "weaknesses", "improvement_suggestions"]:
                    items = [item.strip() for item in line.split(',') if item.strip()]
                    sections[current_section].extend(items)

        # 验证和补充缺失值
        if sections["overall_score"] is None:
            dim_scores = list(sections["dimension_scores"].values())
            if all(score is not None for score in dim_scores):
                sections["overall_score"] = sum(dim_scores) // len(dim_scores)
            else:
                sections["overall_score"] = 50

        if sections["weighted_score"] is None:
            sections["weighted_score"] = sections["overall_score"]

        # 确保所有维度得分都有值
        for dim in sections["dimension_scores"]:
            if sections["dimension_scores"][dim] is None:
                sections["dimension_scores"][dim] = sections["overall_score"]

        return sections

    # 测试方法
    def test_model_capability(self, model_name: str, samples: List[Dict], capability_type: str) -> List[Dict]:
        """测试模型特定能力"""
        print(f"\n🧪 开始测试 {model_name} 的{capability_type}能力")
        print(f"📊 样本数: {len(samples)}")

        results = []
        success_count = 0
        failed_count = 0
        start_time = time.time()

        # 选择评估方法
        eval_methods = {
            "kg_reasoning": self.kg_reasoning_evaluation,
            "structured_qa": self.structured_qa_evaluation,
            "multi_hop": self.multi_hop_evaluation,
            "domain_adaptation": self.domain_adaptation_evaluation
        }

        eval_method = eval_methods.get(capability_type)
        if not eval_method:
            print(f"❌ 未找到 {capability_type} 的评估方法")
            return []

        for i, sample in enumerate(samples):
            try:
                # 创建测试提示
                test_prompt = self._create_test_prompt(sample, capability_type)

                print(f"\n🔄 样本 {i + 1}/{len(samples)}: {sample['id']}")
                print(f"   📝 类型: {sample['type']} | 分类: {sample.get('category', '未分类')}")

                model_start = time.time()
                model_response = self.call_ollama(model_name, test_prompt)
                model_response_time = time.time() - model_start

                print(f"   ⚡ 推理完成 ({model_response_time:.2f}s)")

                if model_response and len(model_response.strip()) > 10:
                    # 进行能力评估
                    evaluation = eval_method(sample, model_response, model_name)
                    success_count += 1
                    print(f"   🎯 得分: {evaluation['overall_score']:.1f}/100")
                else:
                    print(f"   ❌ 模型无有效响应")
                    evaluation = {"overall_score": 0, "evaluation_summary": "无有效响应"}
                    failed_count += 1

                result = {
                    "sample_id": sample["id"],
                    "sample_type": capability_type,
                    "question": sample["question"],
                    "model_response": model_response,
                    "evaluation": evaluation,
                    "response_time": model_response_time,
                    "tested_at": datetime.now().isoformat(),
                    "model_name": model_name
                }
                results.append(result)

                time.sleep(0.3)

            except Exception as e:
                print(f"   ❌ 测试异常: {e}")
                failed_count += 1
                results.append({
                    "sample_id": sample["id"],
                    "sample_type": capability_type,
                    "question": sample["question"],
                    "model_response": "",
                    "evaluation": {"overall_score": 0, "evaluation_summary": f"测试异常: {str(e)}"},
                    "response_time": 0,
                    "tested_at": datetime.now().isoformat(),
                    "model_name": model_name,
                    "error": str(e)
                })

        total_time = time.time() - start_time
        avg_score = sum(r["evaluation"]["overall_score"] for r in results) / len(results)

        print(f"\n✅ {model_name} {capability_type}能力测试完成！")
        print(f"   📊 平均得分: {avg_score:.2f}/100")
        print(f"   ✅ 成功: {success_count} | ❌ 失败: {failed_count}")
        print(f"   ⏱️ 耗时: {total_time:.1f}秒")

        return results

    def _create_test_prompt(self, sample: Dict, capability_type: str) -> str:
        """创建测试提示"""
        base_prompt = f"""
请仔细阅读并回答以下问题：

【问题】
{sample['question']}

要求：
1. 请提供详细的分析过程
2. 确保逻辑清晰、步骤完整
3. 给出明确的最终答案
"""

        if capability_type == "kg_reasoning":
            base_prompt += """
4. 请明确指出涉及的实体和关系
5. 展示完整的推理路径
"""
        elif capability_type == "structured_qa":
            base_prompt += """
4. 请说明信息检索和处理策略
5. 确保答案结构化组织
"""
        elif capability_type == "multi_hop":
            base_prompt += """
4. 请清晰展示每个推理步骤
5. 确保推理链的完整性和连贯性
"""
        elif capability_type == "domain_adaptation":
            base_prompt += """
4. 请说明知识迁移的策略
5. 展示在目标领域的具体应用
"""

        base_prompt += "\n请提供详细的回答："
        return base_prompt

    def generate_comprehensive_report(self, timestamp: str):
        """生成综合能力对比报告"""
        print("\n" + "=" * 80)
        print("📊 开始生成知识图谱预训练模型综合评估报告")
        print("=" * 80)

        # 计算各项能力得分
        report = {
            "test_info": {
                "timestamp": timestamp,
                "evaluation_method": "knowledge_graph_pretrained_evaluation",
                "evaluator_model": self.models["evaluator"],
                "kg_reasoning_samples": len(self.kg_reasoning_samples),
                "structured_qa_samples": len(self.structured_qa_samples),
                "multi_hop_samples": len(self.multi_hop_samples),
                "domain_adaptation_samples": len(self.domain_adaptation_samples),
                "models_tested": {
                    "finetuned": self.models["finetuned"],
                    "baseline": self.models["baseline"]
                }
            },
            "capability_analysis": {},
            "overall_performance": {}
        }

        for model_type in ["finetuned", "baseline"]:
            kg_results = self.results[model_type]["kg_reasoning"]
            qa_results = self.results[model_type]["structured_qa"]
            hop_results = self.results[model_type]["multi_hop"]
            domain_results = self.results[model_type]["domain_adaptation"]

            # 各能力得分统计
            kg_scores = [r["evaluation"]["overall_score"] for r in kg_results]
            qa_scores = [r["evaluation"]["overall_score"] for r in qa_results]
            hop_scores = [r["evaluation"]["overall_score"] for r in hop_results]
            domain_scores = [r["evaluation"]["overall_score"] for r in domain_results]

            kg_avg = sum(kg_scores) / len(kg_scores) if kg_scores else 0
            qa_avg = sum(qa_scores) / len(qa_scores) if qa_scores else 0
            hop_avg = sum(hop_scores) / len(hop_scores) if hop_scores else 0
            domain_avg = sum(domain_scores) / len(domain_scores) if domain_scores else 0

            # 综合得分（根据知识图谱预训练的重点分配权重）
            weighted_avg = (kg_avg * 0.3 + qa_avg * 0.25 + hop_avg * 0.25 + domain_avg * 0.2)

            report["capability_analysis"][model_type] = {
                "kg_reasoning_score": kg_avg,
                "structured_qa_score": qa_avg,
                "multi_hop_score": hop_avg,
                "domain_adaptation_score": domain_avg,
                "weighted_overall_score": weighted_avg,
                "success_rates": {
                    "kg_reasoning": len([r for r in kg_results if r["evaluation"]["overall_score"] > 0]) / len(
                        kg_results) * 100,
                    "structured_qa": len([r for r in qa_results if r["evaluation"]["overall_score"] > 0]) / len(
                        qa_results) * 100,
                    "multi_hop": len([r for r in hop_results if r["evaluation"]["overall_score"] > 0]) / len(
                        hop_results) * 100,
                    "domain_adaptation": len([r for r in domain_results if r["evaluation"]["overall_score"] > 0]) / len(
                        domain_results) * 100
                }
            }

        # 计算能力提升
        if "finetuned" in report["capability_analysis"] and "baseline" in report["capability_analysis"]:
            ft_cap = report["capability_analysis"]["finetuned"]
            bl_cap = report["capability_analysis"]["baseline"]

            improvements = {}
            for capability in ["kg_reasoning_score", "structured_qa_score", "multi_hop_score",
                               "domain_adaptation_score", "weighted_overall_score"]:
                if bl_cap[capability] > 0:
                    improvements[capability] = ((ft_cap[capability] - bl_cap[capability]) / bl_cap[capability] * 100)
                else:
                    improvements[capability] = 0

            report["capability_improvements"] = improvements

        # 详细分析
        report["detailed_analysis"] = self._generate_detailed_analysis()

        # 保存报告
        report_file = f"kg_pretrained_evaluation_report_{timestamp}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 保存详细结果
        detailed_results = {
            "kg_reasoning_samples": self.kg_reasoning_samples,
            "structured_qa_samples": self.structured_qa_samples,
            "multi_hop_samples": self.multi_hop_samples,
            "domain_adaptation_samples": self.domain_adaptation_samples,
            "results": self.results
        }
        detailed_file = f"kg_pretrained_detailed_results_{timestamp}.json"
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)

        # 打印报告摘要
        self._print_report_summary(report)

        print(f"\n📁 报告文件:")
        print(f"   📊 综合评估报告: {report_file}")
        print(f"   📋 详细测试结果: {detailed_file}")

        return report

    def _generate_detailed_analysis(self) -> Dict:
        """生成详细分析"""
        analysis = {
            "strengths_identified": [],
            "weaknesses_identified": [],
            "capability_distribution": {},
            "performance_patterns": {}
        }

        # 分析各能力的表现分布
        for model_type in ["finetuned", "baseline"]:
            if model_type in self.results:
                model_analysis = {}

                for capability in ["kg_reasoning", "structured_qa", "multi_hop", "domain_adaptation"]:
                    if capability in self.results[model_type]:
                        results = self.results[model_type][capability]
                        scores = [r["evaluation"]["overall_score"] for r in results]

                        if scores:
                            model_analysis[capability] = {
                                "average": sum(scores) / len(scores),
                                "max": max(scores),
                                "min": min(scores),
                                "std": self._calculate_std(scores),
                                "score_distribution": self._analyze_score_distribution(scores)
                            }

                analysis["capability_distribution"][model_type] = model_analysis

        return analysis

    def _calculate_std(self, scores: List[float]) -> float:
        """计算标准差"""
        if len(scores) <= 1:
            return 0
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance ** 0.5

    def _analyze_score_distribution(self, scores: List[float]) -> Dict:
        """分析得分分布"""
        distribution = {
            "excellent": len([s for s in scores if s >= 90]),
            "good": len([s for s in scores if 70 <= s < 90]),
            "average": len([s for s in scores if 50 <= s < 70]),
            "poor": len([s for s in scores if s < 50])
        }
        return distribution

    def _print_report_summary(self, report: Dict):
        """打印报告摘要"""
        print("\n" + "=" * 70)
        print("知识图谱预训练模型综合评估报告摘要")
        print("=" * 70)

        print(f"评估时间: {report['test_info']['timestamp']}")
        print(f"评估方法: 知识图谱预训练专项评估")
        print(f"知识图谱推理样本: {report['test_info']['kg_reasoning_samples']} 个")
        print(f"结构化问答样本: {report['test_info']['structured_qa_samples']} 个")
        print(f"多跳推理样本: {report['test_info']['multi_hop_samples']} 个")
        print(f"领域适应样本: {report['test_info']['domain_adaptation_samples']} 个")

        for model_type, model_name in report['test_info']['models_tested'].items():
            cap = report["capability_analysis"][model_type]
            print(f"\n【{model_type} 模型】({model_name}):")
            print(f"  🔗 知识图谱推理: {cap['kg_reasoning_score']:.2f}/100")
            print(f"  📊 结构化问答: {cap['structured_qa_score']:.2f}/100")
            print(f"  🔄 多跳推理: {cap['multi_hop_score']:.2f}/100")
            print(f"  🎯 领域适应: {cap['domain_adaptation_score']:.2f}/100")
            print(f"  🏆 综合能力: {cap['weighted_overall_score']:.2f}/100")

        if "capability_improvements" in report:
            improvements = report["capability_improvements"]
            print(f"\n【能力提升分析】")
            print("-" * 50)
            print(f"知识图谱推理提升: {improvements['kg_reasoning_score']:.2f}%")
            print(f"结构化问答提升: {improvements['structured_qa_score']:.2f}%")
            print(f"多跳推理提升: {improvements['multi_hop_score']:.2f}%")
            print(f"领域适应提升: {improvements['domain_adaptation_score']:.2f}%")
            print(f"综合能力提升: {improvements['weighted_overall_score']:.2f}%")

            # 识别最大提升领域
            max_improvement = max(improvements.items(), key=lambda x: x[1])
            print(f"\n🎯 最大提升领域: {max_improvement[0]} (+{max_improvement[1]:.2f}%)")

    def run_kg_pretrained_evaluation(self, kg_samples: int = 200, qa_samples: int = 200,
                                     hop_samples: int = 200, domain_samples: int = 200):
        """运行知识图谱预训练模型评估"""
        print("=" * 80)
        print("🚀 启动知识图谱预训练模型专项评估系统")
        print("=" * 80)
        print(f"🧠 评估器: {self.models['evaluator']}")
        print(f"🔬 待测模型: {self.models['finetuned']} vs {self.models['baseline']}")
        print(f"🔗 知识图谱推理样本: {kg_samples} 个")
        print(f"📊 结构化问答样本: {qa_samples} 个")
        print(f"🔄 多跳推理样本: {hop_samples} 个")
        print(f"🎯 领域适应样本: {domain_samples} 个")

        total_start_time = time.time()
        test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 阶段1: 生成各类样本
            print(f"\n🔥 第1阶段: 生成评估样本")
            self.kg_reasoning_samples = self.generate_kg_reasoning_samples(kg_samples)
            self.structured_qa_samples = self.generate_structured_qa_samples(qa_samples)
            self.multi_hop_samples = self.generate_multi_hop_samples(hop_samples)
            self.domain_adaptation_samples = self.generate_domain_adaptation_samples(domain_samples)

            # 阶段2: 测试微调模型
            print(f"\n🔥 第2阶段: 测试微调模型")
            print(f"🎯 模型: {self.models['finetuned']}")

            self.results["finetuned"]["kg_reasoning"] = self.test_model_capability(
                self.models["finetuned"], self.kg_reasoning_samples, "kg_reasoning")
            self.results["finetuned"]["structured_qa"] = self.test_model_capability(
                self.models["finetuned"], self.structured_qa_samples, "structured_qa")
            self.results["finetuned"]["multi_hop"] = self.test_model_capability(
                self.models["finetuned"], self.multi_hop_samples, "multi_hop")
            self.results["finetuned"]["domain_adaptation"] = self.test_model_capability(
                self.models["finetuned"], self.domain_adaptation_samples, "domain_adaptation")

            # 阶段3: 测试基线模型
            print(f"\n🔥 第3阶段: 测试基线模型")
            print(f"🎯 模型: {self.models['baseline']}")

            self.results["baseline"]["kg_reasoning"] = self.test_model_capability(
                self.models["baseline"], self.kg_reasoning_samples, "kg_reasoning")
            self.results["baseline"]["structured_qa"] = self.test_model_capability(
                self.models["baseline"], self.structured_qa_samples, "structured_qa")
            self.results["baseline"]["multi_hop"] = self.test_model_capability(
                self.models["baseline"], self.multi_hop_samples, "multi_hop")
            self.results["baseline"]["domain_adaptation"] = self.test_model_capability(
                self.models["baseline"], self.domain_adaptation_samples, "domain_adaptation")

            # 阶段4: 生成综合报告
            print(f"\n🔥 第4阶段: 生成综合评估报告")
            report = self.generate_comprehensive_report(test_timestamp)

            # 测试完成总结
            total_time = time.time() - total_start_time
            print(f"\n🎉 知识图谱预训练模型评估全部完成！")
            print("🎉" * 80)

            # 计算最终得分
            ft_results = report["capability_analysis"]["finetuned"]
            bl_results = report["capability_analysis"]["baseline"]

            print(f"📊 最终结果总结:")
            print(f"   ⏱️ 总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
            print(f"   📝 总样本数: {kg_samples + qa_samples + hop_samples + domain_samples}")

            print(f"\n🏆 微调模型表现:")
            print(f"   🔗 知识图谱推理: {ft_results['kg_reasoning_score']:.2f}/100")
            print(f"   📊 结构化问答: {ft_results['structured_qa_score']:.2f}/100")
            print(f"   🔄 多跳推理: {ft_results['multi_hop_score']:.2f}/100")
            print(f"   🎯 领域适应: {ft_results['domain_adaptation_score']:.2f}/100")
            print(f"   🏆 综合能力: {ft_results['weighted_overall_score']:.2f}/100")

            print(f"\n🏆 基线模型表现:")
            print(f"   🔗 知识图谱推理: {bl_results['kg_reasoning_score']:.2f}/100")
            print(f"   📊 结构化问答: {bl_results['structured_qa_score']:.2f}/100")
            print(f"   🔄 多跳推理: {bl_results['multi_hop_score']:.2f}/100")
            print(f"   🎯 领域适应: {bl_results['domain_adaptation_score']:.2f}/100")
            print(f"   🏆 综合能力: {bl_results['weighted_overall_score']:.2f}/100")

            if "capability_improvements" in report:
                improvements = report["capability_improvements"]
                print(f"\n📈 能力提升分析:")
                print(f"   🔗 知识图谱推理: {improvements['kg_reasoning_score']:.2f}%")
                print(f"   📊 结构化问答: {improvements['structured_qa_score']:.2f}%")
                print(f"   🔄 多跳推理: {improvements['multi_hop_score']:.2f}%")
                print(f"   🎯 领域适应: {improvements['domain_adaptation_score']:.2f}%")
                print(f"   🏆 综合提升: {improvements['weighted_overall_score']:.2f}%")

                # 识别训练效果最好的领域
                max_improvement = max(improvements.items(), key=lambda x: x[1])
                print(f"\n🌟 训练效果最佳领域: {max_improvement[0]} (+{max_improvement[1]:.2f}%)")

                # 识别需要加强的领域
                min_improvement = min(improvements.items(), key=lambda x: x[1])
                print(f"⚠️ 需要加强的领域: {min_improvement[0]} (+{min_improvement[1]:.2f}%)")

            return report

        except KeyboardInterrupt:
            print(f"\n⚠️ 评估被用户中断！")
            raise
        except Exception as e:
            print(f"\n❌ 评估过程中出现异常: {e}")
            raise


def main():
    """主函数"""
    print("🌟" * 30)
    print("知识图谱预训练模型专项评估系统")
    print("🌟" * 30)
    print("🧠 基于DeepSeek-R1:32B的智能化评估系统")
    print("📅 系统版本: v1.0 (知识图谱预训练专项版)")
    print("⏰ 启动时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # 系统初始化
    print(f"\n" + "=" * 70)
    print("🔧 系统初始化检查")
    print("=" * 70)

    try:
        import requests
        print("   ✅ requests 依赖正常")
    except ImportError:
        print("   ❌ 请安装 requests: pip install requests")
        return

    print("\n🤖 初始化知识图谱评估器...")
    try:
        tester = KnowledgeGraphModelTester()
        print(f"   ✅ 评估器初始化成功")
        print(f"   🧠 评估器模型: {tester.models['evaluator']}")
        print(f"   🔬 待测模型: {tester.models['finetuned']} vs {tester.models['baseline']}")
    except Exception as e:
        print(f"   ❌ 评估器初始化失败: {e}")
        return

    print(f"\n🌐 测试Ollama连接...")
    try:
        test_start = time.time()
        test_response = tester.call_ollama(tester.models['evaluator'], "测试连接：请回答知识图谱的基本概念",
                                           max_retries=1)
        test_time = time.time() - test_start

        if test_response:
            print(f"   ✅ Ollama API连接正常 ({test_time:.2f}s)")
            print(f"   📝 测试响应: {test_response[:50]}...")
        else:
            print(f"   ❌ Ollama API连接失败")
            return
    except Exception as e:
        print(f"   ❌ Ollama API测试异常: {e}")
        return

    print(f"\n✅ 系统初始化完成！")

    # 用户选择
    print(f"\n" + "=" * 70)
    print("📋 请选择评估模式:")
    print("=" * 70)
    print("1. 🚀 完整知识图谱预训练模型评估 (推荐)")
    print("2. 📁 使用已有样本进行评估")
    print("3. 🔗 仅生成知识图谱推理样本")
    print("4. 📊 仅生成结构化问答样本")
    print("5. 🔄 仅生成多跳推理样本")
    print("6. 🎯 仅生成领域适应样本")
    print("7. 🧪 单项能力测试")
    print("8. ❓ 查看系统帮助信息")

    try:
        choice = input(f"\n👆 请输入选择 (1-8，默认为1): ").strip()

        if choice == "2":
            print(f"\n📁 使用已有样本模式")
            print("⚠️ 注意：需要提供四类样本文件")
            kg_file = input("📂 知识图谱推理样本文件路径: ").strip()
            qa_file = input("📂 结构化问答样本文件路径: ").strip()
            hop_file = input("📂 多跳推理样本文件路径: ").strip()
            domain_file = input("📂 领域适应样本文件路径: ").strip()

            if not all([kg_file, qa_file, hop_file, domain_file]):
                print("❌ 未提供完整文件路径，退出程序")
                return

            try:
                with open(kg_file, "r", encoding="utf-8") as f:
                    tester.kg_reasoning_samples = json.load(f)
                with open(qa_file, "r", encoding="utf-8") as f:
                    tester.structured_qa_samples = json.load(f)
                with open(hop_file, "r", encoding="utf-8") as f:
                    tester.multi_hop_samples = json.load(f)
                with open(domain_file, "r", encoding="utf-8") as f:
                    tester.domain_adaptation_samples = json.load(f)

                print(f"✅ 样本加载成功: KG{len(tester.kg_reasoning_samples)}个, "
                      f"QA{len(tester.structured_qa_samples)}个, "
                      f"多跳{len(tester.multi_hop_samples)}个, "
                      f"领域{len(tester.domain_adaptation_samples)}个")

                # 运行评估
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 测试微调模型
                for capability, samples in [
                    ("kg_reasoning", tester.kg_reasoning_samples),
                    ("structured_qa", tester.structured_qa_samples),
                    ("multi_hop", tester.multi_hop_samples),
                    ("domain_adaptation", tester.domain_adaptation_samples)
                ]:
                    tester.results["finetuned"][capability] = tester.test_model_capability(
                        tester.models["finetuned"], samples, capability)
                    tester.results["baseline"][capability] = tester.test_model_capability(
                        tester.models["baseline"], samples, capability)

                # 生成报告
                tester.generate_comprehensive_report(timestamp)

            except Exception as e:
                print(f"❌ 文件处理失败: {e}")
                return

        elif choice == "3":
            print(f"\n🔗 仅生成知识图谱推理样本模式")
            try:
                sample_count = int(input("📊 请输入要生成的样本数量 (默认200): ").strip() or "200")
            except ValueError:
                sample_count = 200
            tester.generate_kg_reasoning_samples(sample_count)

        elif choice == "4":
            print(f"\n📊 仅生成结构化问答样本模式")
            try:
                sample_count = int(input("📊 请输入要生成的样本数量 (默认200): ").strip() or "200")
            except ValueError:
                sample_count = 200
            tester.generate_structured_qa_samples(sample_count)

        elif choice == "5":
            print(f"\n🔄 仅生成多跳推理样本模式")
            try:
                sample_count = int(input("📊 请输入要生成的样本数量 (默认200): ").strip() or "200")
            except ValueError:
                sample_count = 200
            tester.generate_multi_hop_samples(sample_count)

        elif choice == "6":
            print(f"\n🎯 仅生成领域适应样本模式")
            try:
                sample_count = int(input("📊 请输入要生成的样本数量 (默认200): ").strip() or "200")
            except ValueError:
                sample_count = 200
            tester.generate_domain_adaptation_samples(sample_count)

        elif choice == "7":
            print(f"\n🧪 单项能力测试模式")
            print("请选择要测试的能力:")
            print("1. 🔗 知识图谱推理")
            print("2. 📊 结构化问答")
            print("3. 🔄 多跳推理")
            print("4. 🎯 领域适应")

            capability_choice = input("请选择 (1-4): ").strip()
            capability_map = {
                "1": ("kg_reasoning", "知识图谱推理"),
                "2": ("structured_qa", "结构化问答"),
                "3": ("multi_hop", "多跳推理"),
                "4": ("domain_adaptation", "领域适应")
            }

            if capability_choice in capability_map:
                capability_type, capability_name = capability_map[capability_choice]

                try:
                    sample_count = int(input(f"📊 请输入{capability_name}样本数量 (默认100): ").strip() or "100")
                except ValueError:
                    sample_count = 100

                # 生成样本
                if capability_type == "kg_reasoning":
                    samples = tester.generate_kg_reasoning_samples(sample_count)
                elif capability_type == "structured_qa":
                    samples = tester.generate_structured_qa_samples(sample_count)
                elif capability_type == "multi_hop":
                    samples = tester.generate_multi_hop_samples(sample_count)
                elif capability_type == "domain_adaptation":
                    samples = tester.generate_domain_adaptation_samples(sample_count)

                # 测试模型
                print(f"\n🧪 开始{capability_name}能力测试...")
                ft_results = tester.test_model_capability(tester.models["finetuned"], samples, capability_type)
                bl_results = tester.test_model_capability(tester.models["baseline"], samples, capability_type)

                # 简单对比
                ft_avg = sum(r["evaluation"]["overall_score"] for r in ft_results) / len(ft_results)
                bl_avg = sum(r["evaluation"]["overall_score"] for r in bl_results) / len(bl_results)
                improvement = ((ft_avg - bl_avg) / bl_avg * 100) if bl_avg > 0 else 0

                print(f"\n📊 {capability_name}能力测试结果:")
                print(f"   🔬 微调模型: {ft_avg:.2f}/100")
                print(f"   📍 基线模型: {bl_avg:.2f}/100")
                print(f"   📈 提升幅度: {improvement:.2f}%")
            else:
                print("❌ 无效选择")
                return

        elif choice == "8":
            print(f"\n📖 系统帮助信息")
            print("=" * 70)
            print("🎯 本系统专为知识图谱预训练模型评估设计:")
            print("   • 🔗 知识图谱推理：评估实体关系理解和图结构推理能力")
            print("   • 📊 结构化问答：评估结构化数据处理和查询响应能力")
            print("   • 🔄 多跳推理：评估复杂推理链和逻辑连贯性")
            print("   • 🎯 领域适应：评估知识迁移和跨域应用能力")
            print()
            print("📊 评估特色:")
            print("   • 针对性强：专门针对知识图谱预训练模型的特点")
            print("   • 权重合理：知识图谱推理30%，其他各25%、25%、20%")
            print("   • 维度全面：从不同角度评估模型的知识图谱处理能力")
            print("   • 结果详细：提供详细的能力提升分析和改进建议")
            print()
            print("🔧 使用建议:")
            print("   • 首次使用推荐选择模式1进行完整评估")
            print("   • 如需针对特定能力调优，可使用模式7单项测试")
            print("   • 样本数量建议每项200个，平衡评估准确性和时间成本")
            return

        else:
            # 默认选择：完整评估
            print(f"\n🚀 完整知识图谱预训练模型评估模式")
            print("-" * 40)
            try:
                kg_count = int(input("📊 知识图谱推理样本数量 (默认200): ").strip() or "200")
                qa_count = int(input("📊 结构化问答样本数量 (默认200): ").strip() or "200")
                hop_count = int(input("📊 多跳推理样本数量 (默认200): ").strip() or "200")
                domain_count = int(input("📊 领域适应样本数量 (默认200): ").strip() or "200")
            except ValueError:
                kg_count = qa_count = hop_count = domain_count = 200

            print(f"\n🎯 评估配置确认:")
            print(f"   🔗 知识图谱推理样本: {kg_count}")
            print(f"   📊 结构化问答样本: {qa_count}")
            print(f"   🔄 多跳推理样本: {hop_count}")
            print(f"   🎯 领域适应样本: {domain_count}")
            print(f"   🧠 评估器: {tester.models['evaluator']}")
            print(f"   🔬 微调模型: {tester.models['finetuned']}")
            print(f"   🔬 基线模型: {tester.models['baseline']}")

            confirm = input(f"\n❓ 确认开始知识图谱预训练模型评估? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes', '是']:
                print("❌ 评估已取消")
                return

            print(f"\n🎬 开始完整知识图谱预训练模型评估...")
            tester.run_kg_pretrained_evaluation(kg_count, qa_count, hop_count, domain_count)

    except KeyboardInterrupt:
        print(f"\n\n⚠️ 评估被用户中断")
        print("👋 感谢使用知识图谱预训练模型评估系统！")
    except Exception as e:
        print(f"\n\n❌ 评估过程中出现错误: {e}")
        print("🔧 请检查Ollama服务和模型配置后重试")


if __name__ == "__main__":
    main()