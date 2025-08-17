// controllers/userSkillController.js
const db = require('../db');

const skill2id = require('../data/data.json');
let id2skill = { "54": "微积分", "86": "向量空间", "1": "拓扑方法", "13": "矩阵乘法", "346": "坐标轴", "277": "组合数", "279": "素数", "48": "几何学", "280": "正实数", "312": "初始相位", "325": "切线", "371": "向量投影", "18": "对称性", "70": "投影", "79": "幂运算", "276": "二项式定理", "338": "平面图", "83": "上界", "310": "椭圆曲线", "311": "圆锥", "22": "分形", "50": "未知数", "278": "阶乘", "75": "因式", "305": "二次收敛", "61": "数学物理", "340": "位似变换", "322": "分部积分法", "27": "基底", "40": "无限循环小数", "49": "锥曲面与球面", "363": "QR分解", "34": "系数", "42": "幂次", "67": "函数复合", "4": "平方根", "81": "同构", "77": "二次方程", "82": "定义域", "358": "标准差", "47": "因式分解", "46": "零点", "296": "齐次函数", "324": "参数方程", "334": "四面体", "336": "三角形", "350": "开区间", "309": "极值点", "378": "向量内积", "103": "孤点", "65": "群论", "15": "标量λ", "326": "梯度", "366": "内积", "356": "几何-调和平均数", "95": "弱分离公理", "25": "微分方程", "51": "函数", "84": "正交群", "26": "阶数", "39": "进位制", "294": "矩阵逆", "297": "线性函数", "301": "反函数", "368": "共轭转置", "21": "固定点", "295": "指数函数", "8": "多项式", "36": "数的分割", "80": "子广群", "287": "求和符号", "314": "角频率", "331": "拉格朗日", "302": "分数指数", "307": "最佳多项式", "333": "柯西", "392": "凹凸性", "17": "变换", "12": "矩阵", "11": "余弦函数", "16": "向量", "74": "一次函数", "298": "多元实函数", "106": "邻域", "240": "无穷级数", "299": "复数空间", "217": "三维正交坐标系", "308": "一阶导数", "166": "闭开集", "317": "二项式展开", "288": "二项式系数的求和公式", "523": "有向曲面", "303": "极限集合", "230": "数学集合理论", "354": "连续函数的闭支撑", "35": "整数加法", "306": "收敛速度", "58": "古典力学", "202": "双曲线", "323": "连续可导函数", "78": "乘法", "304": "不动点理论", "110": "拓扑空间", "362": "直角坐标系", "316": "二阶导数", "85": "判别式", "69": "逆矩阵", "5": "代数", "360": "调和平均数", "365": "正交矩阵", "393": "容积", "2": "单位根", "9": "指数运算", "10": "常数项", "319": "可导性", "339": "科学计数法", "361": "单位圆", "391": "平面几何", "231": "数域包含关系", "6": "模运算", "24": "线性", "284": "三角函数", "281": "向量加法", "63": "子群", "282": "三线坐标", "321": "乘积法则", "32": "线性外代数", "283": "共线", "224": "正实数集合", "62": "狭义相对论", "222": "复数域", "204": "圆锥坐标系", "238": "级数的项", "209": "圆球面", "172": "鞍点", "6868": "几何体", "10466": "实数", "925": "无穷小量", "504": "极限", "10467": "等式", "3996": "等比级数", "1543": "无穷", "490": "有理数", "10468": "实数系", "425": "收敛定理", "5368": "闭区间", "5298": "基数", "10469": "拓扑学", "10470": "位置数字系统", "2585": "标准实数系统", "10471": "小数展开式", "464": "数论", "9950": "10进数", "10472": "小数点", "4019": "算术", "2548": "柯西序列", "10473": "戴德金分割", "10474": "不定式", "1627": "无限小数", "5348": "排列组合", "3951": "空函数", "10475": "集合论中的指数运算", "10476": "幂次法则", "10477": "零次幂", "10478": "常数函数", "4983": "零函数", "10479": "零次函数", "3954": "空集", "3492": "水平线", "10480": "一元运算", "4730": "超越数", "10481": "多项式函数", "559": "泰勒级数", "10482": "微分规则", "10483": "逐项微分", "10484": "级数", "6916": "几何级数", "7252": "切比雪夫多项式", "5088": "首项系数", "560": "泰勒展开", "10485": "误差余项", "10486": "e^x", "5299": "复数", "575": "无理数", "5207": "代数数", "1777": "映射(函数)", "10487": "转置", "4974": "集合", "8647": "合成关系", "6867": "几何代数", "6737": "关系合成", "1307": "整数集(Z)", "5316": "子集", "8145": "双曲几何", "10488": "直线", "5772": "互自切点(tacnode)", "10489": "等边五边形", "10490": "射线", "6946": "凸五边形镶嵌", "10491": "线段", "557": "法线", "10492": "矢", "1914": "最值点", "4497": "角平分线", "5287": "圆", "1011": "摆线", "9407": "垂直平分线", "8745": "同界角", "5841": "五边形镶嵌", "10493": "平行", "10494": "四维柱体柱", "1450": "旋转", "8450": "可展曲面", "4173": "球面几何", "5160": "三维空间", "5655": "二维空间", "10495": "直纹曲面", "10496": "一维空间", "6120": "体积", "5003": "长度", "8925": "周长", "10497": "对称轴", "3351": "正方形", "581": "曲线", "3962": "立方体", "9402": "垂直", "6458": "克莱因瓶", "10498": "相交", "10499": "相切", "10500": "相离", "10501": "镜像", "8296": "反演", "10502": "表面积", "10503": "挠率", "4500": "角度", "10504": "面积", "10505": "离心率", "10506": "三角函数表", "3321": "正弦曲线", "10507": "cis", "10508": "三分之一角公式", "10509": "三角函数恒等式", "10510": "三角多项式", "8135": "双曲三角函数", "5266": "反三角函数", "10511": "诱导公式", "7714": "半正矢公式", "10512": "三角函数精确值", "5098": "高斯函数", "10513": "三角函数积分表", "5270": "双曲函数", "10514": "三角函数线", "491": "有理函数", "677": "无理函数", "3319": "正弦平方", "7393": "割圆八线", "5214": "余弦定理", "9260": "圆心角", "8909": "周期性", "6137": "余函数恒等式", "7715": "半正矢定理", "10515": "精确值", "6138": "余切", "515": "根号", "10516": "tan", "10517": "近似作图", "554": "比例", "3311": "正弦", "10518": "π", "10519": "绝对误差", "6462": "克莱姆法则", "7751": "协方差矩阵", "5296": "基", "546": "正交", "5352": "特征值", "2624": "标量", "8819": "向量子空间", "5353": "特征向量", "4227": "线性方程组", "10520": "对偶空间", "7297": "列空间", "4222": "线性投影", "10521": "矩阵中的项", "8143": "双曲余弦函数", "10522": "矩阵中的Q元素", "5221": "偏导数", "8161": "双曲正弦函数", "10523": "Lax对", "544": "正弦函数", "9637": "复共轭矩阵", "10524": "高维Lax对", "2311": "本征值", "499": "李代数", "2312": "本征向量", "4606": "费马小定理", "10525": "幺正矩阵", "2417": "极分解", "10526": "方块矩阵", "10527": "奇异值分解", "5683": "二重向量", "10528": "克利福德代数", "10529": "幂零矩阵", "10530": "LU分解", "3936": "稀疏矩阵", "4419": "行列式", "2733": "格拉姆-施密特正交化", "7763": "单位上三角矩阵", "10531": "非奇异方阵", "5154": "三角矩阵", "6040": "伴随矩阵", "8263": "反对称矩阵", "5277": "可逆矩阵", "9441": "埃尔米特矩阵", "3910": "秩", "10532": "外积", "2681": "核", "4938": "迹", "4105": "线性空间", "9043": "四元数", "10533": "多重积分", "10534": "左反函数", "5691": "二阶可导的凸函数", "5143": "一元函数", "5245": "凸函数", "5194": "二次函数", "10535": "一元可微函数", "5551": "二次可微函数", "5329": "导数", "5246": "凸集", "4351": "绝对值函数", "10536": "严格凸函数", "7711": "半正定", "507": "极小值", "5056": "黑塞矩阵", "7077": "函数限制", "10537": "一对一函数", "5275": "可测函数", "6404": "光滑函数", "9670": "复合函数", "3958": "空间变换", "10538": "非满射函数", "1386": "方程式根", "6260": "偏函数", "10539": "恒等函数", "4920": "连续函数" }

const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec); // 将 exec 转为 Promise
// console.log(skill2id['域论']);

const getAllUserSkills = async (req, res) => {
  try {
    const [rows] = await db.query('SELECT * FROM user_skill_mastery_llm');
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const getUserSkillById = async (req, res) => {
  try {
    const [rows] = await db.query(
      'SELECT * FROM user_skill_mastery_llm WHERE id = ?',
      [req.params.id]
    );
    if (rows.length === 0) return res.status(404).json({ message: '未找到记录' });
    res.json(rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const createUserSkill = async (req, res) => {
  try {
    const {
      user_id, skill_id, skill, template, inference,
      overall_mastery, master, correct
    } = req.body;

    const [result] = await db.query(
      `INSERT INTO user_skill_mastery_llm 
      (user_id, skill_id, skill, template, inference, overall_mastery, master, correct)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [user_id, skill_id, skill, template, inference, overall_mastery, master, correct]
    );

    res.status(201).json({ id: result.insertId });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const updateUserSkill = async (req, res) => {
  try {
    const {
      user_id, skill_id, skill, template, inference,
      overall_mastery, master, correct
    } = req.body;

    const [result] = await db.query(
      `UPDATE user_skill_mastery_llm SET
      user_id = ?, skill_id = ?, skill = ?, template = ?, inference = ?,
      overall_mastery = ?, master = ?, correct = ?
      WHERE id = ?`,
      [user_id, skill_id, skill, template, inference, overall_mastery, master, correct, req.params.id]
    );

    if (result.affectedRows === 0) return res.status(404).json({ message: '未找到记录' });
    res.json({ message: '更新成功' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const deleteUserSkill = async (req, res) => {
  try {
    const [result] = await db.query(
      'DELETE FROM user_skill_mastery_llm WHERE id = ?',
      [req.params.id]
    );

    if (result.affectedRows === 0) return res.status(404).json({ message: '未找到记录' });
    res.json({ message: '删除成功' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};



const createLearningPath = async (req, res) => {
  const connection = await db.getConnection();

  try {
    await connection.beginTransaction();

    const {
      user_id, skill_id, path
    } = req.body;

    // 1. 插入 learning_paths
    const [pathResult] = await connection.query(
      `INSERT INTO learning_paths (user_id, skill_id, path) VALUES (?, ?, ?)`,
      [user_id, skill_id, path]
    );

    const path_id = pathResult.insertId;

    await connection.commit();

    res.status(201).json({ path_id });
  } catch (err) {
    await connection.rollback();
    res.status(500).json({ error: err.message });
  } finally {
    connection.release();
  }
};

const insertQuestionsWithLearningPath = async (req, res) => {
  const connection = await db.getConnection();

  try {
    await connection.beginTransaction();

    const {
      skill_id, path_id, questions
    } = req.body;
    // console.log(req.body);


    // 2. 检查是否有题目需要插入
    if (questions && JSON.stringify(questions).length > 0) {

      const questionValues = [];


      // item 是 { key1: { ... } }, 遍历每个 key
      for (const key in questions) {
        const value = questions[key];
        const value1 = value.exam

        // 将每个 key 对应的值对象插入为一条记录
        questionValues.push([skill_id, path_id, JSON.stringify({
          [key]: value1
        })]);

      }



      // 批量插入 user_path_questions 表
      const [result] = await connection.query(
        `INSERT INTO user_path_questions (skill_id, path_id, questions) VALUES ?`,
        [questionValues]
      );
      const id = result.insertId
      await connection.commit();
      res.status(201).json({ id });
    }
  } catch (err) {
    await connection.rollback();
    res.status(500).json({ error: err.message });
  } finally {
    connection.release();
  }
};


let temp_questions = null;
const checkKnowledgePoint = async (req, res) => {
  const { questions, skill, user_id } = req.body;


  // 2. 检查是否有题目需要插入
  if (questions && JSON.stringify(questions).length > 0) {
    // item 是 { key1: { ... } }, 遍历每个 key
    temp_questions = { questions: questions, skill: skill, user_id: user_id };
    res.status(201).json({ message: 'Success' })
  }
};

const get_temp_questions = async (req, res) => {
  res.status(201).json(temp_questions);
};

const updateQuestionsWithId = async (req, res) => {
  const connection = await db.getConnection();

  try {
    await connection.beginTransaction();

    const {
      id, skill_id, questions
    } = req.body;
    // console.log(req.body);


    // 2. 检查是否有题目需要插入
    if (questions && JSON.stringify(questions).length > 0) {

      const questionValues = [];


      // item 是 { key1: { ... } }, 遍历每个 key
      for (const key in questions) {
        const value = questions[key];
        const value1 = value.exam

        // 将每个 key 对应的值对象插入为一条记录
        questionValues.push([JSON.stringify({
          [key]: value1
        })]);

      }



      // 批量插入 user_path_questions 表
      const [result] = await connection.query(
        `update user_path_questions 
         set questions=?,skill_id=?
         where id=?`,
        [questionValues[0], skill_id, id]
      );

      await connection.commit();
      res.status(201).json({ message: 'success' });
    }
  } catch (err) {
    await connection.rollback();
    res.status(500).json({ error: err.message });
  } finally {
    connection.release();
  }
};

const insertResources = async (req, res) => {
  const connection = await db.getConnection();

  try {
    await connection.beginTransaction();

    const {
      id, resources
    } = req.body;
    console.log(resources);

    // 检查是否有资源需要插入
    if (resources && Object.keys(resources).length > 0) {
      const [result] = await connection.query(
        `UPDATE user_path_questions SET resources = ? WHERE id = ?`,
        [JSON.stringify(resources), id]
      );



      // const id = result.insertId;
      await connection.commit();
      res.status(201).json({ id });
    } else {
      await connection.commit();
      res.status(200).json({ message: '没有资源需要插入' });
    }
  } catch (err) {
    await connection.rollback();
    res.status(500).json({ error: err.message });
  } finally {
    connection.release();
  }
};


// 查询指定 path_id 的所有题目
const getQuestionsAndResourcesByPathId = async (req, res) => {

  try {
    const { path_id } = req.params;


    const [rows] = await db.query(
      'SELECT id,skill_id,questions,resources FROM user_path_questions WHERE path_id = ?',
      [path_id]
    );

    if (rows.length === 0) {
      return res.status(404).json({ message: '未找到该学习路径的题目' });
    }


    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }



};

const getUserByPathId = async (req, res) => {
  try {
    const { path_id } = req.params;

    const [rows] = await db.query(
      'SELECT user_id FROM learning_paths WHERE path_id = ?',
      [path_id]
    );

    if (rows.length === 0) {
      return res.status(404).json({ message: '未找到该路径对应的用户' });
    }

    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const funGetQuestionsBySkillId = async (skillId) => {
  const skill_id = skillId
  // console.log();

  console.log(skill_id);

  // 查询所有题目
  const [rows] = await db.query(
    'SELECT * FROM skill_questions WHERE skill_id = ?',
    [skill_id]
  );
  console.log(2);

  if (rows.length === 0) {
    return { message: '未找到该技能的题目' };
  }
  console.log(3);

  // 按等级分类
  const level1 = JSON.parse(rows.filter(q => q.question_level === 0)[0]["question"]);
  const level2 = JSON.parse(rows.filter(q => q.question_level === 1)[0]["question"]);
  const level3 = JSON.parse(rows.filter(q => q.question_level === 2)[0]["question"]);


  // 模板配置
  const templates = [
    {
      total: 10,
      simple: { count: 5, score: 40 },
      medium: { count: 3, score: 30 },
      hard: { count: 2, score: 30 }
    },
    {
      total: 15,
      simple: { count: 7, score: 35 },
      medium: { count: 5, score: 35 },
      hard: { count: 3, score: 30 }
    },
    {
      total: 20,
      simple: { count: 12, score: 36 },
      medium: { count: 6, score: 36 },
      hard: { count: 2, score: 28 }
    }
  ];

  // 随机选择一个模板
  const template = templates[Math.floor(Math.random() * templates.length)];

  // 随机抽题函数
  const getRandomQuestions = (questions, count) => {
    // 拷贝原数组
    const arr = [...questions];
    // Fisher-Yates 洗牌算法
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    // 取前 count 个
    return arr.slice(0, count);
  };



  // 抽取题目
  const selectedLevel1 = getRandomQuestions(level1, template.simple.count);
  const selectedLevel2 = getRandomQuestions(level2, template.medium.count);
  const selectedLevel3 = getRandomQuestions(level3, template.hard.count);

  const exam = [
    ...selectedLevel1.map((q, i) => ({
      question_number: i + 1,
      question_type: `${q.question_type}`,
      question_text: `<题目>[${i + 1}] ${q.question_text}`,
      options: q.question_type === "选择题" ? `${JSON.stringify(q.options)}` : [],
      answer: `${q.answer}`,
      explanation: `${q.explanation}`
    })),
    ...selectedLevel2.map((q, i) => ({
      question_number: selectedLevel1.length + i + 1,
      question_type: `${q.question_type}`,
      question_text: `<题目>[${selectedLevel1.length + i + 1}] ${q.question_text}`,
      options: q.question_type === "选择题" ? `${JSON.stringify(q.options)}` : [],
      answer: `${q.answer}`,
      explanation: `${q.explanation}`
    })),
    ...selectedLevel3.map((q, i) => ({
      question_number: selectedLevel1.length + selectedLevel2.length + i + 1,
      question_type: `${q.question_type}`,
      question_text: `<题目>[${selectedLevel1.length + selectedLevel2.length + i + 1}] ${q.question_text}`,
      options: q.question_type === "选择题" ? `${JSON.stringify(q.options)}` : [],
      answer: `${q.answer}`,
      explanation: `${q.explanation}`
    }))
  ];
  console.log(4);

  // 返回结果
  return {
    exam,
    total_questions: exam.length,
  };
}

const getQuestionsBySkillId = async (req, res) => {
  try {
    const { skill_id } = req.params;
    console.log(skill_id);

    result = await funGetQuestionsBySkillId(skill_id)
    // console.log(result);
    res.json(result)
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


const { format } = require('date-fns'); // 用于格式化日期
const { parseISO } = require('date-fns');
const { stdout } = require('process');
const { log } = require('console');

// 难度配置
const difficultyConfig = [
  {
    total: 10,
    simple: { count: 5 },
    medium: { count: 3 },
    hard: { count: 2 }
  },
  {
    total: 15,
    simple: { count: 7 },
    medium: { count: 5 },
    hard: { count: 3 }
  },
  {
    total: 20,
    simple: { count: 12 },
    medium: { count: 6 },
    hard: { count: 2 }
  }
];

// 根据题目总数和题目索引获取 question_level
function getQuestionLevel(totalQuestions, questionIndex) {
  // 匹配配置
  const config = difficultyConfig.find(c => c.total === totalQuestions);
  if (!config) {
    console.warn(`未找到 total=${totalQuestions} 的难度配置`);
    return 2; // 默认为困难
  }

  const { simple, medium, hard } = config;

  if (questionIndex < simple.count) {
    return 0; // 简单
  } else if (questionIndex < simple.count + medium.count) {
    return 1; // 中等
  } else {
    return 2; // 困难
  }
}
// 假设你已经引入了数据库连接 db
const saveUserSkillLogs = async (req, res) => {
  // 获取数据库连接（确保你的db支持getConnection）
  connection = await db.getConnection();
  await connection.beginTransaction();
  try {
    const data = req.body; // 前端传来的完整 JSON 数据

    const { user_id, submitted_at, questions_detail, skill_id } = data;

    if (!questions_detail || !Array.isArray(questions_detail)) {
      return res.status(400).json({ message: 'questions_detail 不存在或格式错误' });
    }

    const submitDate = parseISO(submitted_at);
    const dateOnly = format(submitDate, 'yyyy-MM-dd');

    const logsToInsert = questions_detail.map((q, index) => {
      const question_level = getQuestionLevel(data.total_questions, index);
      const correct = q.is_correct ? 1 : 0;
      const cost_time = q.time_spent || 0;
      // console.log(skill2id.hasOwnProperty(data.knowledge_point));


      return [
        user_id,
        data.path_id,
        data.knowledge_point || 'unknown', // skill 字段
        skill_id,
        correct,
        1, // attempt_count（假设每次答题尝试一次）
        q.hint_count || 0,
        0, // bottom_hint（是否使用底部提示，前端没传，默认 0）
        null, // ms_first_response（前端没传）
        null, // first_action（前端没传）
        null, // original（前端没传）
        null,
        null,
        null, // avg_conf_frustrated
        null, // avg_conf_confused
        null, // avg_conf_concentrating
        null, // avg_conf_bored
        cost_time,
        dateOnly,
        question_level// question_level（可选）
      ];
    });

    // 插入数据库
    const query = `
      INSERT INTO user_skill_logs (
        user_id,path_id, skill, skill_id, correct, attempt_count, hint_count, 
        bottom_hint, ms_first_response, first_action, original, 
        start_time, end_time, avg_conf_frustrated, avg_conf_confused, 
        avg_conf_concentrating, avg_conf_bored, cost_time, date, question_level
      ) VALUES ?
    `;

    const [results] = await db.query(query, [logsToInsert]);

    const insertedIds = [];
    let stdout_py = '';
    if (results && results.affectedRows > 0) {
      // 获取第一个插入的ID
      const firstId = results.insertId;
      // 生成所有插入的ID（批量插入的ID是连续的）
      const rows = results.affectedRows;

      // 构造参数字符串（只传 id 为例）
      const args = `--first_id ${firstId} --rows=${rows} --path_id=${data.path_id}`;

      // 执行 Python 脚本并传参
      const { stdout, stderr } = await execPromise(`/home/lms/.conda/envs/test/bin/python /home/lms/project/dify-backend/python_script/llm.py ${args}`);
      if (stderr) {
        console.error(`Python stderr: ${stderr}`);
      }
      stdout_py = stdout;
      // stdout_py = execPromise(`/home/lms/.conda/envs/test/bin/python /home/lms/project/dify-backend/python_script/llm.py ${args}`, (error, stdout, stderr) => {
      //   if (error) {
      //     console.error(`执行出错: ${error.message}`);
      //     return;
      //   }
      //   if (stderr) {
      //     console.error(`错误输出: ${stderr}`);
      //   }
      console.log(`Python 输出: ${stdout}`);

    }

    const sql = `
     select count from user_skill_mastery_statistic where path_id=? and skill_id=?   
    `
    const count = await db.query(sql, [data.path_id, logsToInsert[0][3]]);
    console.log(count);

    await connection.commit();
    res.status(201).json({
      message: '数据保存成功',
      inserted_ids: insertedIds,  // 返回所有新插入记录的ID
      result: stdout_py,
      count: count[0][0].count
    });


  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  } finally {
    // 释放数据库连接
    if (connection) connection.release();
  }
};



const checkAnswer = async (req, res) => {
  // 获取数据库连接（确保你的db支持getConnection）

  try {
    const data = req.body; // 前端传来的完整 JSON 数据

    const { questions_detail, skill, user_id } = data;

    if (!questions_detail || !Array.isArray(questions_detail)) {
      return res.status(400).json({ message: 'questions_detail 不存在或格式错误' });
    }
    console.log('skill:', skill);
    console.log('user_id:', user_id);

    console.log('questions_detail:', questions_detail);

    // 构造调用Python脚本的命令，正确转义参数
    const pythonScriptPath = '/home/lms/project/dify-backend/python_script/llm2.py';
    const pythonPath = '/home/lms/.conda/envs/test/bin/python';

    // 使用spawn方法正确传递参数
    const { spawn } = require('child_process');

    const pythonProcess = spawn(pythonPath, [
      pythonScriptPath,
      '--user_id', user_id,
      '--questions_detail', JSON.stringify(questions_detail),
      '--skill', skill,
    ]);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', async (code) => {
      if (code !== 0) {
        console.error('Python脚本执行错误，退出码:', code);
        console.error('stderr:', stderr);
        return res.status(500).json({
          error: 'Python脚本执行错误',
          stderr: stderr,
          stdout: stdout,
          exitCode: code
        });
      }

      if (stderr) {
        console.error('Python脚本stderr:', stderr);
      }

      console.log('Python脚本stdout:', stdout);
      console.log(`Python 输出: ${stdout}`);
      res.status(201).json({
        message: '数据保存成功',
        result: stdout,
      });
    })

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
};



// 根据skill_id获取随机资源
// 在文件末尾添加以下函数

const funGetResourceBySkillIdAndType = async (skillId, type) => {
  const skill_id = skillId;
  const learning_p = type;
  // 验证 skill_id 参数
  if (!skill_id) {
    return res.status(400).json({ message: 'skill_id 参数是必需的' });
  }

  // 验证 learning_p 是否为允许的值之一，如果不是则默认为视频资源
  const allowedTypes = ['视频资源', '课件资源', '论文资源', '练习资源'];
  let resourceType = learning_p;
  if (!resourceType || !allowedTypes.includes(resourceType)) {
    resourceType = '视频资源'; // 默认为视频资源
  }

  // 查询指定 skill_id 的所有资源
  const [rows] = await db.query(
    'SELECT * FROM skill_with_resources WHERE skill_id = ?',
    [skill_id]
  );

  if (rows.length === 0) {
    return res.status(404).json({ message: '未找到该技能的资源' });
  }
  // console.log(rows);

  // 根据 resourceType 类型过滤资源
  let filteredResources = [];

  for (let i = 0; i < rows.length; i++) {
    if (rows[i].type === resourceType) {
      filteredResources.push(rows[i]);
    }
  }

  // 如果没有特定类型的资源，使用所有资源
  if (filteredResources.length === 0) {
    filteredResources = rows;
  }

  // 随机选择一条记录
  const randomIndex = Math.floor(Math.random() * filteredResources.length);
  const randomResource = filteredResources[randomIndex];

  return {
    resource_type: resourceType,
    resource: randomResource
  };
};
const getResourceBySkillIdAndType = async (req, res) => {
  try {
    const { skill_id, learning_p } = req.body;
    const result = await funGetResourceBySkillIdAndType(skill_id, learning_p);
    console.log(req.body);

    // 返回资源类型和对应的资源
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


const changeResourceAndQuestion = async (req, res) => {
  connection = await db.getConnection();
  await connection.beginTransaction();
  try {
    const { id, knowledge_point, resource_type } = req.body;

    console.log(id);

    // 验证 skill_id 参数
    if (!id) {
      return res.status(400).json({ message: 'id 参数是必需的' });
    }



    // 查询指定 skill_id 的所有资源
    const [rows] = await db.query(
      'SELECT skill_id,resources FROM user_path_questions WHERE id = ?',
      [id]
    );
    console.log(rows);
    if (rows.length === 0) {
      return res.status(404).json({ message: '未找到该资源' });
    }
    const skill_id = rows[0].skill_id;
    const resources = JSON.parse(rows[0].resources);
    const keys = Object.keys(resources);
    // const resource_type = resources[keys[0]].resource_type;
    const skill = knowledge_point
    console.log(skill);

    console.log(resource_type);

    const new_questions1 = await funGetQuestionsBySkillId(skill_id)

    const new_questions = { [skill]: new_questions1 }
    const new_resources = await funGetResourceBySkillIdAndType(skill_id, resource_type)

    // 2. 检查是否有题目需要插入
    if (new_questions && JSON.stringify(new_questions).length > 0) {

      const questionValues = [];


      // item 是 { key1: { ... } }, 遍历每个 key
      for (const key in new_questions) {
        console.log(key);

        const value = new_questions[key];
        const value1 = value.exam

        // 将每个 key 对应的值对象插入为一条记录
        questionValues.push(JSON.stringify({
          [key]: value1
        }));

      }

      console.log(111);
      console.log(JSON.stringify({ [skill]: new_resources }));


      // 批量插入 user_path_questions 表
      await connection.query(
        `update user_path_questions 
         set questions=?,resources=?
         where id=?`,
        [questionValues[0], JSON.stringify({ [skill]: new_resources }), id]
      )

      // await connection.query(
      //   `update user_path_questions
      //    set resources=?
      //    where id=?`,
      //   [JSON.stringify({ [skill]: new_resources }), id]
      // )

    }
    await connection.commit();
    // 返回资源类型和对应的资源
    res.json({
      // resource_type: resourceType,
      // resource: randomResource
      message: "success",
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}


const getUserKnowledgeProfile = async (req, res) => {
  try {
    const { user_id } = req.params;

    // 查询用户知识档案
    const [rows] = await db.query(
      'SELECT accquired_knowledges, weak_knowledges, preference FROM user_knowledge_profile WHERE user_id = ?',
      [user_id]
    );

    // 检查是否找到记录
    if (rows.length === 0) {
      return res.status(404).json({ message: '未找到该用户的知识档案',profile_summary: "", });
    }

    const profile = rows[0];

    // 处理 JSON 字段
    let accquired_knowledges = [];
    let weak_knowledges = [];

    try {
      accquired_knowledges = profile.accquired_knowledges
    } catch (e) {
      console.warn('解析已掌握知识点失败:', e);
      accquired_knowledges = [];
    }

    try {
      weak_knowledges = profile.weak_knowledges
    } catch (e) {
      console.warn('解析薄弱知识点失败:', e);
      weak_knowledges = [];
    }

    // 获取偏好设置
    const preference = profile.preference || '未设置';

    // 提取知识点名称并转换为字符串
    const accquiredSkills = accquired_knowledges.map(item => `${item.skill}，掌握度为${item.overall_mastery}`);
    const weakSkills = weak_knowledges.map(item => `${item.skill}，掌握度为${item.overall_mastery}`);

    const accquiredStr = accquiredSkills.length > 0
      ? accquiredSkills.join('、')
      : '暂无';

    const weakStr = weakSkills.length > 0
      ? weakSkills.join('、')
      : '暂无';

    // 拼接完整字符串
    const resultString = `用户已掌握的知识点的情况为${accquiredStr}，掌握较弱的知识点的情况为${weakStr}，以及用户的学习偏好为${preference}`;

    // 返回结果
    res.json({
      user_id: user_id,
      profile_summary: resultString,
      details: {
        accquired_knowledges: accquired_knowledges,
        weak_knowledges: weak_knowledges,
        preference: preference
      }
    });

  } catch (err) {
    console.error('获取用户知识档案失败:', err);
    res.status(500).json({ error: err.message });
  }
};

const updateUserKnowledgeProfile = async (req, res) => {
  try {
    const { user_id } = req.params;
    const updateFields = req.body;

    // 检查是否有更新字段
    if (!updateFields || Object.keys(updateFields).length === 0) {
      return res.status(400).json({ message: '没有提供要更新的字段' });
    }

    // 构建动态更新语句
    const allowedFields = [
      'accquired_knowledges',
      'weak_knowledges',
      'current_knowledge',
      'preference',
      'current_path'
    ];

    // 过滤出允许更新的字段
    const fieldsToUpdate = {};
    for (const [key, value] of Object.entries(updateFields)) {
      if (allowedFields.includes(key)) {
        fieldsToUpdate[key] = value;
      }
    }

    // 检查过滤后是否还有字段需要更新
    if (Object.keys(fieldsToUpdate).length === 0) {
      return res.status(400).json({ message: '没有有效的字段需要更新' });
    }

    // 构建 SQL 更新语句
    const setClause = Object.keys(fieldsToUpdate)
      .map(field => `${field} = ?`)
      .join(', ');

    const values = [...Object.values(fieldsToUpdate), user_id];
    console.log(Object.keys(fieldsToUpdate).join(', '));
    
    const query = `
      INSERT INTO user_knowledge_profile (user_id, ${Object.keys(fieldsToUpdate).join(', ')})
      VALUES (?, ${Object.keys(fieldsToUpdate).map(() => '?').join(', ')})
      ON DUPLICATE KEY UPDATE
      ${setClause}
    `;

    const [result] = await db.query(query, values);

    res.json({
      message: '用户知识档案更新成功',
      affectedRows: result.affectedRows,
      changedRows: result.changedRows
    });
  } catch (err) {
    console.error('更新用户知识档案失败:', err);
    res.status(500).json({ error: err.message });
  }
};

const changeLearningPath = async (req, res) => {
  try {
    // 获取前端传入的参数
    const { id, path_id, skill_id, user_id, knowledge_point } = req.body;

    // 参数验证
    if (!id || !path_id || !user_id || !knowledge_point) {
      return res.status(400).json({
        message: '缺少必要参数',
        required: ['id', 'path_id', 'user_id', 'knowledge_point']
      });
    }

    console.log('接收到的参数:', { id, path_id, user_id, knowledge_point });

    // 构造调用Python脚本的命令
    const pythonScriptPath = '/home/lms/project/dify-backend/python_script/plan_repeat.py';
    const pythonPath = '/home/lms/.conda/envs/test/bin/python';

    // 构造命令行参数
    const args = `--id ${id} --path_id ${path_id} --skill_id ${skill_id} --skill "${knowledge_point}"`;
    const command = `${pythonPath} ${pythonScriptPath} ${args}`;

    console.log('执行命令:', command);

    // 执行Python脚本并获取输出
    const { stdout, stderr } = await execPromise(command);

    if (stderr) {
      console.error('Python脚本stderr:', stderr);
    }

    console.log('Python脚本stdout:', stdout);

    // 解析Python脚本的输出（假设输出是JSON格式）
    let pythonResult;
    try {

      pythonResult = JSON.parse(stdout.trim());
    } catch (parseError) {
      // 如果解析失败，返回原始stdout
      pythonResult = { raw_output: stdout };
    }

    // 返回成功响应
    res.status(200).json({
      message: '是否要更新学习路径',
      python_output: pythonResult,
    });

  } catch (err) {
    console.error('changeLearningPath接口错误:', err);

    // 根据错误类型返回不同的响应
    if (err.code === 'ENOENT') {
      return res.status(500).json({
        message: 'Python脚本文件未找到',
        error: err.message
      });
    }

    res.status(500).json({
      message: '处理学习路径变更时发生错误',
      error: err.message
    });
  }

};

const confirmPathChange = async (req, res) => {
  try {
    // 获取前端传入的参数
    const { path_id, user_id, python_output } = req.body;

    // 参数验证
    if (!path_id || !user_id || !python_output) {
      return res.status(400).json({
        message: '缺少必要参数',
        required: ['path_id', 'user_id', 'python_output']
      });
    }

    // 验证python_output格式
    if (!python_output.final_plan || !python_output.candidate_ids || !python_output.new_plan_list) {
      return res.status(400).json({
        message: 'python_output格式不正确',
        required: ['final_plan', 'candidate_ids', 'new_plan_list']
      });
    }

    console.log('接收到的参数:', { path_id, user_id, python_output });

    // 构造调用Python脚本的命令，正确转义参数
    const pythonScriptPath = '/home/lms/project/dify-backend/python_script/change_path.py';
    const pythonPath = '/home/lms/.conda/envs/test/bin/python';

    // 使用spawn方法正确传递参数
    const { spawn } = require('child_process');

    const pythonProcess = spawn(pythonPath, [
      pythonScriptPath,
      '--path_id', path_id.toString(),
      '--final_plan', JSON.stringify(python_output.final_plan),
      '--candidate_ids', JSON.stringify(python_output.candidate_ids),
      '--new_plan_list', JSON.stringify(python_output.new_plan_list)
    ]);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', async (code) => {
      if (code !== 0) {
        console.error('Python脚本执行错误，退出码:', code);
        console.error('stderr:', stderr);
        return res.status(500).json({
          error: 'Python脚本执行错误',
          stderr: stderr,
          stdout: stdout,
          exitCode: code
        });
      }

      if (stderr) {
        console.error('Python脚本stderr:', stderr);
      }

      console.log('Python脚本stdout:', stdout);

      try {
        const [rows] = await db.query(
          'SELECT id,skill_id,questions,resources FROM user_path_questions WHERE path_id = ?',
          [path_id]
        );

        if (rows.length === 0) {
          return res.status(404).json({ message: '未找到该学习路径的题目' });
        }

        // 返回成功响应
        res.status(200).json(rows);
      } catch (dbErr) {
        console.error('数据库查询错误:', dbErr);
        res.status(500).json({ error: '数据库查询失败', details: dbErr.message });
      }
    });

  } catch (err) {
    console.error('confirmPathChange接口错误:', err);

    res.status(500).json({
      message: '处理学习路径确认时发生错误',
      error: err.message
    });
  }
};

const generateFinalExam = async (req, res) => {
  try {
    const { path_id } = req.params;

    // 1. 查询指定 path_id 的所有题目
    const [rows] = await db.query(
      'SELECT questions FROM user_path_questions WHERE path_id = ?',
      [path_id]
    );

    if (rows.length === 0) {
      return res.status(404).json({ message: '未找到该学习路径的题目' });
    }

    // 2. 解析所有题目数据
    let allQuestions = [];
    rows.forEach(row => {
      const questionsObj = JSON.parse(row.questions);
      // 遍历对象中的每个知识点
      Object.keys(questionsObj).forEach(skill => {
        const skillQuestions = questionsObj[skill];
        if (skillQuestions && Array.isArray(skillQuestions)) {
          allQuestions = allQuestions.concat(skillQuestions);
        }
      });
    });

    if (allQuestions.length === 0) {
      return res.status(404).json({ message: '未找到有效的题目数据' });
    }

    // 3. 按难度分类题目（根据question_number推断难度）
    const difficultyConfig = [
      {
        total: 10,
        simple: { count: 5 },
        medium: { count: 3 },
        hard: { count: 2 }
      },
      {
        total: 15,
        simple: { count: 7 },
        medium: { count: 5 },
        hard: { count: 3 }
      },
      {
        total: 20,
        simple: { count: 12 },
        medium: { count: 6 },
        hard: { count: 2 }
      }
    ];

    // 4. 随机选择一个模板
    const template = difficultyConfig[Math.floor(Math.random() * difficultyConfig.length)];
    const totalQuestions = template.total;

    // 5. 根据题目总数确定难度分布
    function getQuestionLevel(totalQuestions, questionIndex) {
      const config = difficultyConfig.find(c => c.total === totalQuestions);
      if (!config) return 2; // 默认为困难

      const { simple, medium, hard } = config;

      if (questionIndex < simple.count) {
        return 0; // 简单
      } else if (questionIndex < simple.count + medium.count) {
        return 1; // 中等
      } else {
        return 2; // 困难
      }
    }

    // 6. 从所有题目中随机选择题目
    // 洗牌算法
    function shuffleArray(array) {
      const arr = [...array];
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    }

    // 7. 随机打乱所有题目
    const shuffledQuestions = shuffleArray(allQuestions);

    // 8. 选择指定数量的题目
    const selectedQuestions = shuffledQuestions.slice(0, totalQuestions);

    // 9. 构造最终的题目格式
    const exam = selectedQuestions.map((q, index) => {
      // 重新编号题目
      const questionNumber = index + 1;

      // 保持原有题目信息，只更新编号
      return {
        question_number: questionNumber,
        question_type: q.question_type,
        question_text: q.question_text.replace(/\[.*?\]/, `[${questionNumber}]`), // 更新题目编号
        options: q.options || [],
        answer: q.answer,
        explanation: q.explanation || ''
      };
    });
    const skill_id = await db.query(
      'SELECT skill_id FROM learning_paths WHERE path_id = ?',
      path_id
    );
    const id = skill_id[0][0].skill_id;

    // console.log(id2skill[id]);
    temp_questions = {
      questions: { "exam": exam },
      skill: id2skill[id],
      total_questions: exam.length
    }
    // 10. 返回结果
    res.status(201).json({message: '生成最终测试题目成功'});

  } catch (err) {
    console.error('生成最终测试题目失败:', err);
    res.status(500).json({ error: err.message });
  }
};


module.exports = {
  getAllUserSkills,
  getUserSkillById,
  createUserSkill,
  updateUserSkill,
  deleteUserSkill,
  createLearningPath,
  getQuestionsAndResourcesByPathId,
  getQuestionsBySkillId,
  insertQuestionsWithLearningPath,
  updateQuestionsWithId,
  insertResources,
  getUserByPathId,
  saveUserSkillLogs,
  getResourceBySkillIdAndType,
  changeResourceAndQuestion,
  getUserKnowledgeProfile,
  updateUserKnowledgeProfile,
  changeLearningPath,
  confirmPathChange,
  checkKnowledgePoint,
  get_temp_questions,
  checkAnswer,
  generateFinalExam
};