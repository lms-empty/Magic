import requests
import sys
import argparse
import pymysql
import pandas as pd
from datetime import datetime
from json_repair import repair_json
import json
# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser(description="接收 Node.js 传入的参数")
parser.add_argument('--final_plan', type=str, help='Final learning plan as JSON string')
parser.add_argument('--candidate_ids', type=str, help='Candidate IDs as JSON string')
parser.add_argument('--new_plan_list', type=str, help='New plan list as JSON string')
parser.add_argument('--path_id', type=int, help='Path ID')
parser.add_argument('--learning_p', type=str,default="视频资源" ,help='Learning resource preference')


args = parser.parse_args()

path_id=args.path_id
learning_p=args.learning_p
plan_list=[]
candidate_ids=[]


try:
    final_plan = json.loads(args.final_plan) if args.final_plan else []
    candidate_ids = json.loads(args.candidate_ids) if args.candidate_ids else []
    plan_list = json.loads(args.new_plan_list) if args.new_plan_list else []
except json.JSONDecodeError as e:
    print(f"参数解析错误: {e}")
    sys.exit(1)
def calculate_overlap(str1, str2):
    
    str1=str1.upper()
    str2=str2.upper()
    # 将字符串转换为字符集合
    set1 = set(str1)
    set2 = set(str2)
    
    # 计算交集（重合字符）
    overlap_chars = set1 & set2
    overlap_count = len(overlap_chars)
    
    # 计算并集（总字符数，去重）
    total_chars = set1 | set2
    total_count = len(total_chars)
    
    # 计算重合度（避免除以零）
    if total_count == 0:
        similarity = 0.0
    else:
        similarity = overlap_count * 2 / total_count
    
    return similarity, overlap_count, total_count

id2skill={"54":"微积分","86":"向量空间","1":"拓扑方法","13":"矩阵乘法","346":"坐标轴","277":"组合数","279":"素数","48":"几何学","280":"正实数","312":"初始相位","325":"切线","371":"向量投影","18":"对称性","70":"投影","79":"幂运算","276":"二项式定理","338":"平面图","83":"上界","310":"椭圆曲线","311":"圆锥","22":"分形","50":"未知数","278":"阶乘","75":"因式","305":"二次收敛","61":"数学物理","340":"位似变换","322":"分部积分法","27":"基底","40":"无限循环小数","49":"锥曲面与球面","363":"QR分解","34":"系数","42":"幂次","67":"函数复合","4":"平方根","81":"同构","77":"二次方程","82":"定义域","358":"标准差","47":"因式分解","46":"零点","296":"齐次函数","324":"参数方程","334":"四面体","336":"三角形","350":"开区间","309":"极值点","378":"向量内积","103":"孤点","65":"群论","15":"标量λ","326":"梯度","366":"内积","356":"几何-调和平均数","95":"弱分离公理","25":"微分方程","51":"函数","84":"正交群","26":"阶数","39":"进位制","294":"矩阵逆","297":"线性函数","301":"反函数","368":"共轭转置","21":"固定点","295":"指数函数","8":"多项式","36":"数的分割","80":"子广群","287":"求和符号","314":"角频率","331":"拉格朗日","302":"分数指数","307":"最佳多项式","333":"柯西","392":"凹凸性","17":"变换","12":"矩阵","11":"余弦函数","16":"向量","74":"一次函数","298":"多元实函数","106":"邻域","240":"无穷级数","299":"复数空间","217":"三维正交坐标系","308":"一阶导数","166":"闭开集","317":"二项式展开","288":"二项式系数的求和公式","523":"有向曲面","303":"极限集合","230":"数学集合理论","354":"连续函数的闭支撑","35":"整数加法","306":"收敛速度","58":"古典力学","202":"双曲线","323":"连续可导函数","78":"乘法","304":"不动点理论","110":"拓扑空间","362":"直角坐标系","316":"二阶导数","85":"判别式","69":"逆矩阵","5":"代数","360":"调和平均数","365":"正交矩阵","393":"容积","2":"单位根","9":"指数运算","10":"常数项","319":"可导性","339":"科学计数法","361":"单位圆","391":"平面几何","231":"数域包含关系","6":"模运算","24":"线性","284":"三角函数","281":"向量加法","63":"子群","282":"三线坐标","321":"乘积法则","32":"线性外代数","283":"共线","224":"正实数集合","62":"狭义相对论","222":"复数域","204":"圆锥坐标系","238":"级数的项","209":"圆球面","172":"鞍点","6868":"几何体","10466":"实数","925":"无穷小量","504":"极限","10467":"等式","3996":"等比级数","1543":"无穷","490":"有理数","10468":"实数系","425":"收敛定理","5368":"闭区间","5298":"基数","10469":"拓扑学","10470":"位置数字系统","2585":"标准实数系统","10471":"小数展开式","464":"数论","9950":"10进数","10472":"小数点","4019":"算术","2548":"柯西序列","10473":"戴德金分割","10474":"不定式","1627":"无限小数","5348":"排列组合","3951":"空函数","10475":"集合论中的指数运算","10476":"幂次法则","10477":"零次幂","10478":"常数函数","4983":"零函数","10479":"零次函数","3954":"空集","3492":"水平线","10480":"一元运算","4730":"超越数","10481":"多项式函数","559":"泰勒级数","10482":"微分规则","10483":"逐项微分","10484":"级数","6916":"几何级数","7252":"切比雪夫多项式","5088":"首项系数","560":"泰勒展开","10485":"误差余项","10486":"e^x","5299":"复数","575":"无理数","5207":"代数数","1777":"映射(函数)","10487":"转置","4974":"集合","8647":"合成关系","6867":"几何代数","6737":"关系合成","1307":"整数集(Z)","5316":"子集","8145":"双曲几何","10488":"直线","5772":"互自切点(tacnode)","10489":"等边五边形","10490":"射线","6946":"凸五边形镶嵌","10491":"线段","557":"法线","10492":"矢","1914":"最值点","4497":"角平分线","5287":"圆","1011":"摆线","9407":"垂直平分线","8745":"同界角","5841":"五边形镶嵌","10493":"平行","10494":"四维柱体柱","1450":"旋转","8450":"可展曲面","4173":"球面几何","5160":"三维空间","5655":"二维空间","10495":"直纹曲面","10496":"一维空间","6120":"体积","5003":"长度","8925":"周长","10497":"对称轴","3351":"正方形","581":"曲线","3962":"立方体","9402":"垂直","6458":"克莱因瓶","10498":"相交","10499":"相切","10500":"相离","10501":"镜像","8296":"反演","10502":"表面积","10503":"挠率","4500":"角度","10504":"面积","10505":"离心率","10506":"三角函数表","3321":"正弦曲线","10507":"cis","10508":"三分之一角公式","10509":"三角函数恒等式","10510":"三角多项式","8135":"双曲三角函数","5266":"反三角函数","10511":"诱导公式","7714":"半正矢公式","10512":"三角函数精确值","5098":"高斯函数","10513":"三角函数积分表","5270":"双曲函数","10514":"三角函数线","491":"有理函数","677":"无理函数","3319":"正弦平方","7393":"割圆八线","5214":"余弦定理","9260":"圆心角","8909":"周期性","6137":"余函数恒等式","7715":"半正矢定理","10515":"精确值","6138":"余切","515":"根号","10516":"tan","10517":"近似作图","554":"比例","3311":"正弦","10518":"π","10519":"绝对误差","6462":"克莱姆法则","7751":"协方差矩阵","5296":"基","546":"正交","5352":"特征值","2624":"标量","8819":"向量子空间","5353":"特征向量","4227":"线性方程组","10520":"对偶空间","7297":"列空间","4222":"线性投影","10521":"矩阵中的项","8143":"双曲余弦函数","10522":"矩阵中的Q元素","5221":"偏导数","8161":"双曲正弦函数","10523":"Lax对","544":"正弦函数","9637":"复共轭矩阵","10524":"高维Lax对","2311":"本征值","499":"李代数","2312":"本征向量","4606":"费马小定理","10525":"幺正矩阵","2417":"极分解","10526":"方块矩阵","10527":"奇异值分解","5683":"二重向量","10528":"克利福德代数","10529":"幂零矩阵","10530":"LU分解","3936":"稀疏矩阵","4419":"行列式","2733":"格拉姆-施密特正交化","7763":"单位上三角矩阵","10531":"非奇异方阵","5154":"三角矩阵","6040":"伴随矩阵","8263":"反对称矩阵","5277":"可逆矩阵","9441":"埃尔米特矩阵","3910":"秩","10532":"外积","2681":"核","4938":"迹","4105":"线性空间","9043":"四元数","10533":"多重积分","10534":"左反函数","5691":"二阶可导的凸函数","5143":"一元函数","5245":"凸函数","5194":"二次函数","10535":"一元可微函数","5551":"二次可微函数","5329":"导数","5246":"凸集","4351":"绝对值函数","10536":"严格凸函数","7711":"半正定","507":"极小值","5056":"黑塞矩阵","7077":"函数限制","10537":"一对一函数","5275":"可测函数","6404":"光滑函数","9670":"复合函数","3958":"空间变换","10538":"非满射函数","1386":"方程式根","6260":"偏函数","10539":"恒等函数","4920":"连续函数"}
skill2id={"微积分":"54","向量空间":"86","拓扑方法":"1","矩阵乘法":"13","坐标轴":"346","组合数":"277","素数":"279","几何学":"48","正实数":"280","初始相位":"312","切线":"325","向量投影":"371","对称性":"18","投影":"70","幂运算":"79","二项式定理":"276","平面图":"338","上界":"83","椭圆曲线":"310","圆锥":"311","分形":"22","未知数":"50","阶乘":"278","因式":"75","二次收敛":"305","数学物理":"61","位似变换":"340","分部积分法":"322","基底":"27","无限循环小数":"40","锥曲面与球面":"49","QR分解":"363","系数":"34","幂次":"42","函数复合":"67","平方根":"4","同构":"81","二次方程":"77","定义域":"82","标准差":"358","因式分解":"47","零点":"46","齐次函数":"296","参数方程":"324","四面体":"334","三角形":"336","开区间":"350","极值点":"309","向量内积":"378","孤点":"103","群论":"65","标量λ":"15","梯度":"326","内积":"366","几何-调和平均数":"356","弱分离公理":"95","微分方程":"25","函数":"51","正交群":"84","阶数":"26","进位制":"39","矩阵逆":"294","线性函数":"297","反函数":"301","共轭转置":"368","固定点":"21","指数函数":"295","多项式":"8","数的分割":"36","子广群":"80","求和符号":"287","角频率":"314","拉格朗日":"331","分数指数":"302","最佳多项式":"307","柯西":"333","凹凸性":"392","变换":"17","矩阵":"12","余弦函数":"11","向量":"16","一次函数":"74","多元实函数":"298","邻域":"106","无穷级数":"240","复数空间":"299","三维正交坐标系":"217","一阶导数":"308","闭开集":"166","二项式展开":"317","二项式系数的求和公式":"288","有向曲面":"523","极限集合":"303","数学集合理论":"230","连续函数的闭支撑":"354","整数加法":"35","收敛速度":"306","古典力学":"58","双曲线":"202","连续可导函数":"323","乘法":"78","不动点理论":"304","拓扑空间":"110","直角坐标系":"362","二阶导数":"316","判别式":"85","逆矩阵":"69","代数":"5","调和平均数":"360","正交矩阵":"365","容积":"393","单位根":"2","指数运算":"9","常数项":"10","可导性":"319","科学计数法":"339","单位圆":"361","平面几何":"391","数域包含关系":"231","模运算":"6","线性":"24","三角函数":"284","向量加法":"281","子群":"63","三线坐标":"282","乘积法则":"321","线性外代数":"32","共线":"283","正实数集合":"224","狭义相对论":"62","复数域":"222","圆锥坐标系":"204","级数的项":"238","圆球面":"209","鞍点":"172","几何体":"6868","实数":"10466","无穷小量":"925","极限":"504","等式":"10467","等比级数":"3996","无穷":"1543","有理数":"490","实数系":"10468","收敛定理":"425","闭区间":"5368","基数":"5298","拓扑学":"10469","位置数字系统":"10470","标准实数系统":"2585","小数展开式":"10471","数论":"464","10进数":"9950","小数点":"10472","算术":"4019","柯西序列":"2548","戴德金分割":"10473","不定式":"10474","无限小数":"1627","排列组合":"5348","空函数":"3951","集合论中的指数运算":"10475","幂次法则":"10476","零次幂":"10477","常数函数":"10478","零函数":"4983","零次函数":"10479","空集":"3954","水平线":"3492","一元运算":"10480","超越数":"4730","多项式函数":"10481","泰勒级数":"559","微分规则":"10482","逐项微分":"10483","级数":"10484","几何级数":"6916","切比雪夫多项式":"7252","首项系数":"5088","泰勒展开":"560","误差余项":"10485","e^x":"10486","复数":"5299","无理数":"575","代数数":"5207","映射(函数)":"1777","转置":"10487","集合":"4974","合成关系":"8647","几何代数":"6867","关系合成":"6737","整数集(Z)":"1307","子集":"5316","双曲几何":"8145","直线":"10488","互自切点(tacnode)":"5772","等边五边形":"10489","射线":"10490","凸五边形镶嵌":"6946","线段":"10491","法线":"557","矢":"10492","最值点":"1914","角平分线":"4497","圆":"5287","摆线":"1011","垂直平分线":"9407","同界角":"8745","五边形镶嵌":"5841","平行":"10493","四维柱体柱":"10494","旋转":"1450","可展曲面":"8450","球面几何":"4173","三维空间":"5160","二维空间":"5655","直纹曲面":"10495","一维空间":"10496","体积":"6120","长度":"5003","周长":"8925","对称轴":"10497","正方形":"3351","曲线":"581","立方体":"3962","垂直":"9402","克莱因瓶":"6458","相交":"10498","相切":"10499","相离":"10500","镜像":"10501","反演":"8296","表面积":"10502","挠率":"10503","角度":"4500","面积":"10504","离心率":"10505","三角函数表":"10506","正弦曲线":"3321","cis":"10507","三分之一角公式":"10508","三角函数恒等式":"10509","三角多项式":"10510","双曲三角函数":"8135","反三角函数":"5266","诱导公式":"10511","半正矢公式":"7714","三角函数精确值":"10512","高斯函数":"5098","三角函数积分表":"10513","双曲函数":"5270","三角函数线":"10514","有理函数":"491","无理函数":"677","正弦平方":"3319","割圆八线":"7393","余弦定理":"5214","圆心角":"9260","周期性":"8909","余函数恒等式":"6137","半正矢定理":"7715","精确值":"10515","余切":"6138","根号":"515","tan":"10516","近似作图":"10517","比例":"554","正弦":"3311","π":"10518","绝对误差":"10519","克莱姆法则":"6462","协方差矩阵":"7751","基":"5296","正交":"546","特征值":"5352","标量":"2624","向量子空间":"8819","特征向量":"5353","线性方程组":"4227","对偶空间":"10520","列空间":"7297","线性投影":"4222","矩阵中的项":"10521","双曲余弦函数":"8143","矩阵中的Q元素":"10522","偏导数":"5221","双曲正弦函数":"8161","Lax对":"10523","正弦函数":"544","复共轭矩阵":"9637","高维Lax对":"10524","本征值":"2311","李代数":"499","本征向量":"2312","费马小定理":"4606","幺正矩阵":"10525","极分解":"2417","方块矩阵":"10526","奇异值分解":"10527","二重向量":"5683","克利福德代数":"10528","幂零矩阵":"10529","LU分解":"10530","稀疏矩阵":"3936","行列式":"4419","格拉姆-施密特正交化":"2733","单位上三角矩阵":"7763","非奇异方阵":"10531","三角矩阵":"5154","伴随矩阵":"6040","反对称矩阵":"8263","可逆矩阵":"5277","埃尔米特矩阵":"9441","秩":"3910","外积":"10532","核":"2681","迹":"4938","线性空间":"4105","四元数":"9043","多重积分":"10533","左反函数":"10534","二阶可导的凸函数":"5691","一元函数":"5143","凸函数":"5245","二次函数":"5194","一元可微函数":"10535","二次可微函数":"5551","导数":"5329","凸集":"5246","绝对值函数":"4351","严格凸函数":"10536","半正定":"7711","极小值":"507","黑塞矩阵":"5056","函数限制":"7077","一对一函数":"10537","可测函数":"5275","光滑函数":"6404","复合函数":"9670","空间变换":"3958","非满射函数":"10538","方程式根":"1386","偏函数":"6260","恒等函数":"10539","连续函数":"4920"}




def get_skill_id(arg1,knowledges1,skill2id) -> dict:
    # 测试示例
    knowledge1 = arg1

    knowledges=knowledges1
    values_list = list(knowledges.values())
    # print(values_list)
    sim_list=[]
    for val in values_list:
        similarity, overlap_count, total_count = calculate_overlap(knowledge1, val)
        if(similarity>0):
            sim_list.append({
                "{}".format(str(similarity)):val
            })

    sorted_data = sorted(sim_list, key=lambda item: float(list(item.keys())[0]),reverse=True)
    if len(sorted_data):
        res=list(sorted_data[0].values())
        return {
            "result": skill2id[res[0]],
        }
    else:
        return {
            "result": "",
        }

# 连接数据库
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    database='dify'
)
import json


def update_learning_path(path_id, new_path_list):
    """
    根据path_id更新learning_paths表的path字段
    
    Args:
        path_id (int): 路径ID
        new_path_list (list): 新的路径列表
    
    Returns:
        bool: 更新成功返回True，否则返回False
    """
    try:
        # 将列表转换为JSON字符串
        import json
        path_json = json.dumps(new_path_list, ensure_ascii=False)
        
        # 更新learning_paths表
        with conn.cursor() as cursor:
            sql = "UPDATE learning_paths SET path = %s WHERE path_id = %s"
            cursor.execute(sql, (path_json, path_id))
            conn.commit()
            
            # 检查是否有行被更新
            if cursor.rowcount > 0:
                return True
            else:
                return False
                
    except Exception as e:
        print(f"❌ 更新学习路径失败: {e}")
        conn.rollback()
        return False


def get_questions_by_skill_id(skill_id):
    """
    根据skill_id获取问题列表
    
    Args:
        skill_id (int or str): 技能ID
        
    Returns:
        dict: 接口返回的数据
    """
    # 构建请求URL
    url = f"http://localhost:3000/api/get-questions-by_skill_id/{skill_id}"
    
    try:
        # 发送GET请求
        response = requests.get(url)
        
        # 检查响应状态码
        if response.status_code == 200:
            # 解析JSON响应
            data = response.json()
            return data
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"请求发生异常: {e}")
        return None
    
def get_resources_by_skill_id(skill_id, learning_p=None, base_url="http://172.20.192.113:3000", headers=None):
    """
    根据skill_id和学习资源类型获取资源数据
    
    Args:
        skill_id (int): 技能ID
        learning_p (str, optional): 学习资源类型，可选值包括：
            - "视频资源"
            - "课件资源" 
            - "论文资源"
            - "练习资源"
            如果不提供，默认为"视频资源"
        base_url (str): 基础URL地址
        
    Returns:
        dict: 接口返回的数据
    """
    # 验证skill_id参数
    if not skill_id:
        raise ValueError("skill_id 参数是必需的")
    
    # 验证learning_p参数（如果提供）
    allowed_types = ['视频资源', '课件资源', '论文资源', '练习资源']
    if learning_p and learning_p not in allowed_types:
        learning_p='视频资源'
    
    # 构建请求URL
    url = f"{base_url}/api/get-resources-by-skill-id"
    
    # 构建请求数据
    data = {
        "skill_id": skill_id
    }
    
    # 如果提供了学习资源类型，则添加到请求数据中
    if learning_p:
        data["learning_p"] = learning_p
    
    try:
        # 发送POST请求

        response = requests.post(url, json=data, timeout=30)
        
        # 检查响应状态码
        response.raise_for_status()  # 如果状态码表示错误，会抛出异常
        
        # 解析JSON响应
        result = response.json()
        return result
        
    except requests.exceptions.Timeout:
        print("请求超时")
        return None
    except requests.exceptions.ConnectionError:
        print("连接错误，请检查网络或服务器是否可用")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP错误: {e}")
        print(f"响应内容: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print(f"响应内容: {response.text}")
        return None



# 首先更新学习路径
flag=update_learning_path(path_id,final_plan)

# 获取需要保留的记录ID（除了candidate_ids的第一个）
ids_to_keep = candidate_ids[1:] if len(candidate_ids) > 1 else []
print(f"需要保留的记录ID: {ids_to_keep}")


preserved_records = []
if ids_to_keep:
    try:
        # 查询需要保留的记录数据
        format_strings = ','.join(['%s'] * len(ids_to_keep))
        select_sql = f"SELECT id, skill_id, questions,resources FROM user_path_questions WHERE id IN ({format_strings})"
        preserved_records_df = pd.read_sql(select_sql, conn, params=ids_to_keep)
            
        # 保存记录数据
        for idx, row in preserved_records_df.iterrows():
            preserved_records.append({
                'skill_id': row['skill_id'],
                'questions': row['questions'],
                'resources': row['resources']
            })
        print(f"已保存 {len(preserved_records)} 条记录的数据")
    except Exception as e:
        print(f"保存记录数据失败: {e}")


# 删除candidate_ids中的所有记录
if candidate_ids:
    try:
        with conn.cursor() as cursor:
                # 构造SQL语句删除所有candidate_ids记录
            format_strings = ','.join(['%s'] * len(candidate_ids))
            delete_sql = f"DELETE FROM user_path_questions WHERE id IN ({format_strings})"
            cursor.execute(delete_sql, tuple(candidate_ids))
            conn.commit()
            print(f"已删除记录 IDs: {candidate_ids}")
    except Exception as e:
        print(f"删除记录失败: {e}")
        conn.rollback()


# 为plan_list中的每个知识点创建新记录
new_record_ids = []
for i, plan in enumerate(plan_list):
    skill_id1 = get_skill_id(plan, id2skill, skill2id)
    print(skill_id1)
        
    if skill_id1["result"]:
            # 获取问题数据
        questions_result = get_questions_by_skill_id(int(skill_id1["result"]))
        resources_result = get_resources_by_skill_id(int(skill_id1["result"]),learning_p)
        if questions_result:
            try:
                with conn.cursor() as cursor:
                    # 构造问题数据
                    questions_data = {plan: questions_result["exam"] if "exam" in questions_result else questions_result}
                    questions_json = json.dumps(questions_data, ensure_ascii=False)
                    
                    #构造资源数据
                    resources_data={plan:resources_result if "resource_type" in resources_result else {}}
                    resources_json = json.dumps(resources_data, ensure_ascii=False)
                    # 插入新记录
                    insert_sql = """
                    INSERT INTO user_path_questions (skill_id, path_id, questions,resources) 
                    VALUES (%s, %s, %s,%s)
                    """
                    cursor.execute(insert_sql, (
                        int(skill_id1["result"]), 
                        path_id, 
                        questions_json,
                        resources_json
                    ))
                    conn.commit()
                    new_id = cursor.lastrowid
                    new_record_ids.append(new_id)
                    print(f"知识点 '{plan}' 的问题数据插入成功，新ID: {new_id}")
                        
            except Exception as e:
                print(f"插入问题数据失败: {e}")
                conn.rollback()
        else:
            print(f"获取知识点 '{plan}' 的问题数据失败")
    else:
        print(f"未找到知识点 '{plan}' 对应的skill_id")
    
    # 重新插入保留的记录
if preserved_records:
    try:
        with conn.cursor() as cursor:
            for record in preserved_records:
                insert_sql = """
                INSERT INTO user_path_questions (skill_id, path_id, questions,resources) 
                VALUES (%s, %s, %s,%s)
                """
                cursor.execute(insert_sql, (
                    record['skill_id'],
                    path_id,
                    record['questions'],
                    record['resources']
                ))
            conn.commit()
            print(f"已重新插入 {len(preserved_records)} 条保留记录")
    except Exception as e:
        print(f"重新插入保留记录失败: {e}")
        conn.rollback()
    
print(f"新插入的记录IDs: {new_record_ids}")
print("已完成记录重组操作")

