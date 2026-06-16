# model_utils.py
import os
from dotenv import load_dotenv

# Hugging Face reads endpoint/cache settings while its modules are imported.
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
import requests
import numpy as np
from typing import List, Tuple,Dict,Optional
from DocumentSplitter import DocumentSplitter, GeneralDocumentSplitter
from BM25Retriever import BM25Retriever
from DocumentProcessor import DocumentProcessor
import json
import traceback
import collections
from langchain_community.embeddings import HuggingFaceEmbeddings
import re




class QueryEnhancer:
    """
    检索查询增强器 (优化版)：
    1. 法律名称强制全称化 + 书名号包裹
    2. 阿拉伯数字直接替换为中文数字
    """
    
    # 法律别名库 (保持原有丰富度)
    LAW_ALIASES = {
        # 纪检监察
        "中国共产党纪律处分条例": "中国共产党纪律处分条例",
        "纪律处分条例": "中国共产党纪律处分条例",
        "监察法实施条例": "中华人民共和国监察法实施条例",
        "监察法": "中华人民共和国监察法",
        "公职人员政务处分法": "中华人民共和国公职人员政务处分法",
        "政务处分法": "中华人民共和国公职人员政务处分法",

        # 核心大法
        "民法典": "中华人民共和国民法典",
        "刑法": "中华人民共和国刑法",
        "宪法": "中华人民共和国宪法",
        "刑事诉讼法": "中华人民共和国刑事诉讼法",
        "刑诉法": "中华人民共和国刑事诉讼法",
        "民事诉讼法": "中华人民共和国民事诉讼法",
        "民诉法": "中华人民共和国民事诉讼法",
        "行政诉讼法": "中华人民共和国行政诉讼法",
        "行诉法": "中华人民共和国行政诉讼法",
        "标准化法":"中华人民共和国标准化法",
        "残疾人保障法":"中华人民共和国残疾人保障法",
        "草原法":"中华人民共和国草原法",
        "测绘法": "中华人民共和国测绘法",
        "车船税法": "中华人民共和国车船税法",
        "车辆购置税法":"中华人民共和国车辆购置税法",
        "房地产管理法": "中华人民共和国城市房地产管理法",
        "城市维护建设税法": "中华人民共和国城市维护建设税法",
        "城乡规划法":"中华人民共和国城乡规划法",
        "畜牧法": "中华人民共和国畜牧法",
        "反家暴法":"中华人民共和国反家庭暴力法",
        
        
        # 劳动与社保
        "劳动法": "中华人民共和国劳动法",
        "劳动合同法": "中华人民共和国劳动合同法",
        "社保法": "中华人民共和国社会保险法",
        "工伤保险条例": "工伤保险条例",
        "失业条例": "失业保险条例",
        
        # 商事与经济
        "公司法": "中华人民共和国公司法",
        "合伙企业法": "中华人民共和国合伙企业法",
        "破产法": "中华人民共和国企业破产法",
        "证券法": "中华人民共和国证券法",
        "保险法": "中华人民共和国保险法",
        "票据法": "中华人民共和国票据法",
        "税法": "中华人民共和国税收征收管理法",
        "个人所得税法": "中华人民共和国个人所得税法",
        "企业所得税法": "中华人民共和国企业所得税法",
        "增值税法": "中华人民共和国增值税法",
        
        # 行政与治安
        "治安管理处罚法": "中华人民共和国治安管理处罚法",
        "治安处罚法": "中华人民共和国治安管理处罚法",
        "行政处罚法": "中华人民共和国行政处罚法",
        "行政许可法": "中华人民共和国行政许可法",
        "强制法": "中华人民共和国行政强制法",
        "复议法": "中华人民共和国行政复议法",
        
        # 民事专项
        "著作权法": "中华人民共和国著作权法",
        "专利法": "中华人民共和国专利法",
        "商标法": "中华人民共和国商标法",
        "反不正当竞争法": "中华人民共和国反不正当竞争法",
        "消费者权益保护法": "中华人民共和国消费者权益保护法",
        "消保法": "中华人民共和国消费者权益保护法",
        "产品质量法": "中华人民共和国产品质量法",
        "食品安全法": "中华人民共和国食品安全法",
        
        "土地管理法": "中华人民共和国土地管理法",
        "农村土地承包法": "中华人民共和国农村土地承包法",
        
        # 刑事专项
        "禁毒法": "中华人民共和国禁毒法",
        "反恐法": "中华人民共和国反恐怖主义法",
        "国安法": "中华人民共和国国家安全法",
        
        # 其他高频
        "道路交通安全法": "中华人民共和国道路交通安全法",
        "道交法": "中华人民共和国道路交通安全法",
        "交规": "中华人民共和国道路交通安全法",
        "环境保护法": "中华人民共和国环境保护法",
        "未成年人保护法": "中华人民共和国未成年人保护法",
        "妇女权益保障法": "中华人民共和国妇女权益保障法",
        "老年人权益保障法": "中华人民共和国老年人权益保障法",
        "教育法": "中华人民共和国教育法",
        "教师法": "中华人民共和国教师法",
        "医师法": "中华人民共和国医师法",
        "网络安全法": "中华人民共和国网络安全法",
        "数据安全法": "中华人民共和国数据安全法",
        "个人信息保护法": "中华人民共和国个人信息保护法",
        "个保法": "中华人民共和国个人信息保护法",
        "电子商务法": "中华人民共和国电子商务法",
        "旅游法": "中华人民共和国旅游法",
        "招标投标法": "中华人民共和国招标投标法",
        "政府采购法": "中华人民共和国政府采购法",
        "法律援助法": "中华人民共和国法律援助法",
        "人民调解法": "中华人民共和国人民调解法",
        "公证法": "中华人民共和国公证法",
        "律师法": "中华人民共和国律师法",
        "法官法": "中华人民共和国法官法",
        "检察官法": "中华人民共和国检察官法",
        "警察法": "中华人民共和国人民警察法",
        "兵役法": "中华人民共和国兵役法",
        "退役军人保障法": "中华人民共和国退役军人保障法",
        "慈善法": "中华人民共和国慈善法", #
        "红十字会法": "中华人民共和国红十字会法",
        "献血法": "中华人民共和国献血法",
        "传染病防治法": "中华人民共和国传染病防治法",
        "疫苗管理法": "中华人民共和国疫苗管理法",
        "药品管理法": "中华人民共和国药品管理法",
        "中医药法": "中华人民共和国中医药法",
        "体育法": "中华人民共和国体育法",
        "电影产业促进法": "中华人民共和国电影产业促进法",
        "公共图书馆法": "中华人民共和国公共图书馆法",
        "博物馆条例": "博物馆条例",
        "文物保护法": "中华人民共和国文物保护法",
        "非物质文化遗产法": "中华人民共和国非物质文化遗产法",
        "档案法": "中华人民共和国档案法",#
        "保密法": "中华人民共和国保守国家秘密法",
        
        "气象法": "中华人民共和国气象法",
        "防震减灾法": "中华人民共和国防震减灾法",
        "消防法": "中华人民共和国消防法",
        "安全生产法": "中华人民共和国安全生产法",
        "矿山安全法": "中华人民共和国矿山安全法",
        "铁路法": "中华人民共和国铁路法",
        "公路法": "中华人民共和国公路法",
        "港口法": "中华人民共和国港口法",
        "民用航空法": "中华人民共和国民用航空法",
        "邮政法": "中华人民共和国邮政法",
        "电信条例": "中华人民共和国电信条例",
        "无线电管理条例": "中华人民共和国无线电管理条例",
        "计算机信息系统安全保护条例": "计算机信息系统安全保护条例",
        "信息网络传播权保护条例": "信息网络传播权保护条例",
        "计算机软件保护条例": "计算机软件保护条例",
        "集成电路布图设计保护条例": "集成电路布图设计保护条例",
        "植物新品种保护条例": "中华人民共和国植物新品种保护条例",
        "人类遗传资源管理条例": "人类遗传资源管理条例",
        "生物安全法": "中华人民共和国生物安全法",
        "长江保护法": "中华人民共和国长江保护法",
        "黄河保护法": "中华人民共和国黄河保护法",
        "湿地保护法": "中华人民共和国湿地保护法",
        "噪声污染防治法": "中华人民共和国噪声污染防治法",
        "固体废物污染环境防治法": "中华人民共和国固体废物污染环境防治法",
        "水污染防治法": "中华人民共和国水污染防治法",
        "大气污染防治法": "中华人民共和国大气污染防治法",#
        "土壤污染防治法": "中华人民共和国土壤污染防治法",
        "核安全法": "中华人民共和国核安全法",
        "放射性污染防治法": "中华人民共和国放射性污染防治法",
        "清洁生产促进法": "中华人民共和国清洁生产促进法",
        "循环经济促进法": "中华人民共和国循环经济促进法",
        "节约能源法": "中华人民共和国节约能源法",
        "可再生能源法": "中华人民共和国可再生能源法",
        "电力法": "中华人民共和国电力法",
        "煤炭法": "中华人民共和国煤炭法",
        "石油天然气管道保护法": "中华人民共和国石油天然气管道保护法",
        "矿产资源法": "中华人民共和国矿产资源法",
        "水法": "中华人民共和国水法",
        "水土保持法": "中华人民共和国水土保持法",
        "渔业法": "中华人民共和国渔业法",
        "种子法": "中华人民共和国种子法",
        "农业法": "中华人民共和国农业法",
        
        "动物防疫法": "中华人民共和国动物防疫法",
        "进出境动植物检疫法": "中华人民共和国进出境动植物检疫法",
        "粮食流通管理条例": "粮食流通管理条例",
        "中央储备粮管理条例": "中央储备粮管理条例",
        "价格法": "中华人民共和国价格法",
        "反垄断法": "中华人民共和国反垄断法",
        "对外贸易法": "中华人民共和国对外贸易法",
        "海关法": "中华人民共和国海关法",
        "进出口商品检验法": "中华人民共和国进出口商品检验法",
        "国境卫生检疫法": "中华人民共和国国境卫生检疫法",
        "外汇管理条例": "中华人民共和国外汇管理条例",
        "外资银行管理条例": "中华人民共和国外资银行管理条例",
        "外资保险公司管理条例": "中华人民共和国外资保险公司管理条例",
        "境外非政府组织境内活动管理法": "中华人民共和国境外非政府组织境内活动管理法",
        "外商投资法": "中华人民共和国外商投资法",
        "台湾同胞投资保护法": "中华人民共和国台湾同胞投资保护法",
        "归侨侨眷权益保护法": "中华人民共和国归侨侨眷权益保护法",
        "民族区域自治法": "中华人民共和国民族区域自治法",
        "居民委员会组织法": "中华人民共和国城市居民委员会组织法",
        "村民委员会组织法": "中华人民共和国村民委员会组织法",
        "选举法": "中华人民共和国全国人民代表大会和地方各级人民代表大会选举法",
        "代表法": "中华人民共和国全国人民代表大会和地方各级人民代表大会代表法",
        "立法法": "中华人民共和国立法法",
        "监督法": "中华人民共和国各级人民代表大会常务委员会监督法",
        "预算法": "中华人民共和国预算法",
        "审计法": "中华人民共和国审计法",
        "统计法": "中华人民共和国统计法",
        "会计法": "中华人民共和国会计法",
        "注册会计师法": "中华人民共和国注册会计师法",
        "资产评估法": "中华人民共和国资产评估法",
        "税务征管法": "中华人民共和国税收征收管理法",
        
        "烟叶税法": "中华人民共和国烟叶税法",
        "船舶吨税法": "中华人民共和国船舶吨税法", #
        "耕地占用税法": "中华人民共和国耕地占用税法",
        "契税法": "中华人民共和国契税法",
        
        "印花税法": "中华人民共和国印花税法",
        "环境保护税法": "中华人民共和国环境保护税法",
        "资源税法": "中华人民共和国资源税法",
        "土地增值税暂行条例": "中华人民共和国土地增值税暂行条例",
        "房产税暂行条例": "中华人民共和国房产税暂行条例",
        "城镇土地使用税暂行条例": "中华人民共和国城镇土地使用税暂行条例",
        "车辆购置税暂行条例": "中华人民共和国车辆购置税暂行条例",
        "印花税暂行条例": "中华人民共和国印花税暂行条例",
        "契税暂行条例": "中华人民共和国契税暂行条例",
        "资源税暂行条例": "中华人民共和国资源税暂行条例",
        "环保税暂行条例": "中华人民共和国环境保护税暂行条例",
        "烟叶税暂行条例": "中华人民共和国烟叶税暂行条例",
        "船舶吨税暂行条例": "中华人民共和国船舶吨税暂行条例",
        "车船税暂行条例": "中华人民共和国车船税暂行条例",
        "耕地占用税暂行条例": "中华人民共和国耕地占用税暂行条例",
        "城市维护建设税暂行条例": "中华人民共和国城市维护建设税暂行条例",
        "增值税暂行条例": "中华人民共和国增值税暂行条例",
        "消费税暂行条例": "中华人民共和国消费税暂行条例"
    }

    @staticmethod
    def number_to_chinese(num_str: str) -> str:
        """
        将阿拉伯数字转换为中文数字 (支持 0-9999)
        专门优化法条场景
        """
        try:
            num = int(num_str)
        except ValueError:
            return num_str
            
        if num == 0: return "零"
        if num < 0: return "负" + QueryEnhancer.number_to_chinese(str(-num))
        
        cn_nums = "零一二三四五六七八九"
        
        # 处理 0-20 的特殊读法
        if num < 11:
            return cn_nums[num]
        elif num < 20:
            return "十" + (cn_nums[num - 10] if num > 10 else "")
        
        # 处理 20-99
        if num < 100:
            tens = num // 10
            ones = num % 10
            res = cn_nums[tens] + "十"
            if ones > 0:
                res += cn_nums[ones]
            return res
        
        # 处理 100-999
        if num < 1000:
            hundreds = num // 100
            rest = num % 100
            res = cn_nums[hundreds] + "百"
            if rest > 0:
                if rest < 10:
                    res += "零" + cn_nums[rest]
                else:
                    res += QueryEnhancer.number_to_chinese(str(rest))
            return res
            
        # 处理 1000-9999
        if num < 10000:
            thousands = num // 1000
            rest = num % 1000
            res = cn_nums[thousands] + "千"
            if rest > 0:
                if rest < 100:
                    if rest < 10:
                        res += "零零" + cn_nums[rest]
                    else:
                        res += "零" + QueryEnhancer.number_to_chinese(str(rest))
                else:
                    res += QueryEnhancer.number_to_chinese(str(rest))
            return res

        return num_str

    @classmethod
    def enhance(cls, query: str) -> str:
        """
        执行查询增强：
        1. 法律名称 -> 《全称》
        2. 阿拉伯数字 -> 直接替换为中文数字
        """
        enhanced_query = query

        # 纪检监察场景中的常见口误。
        if "监察法" in enhanced_query and "检查对象" in enhanced_query:
            enhanced_query = enhanced_query.replace("检查对象", "监察对象")
        
        # --- 步骤 1: 法律名称标准化 (加书名号 + 全称) ---
        # 按长度排序，优先匹配长名字，避免短词误匹配
        sorted_aliases = sorted(cls.LAW_ALIASES.keys(), key=len, reverse=True)
        
        for alias in sorted_aliases:
            if alias in enhanced_query:
                full_name = cls.LAW_ALIASES[alias]
                
                # 定义匹配模式：查找《别名》或 别名
                # 使用正则边界确保匹配完整词，避免 "民法典" 匹配到 "民法典释义" 中的部分（虽然替换后通常没问题，但更严谨）
                # 这里简化处理：直接查找包含别名的情况，分两种情形
                
                # 情形 A: 已经被书名号包裹，如《民法典》->《中华人民共和国民法典》
                # 模式：《...别名...》
                pattern_book = re.compile(r'《(.*?)' + re.escape(alias) + r'(.*?)》')
                
                # 先处理有书名号的情况
                def replace_in_books(match):
                    # 提取书名号内别名前后的内容（通常为空，或者是修饰语，这里假设别名就是核心）
                    # 为了简单且准确，我们假设用户写的是《民法典》，中间没有其他字
                    # 如果用户写《新民法典》，逻辑会稍微复杂，这里主要处理标准简称
                    # 策略：直接替换整个书名号内容为《全称》
                    return f"《{full_name}》"
                
                # 检查是否存在《...别名...》
                if re.search(r'《[^》]*' + re.escape(alias) + r'[^》]*》', enhanced_query):
                    dynamic_pattern = re.compile(r'《([^》]*?)' + re.escape(alias) + r'([^》]*?)》')

                    def preserve_version_in_book(match):
                        prefix, suffix = match.group(1), match.group(2)
                        content = f"{prefix}{alias}{suffix}"
                        if full_name in content:
                            return f"《{content}》"
                        return f"《{prefix}{full_name}{suffix}》"

                    # 保留“2003年版”“2024修正”等版本限定词。
                    enhanced_query = dynamic_pattern.sub(
                        preserve_version_in_book,
                        enhanced_query,
                    )
                else:
                    # 情形 B: 没有书名号，如 民法典 -> 《中华人民共和国民法典》
                    # 使用单词边界或简单的字符串替换，注意不要替换掉已经是全称的情况
                    if full_name not in enhanced_query:
                        # 简单的全局替换，因为别名通常比较独特
                        enhanced_query = enhanced_query.replace(alias, f"《{full_name}》")

        # --- 步骤 2: 数字直接替换 ---
        # 匹配模式：第 123 条，123 条，第 123 款，123 款，第 123 项，123 项
        # 同时也匹配 standalone 的数字吗？通常法条场景下，数字都伴随量词。
        # 为了安全，只替换伴随法条量词（条、款、项、编、章、节）的数字，避免替换日期或金额
        patterns = [
            (r'第\s*(\d+)\s*条', '条'),
            (r'(\d+)\s*条', '条'),
            (r'第\s*(\d+)\s*款', '款'),
            (r'(\d+)\s*款', '款'),
            (r'第\s*(\d+)\s*项', '项'),
            (r'(\d+)\s*项', '项'),
            (r'第\s*(\d+)\s*编', '编'),
            (r'(\d+)\s*编', '编'),
            (r'第\s*(\d+)\s*章', '章'),
            (r'(\d+)\s*章', '章'),
            (r'第\s*(\d+)\s*节', '节'),
            (r'(\d+)\s*节', '节'),
        ]
        
        for pattern, suffix in patterns:
            def replace_num(match):
                # match.group(0) 是整体 (如 "第 82 条" 或 "82 条")
                # match.group(1) 是数字部分 (如 "82")
                num_str = match.group(1)
                chinese_num = cls.number_to_chinese(num_str)
                
                # 重建字符串
                prefix = match.group(0).split(num_str)[0] # 获取数字前的部分 (如 "第 " 或 "")
                return f"{prefix}{chinese_num}{suffix}"
            
            enhanced_query = re.sub(pattern, replace_num, enhanced_query)
        
        return enhanced_query





class LegalRAGapi:

    def __init__(
        self,
        db_path: str = None,
        source_dir: str = None,
        vector_weight: float = -1.0,
        bm25_weight: float = -1.0,
        embedding_model=None,
    ):
        """
        初始化 RAG 系统核心组件 (向量库、嵌入模型、检索器)
        注意：已移除 Memory 模块和 Prompt 加载模块，由 LangGraph 统一管理

        参数:
          source_dir: 案例 JSON 源文件目录（用于引用溯源的按需回源查询）
          embedding_model: 可复用的嵌入模型实例，避免多个知识库重复加载大模型
        """
        # 1. 确定数据库路径
        if db_path is None:
            db_path = os.getenv("VECTOR_CASE_DB_PATH", "law_faiss_case_pattern_pid") # 默认案例数据库路径
        self.db_path = db_path

        # 引用溯源：源文件目录 + LRU 缓存
        self.source_dir = source_dir
        self._lru_cache = collections.OrderedDict()
        self._record_lru_cache = collections.OrderedDict()
        self._lru_max_size = 256


        # 2. 初始化或复用嵌入模型。案例库和法条库必须共享同一实例，
        # 否则重复加载 BGE-M3 可能导致 PyTorch c10.dll 原生崩溃。
        if embedding_model is not None:
            self.embedding_model = embedding_model
            print("♻️ 复用已加载的嵌入模型")
        else:
            print("🧠 正在加载嵌入模型...")
            embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
            model_cache_dir = os.path.abspath(
                os.getenv("MODEL_CACHE_DIR", os.path.join(os.getcwd(), "model_cache"))
            )
            os.makedirs(model_cache_dir, exist_ok=True)
            import torch
            embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    cache_folder=model_cache_dir,
                    model_kwargs={'device': embedding_device},
                    encode_kwargs={
                        'normalize_embeddings': True,
                        'batch_size': 64 if embedding_device == "cuda" else 8,
                    }
                )
                if embedding_device == "cuda":
                    # FP16加速：显存减半，编码速度提升
                    self.embedding_model.client.half()
                self.embedding_model.client.max_seq_length = 2048
                print(
                    f"✅ 嵌入模型加载完成 "
                    f"({embedding_device.upper()}, max_seq=2048)"
                )
            except Exception as e:
                print(f"❌ 嵌入模型加载失败：{e}")
                # 降级到 CPU 尝试
                try:
                    print("🔄 尝试降级到 CPU 加载...")
                    self.embedding_model = HuggingFaceEmbeddings(
                        model_name=embedding_model_name,
                        cache_folder=model_cache_dir,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    print("✅ 嵌入模型加载完成 (CPU 模式)")
                except Exception as e2:
                    raise e2

        # 初始化问题增强器
        self.enhancer = QueryEnhancer()

        
        # 3. 初始化向量数据库
        self.vector_db = None
        if os.path.exists(self.db_path):
            print(f"📂 加载已存在的向量数据库：{self.db_path}")
            self.load_vector_db()
        else:
            print(f"⚠️ 向量数据库不存在：{self.db_path}，将在添加文档时创建")

        

        # 4. 初始化 BM25 检索器
        bm25_filename = "bm25_index.pkl" 
        bm25_full_path = os.path.join(db_path, bm25_filename)
        
        self.bm25_retriever = BM25Retriever(bm25_full_path, rebuild_threshold=50)
        if not self.bm25_retriever.load_index():
            print("ℹ️ BM25 索引不存在，将在添加文档时构建")
        else:
            print(f"✅ BM25 索引加载成功，文档数量：{self.bm25_retriever.get_document_count()}")

        # 5. 初始化其他组件
        self.document_processor = DocumentProcessor()
        self.general_splitter = GeneralDocumentSplitter(chunk_size=200, chunk_overlap=20)
        
        # Reranker 已通过消融实验验证在本场景下无正向收益，跳过加载以节省显存
        self.reranker = None



        # 检索权重配置
        self.vector_weight = vector_weight if vector_weight >= 0 else 0.3
        self.bm25_weight = bm25_weight if bm25_weight >=0 else 0.7

    def _get_llm_client(self, api_key: str, base_url: str, model_name: str):
        """
        【关键方法】动态获取 LLM 客户端
        支持传入不同的 API Key 和 Base URL，适配多模型切换
        """
        if not api_key:
            class _DummyLLM:
                def invoke(self, messages):
                    return type('obj', (object,), {'content': f"❌ 错误：API Key 未配置。"})()
                def stream(self, messages):
                    def _gen():
                        yield type('obj', (object,), {'content': f"❌ 错误：API Key 未配置。"})()
                    return _gen()
            return _DummyLLM()

        try:
            return ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model_name,
                http_client=None,
                streaming=True,
                temperature=0.7
            )
        except Exception as e:
            print(f"⚠️ 初始化 LLM 客户端失败 ({model_name}): {e}")
            class _ErrorLLM:
                def invoke(self, messages):
                    return type('obj', (object,), {'content': f"❌ LLM 初始化错误：{str(e)}"})()
                def stream(self, messages):
                    def _gen():
                        yield type('obj', (object,), {'content': f"❌ LLM 初始化错误：{str(e)}"})()
                    return _gen()
            return _ErrorLLM()
        

    def _rerank_documents(self, query: str, documents: List[str], pids: List[Optional[str]] = None, top_k: int = 10) -> List[Tuple[str, float, Optional[str]]]:
        """本地 Reranker 对文档进行重排序，返回 (text, score, pid)"""
        if pids is None:
            pids = [None] * len(documents)

        if self.reranker is None:
            return [(doc, 0.0, pid) for doc, pid in zip(documents[:top_k], pids[:top_k])]

        try:
            pairs = [[query, doc] for doc in documents]
            scores = self.reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]

            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            reranked = [(documents[i], s, pids[i]) for i, s in indexed]
            return reranked[:top_k]
        except Exception as e:
            print(f"Reranker 调用失败：{e}")
            return [(doc, 0.0, pid) for doc, pid in zip(documents[:top_k], pids[:top_k])]

    def add_documents(self, documents: List[str], pids: List[Optional[str]] = None, save_to_disk: bool = False):
        """添加文档到向量数据库和 BM25 索引"""
        if not documents:
            return

        if pids is None:
            pids = [None] * len(documents)

        print(f"正在向向量数据库添加 {len(documents)} 个文档块...")
        # 分批编码，避免GPU显存溢出
        batch_size = 256
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            all_embeddings.extend(self.embedding_model.embed_documents(batch))
            if (i // batch_size + 1) % 50 == 0:
                print(f"   编码进度: {min(i+batch_size, len(documents))}/{len(documents)}")
        embeddings = all_embeddings
        embeddings_array = np.array(embeddings, dtype=np.float32)

        if len(embeddings_array.shape) != 2:
            raise ValueError(f"嵌入维度不正确")

        metadatas = [{"pid": pid} for pid in pids]

        if self.vector_db is None:
            docs = [Document(page_content=text) for text in documents]
            self.vector_db = FAISS.from_embeddings(
                text_embeddings=list(zip(documents, embeddings_array)),
                embedding=self.embedding_model,
                metadatas=metadatas
            )
        else:
            self.vector_db.add_texts(documents, embeddings=embeddings_array, metadatas=metadatas)

        self.bm25_retriever.add_documents(documents, pids=pids)

        if save_to_disk:
            self.save_vector_db()
            self.bm25_retriever.save_index()

        print(f"文档添加完成 - 向量库：{self.get_document_count()}, BM25: {self.bm25_retriever.get_document_count()}")
        
    def add_file_documents(self, file_path: str, save_to_disk: bool = True):
        """添加单个文件文档 (PDF, Word, TXT, JSON)"""
        print(f"正在处理文档：{file_path}")

        try:
        
            # 1.json文件处理，案例库分块专用，固定分块
            if file_path.lower().endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                records = [data] if isinstance(data, dict) else data
                texts_to_add = []
                pids_to_add = []
                for record in records:
                    pid = str(record.get('pid', '')) or None
                    text_parts = []
                    if 'fact' in record and record['fact'].strip():
                        text_parts.append("【案件事实】" + record['fact'])
                    if 'reason' in record and record['reason'].strip():
                        text_parts.append("【判决理由】" + record['reason'])
                    if 'result' in record and record['result'].strip():
                        text_parts.append("【判决结果】" + record['result'])
                    if 'charge' in record and record['charge']:
                        charges = "、".join(record['charge'])
                        text_parts.append(f"【涉及罪名】{charges}")
                    if 'article' in record and record['article']:
                        article_refs = "、".join([f"第{num}条" for num in record['article']])
                        text_parts.append(f"【引用法条】《中华人民共和国刑法》{article_refs}")
                    if 'qw' in record and record['qw'].strip():
                        text_parts.append("【裁判文书全文】" + record['qw'][:10000])

                    full_text = "\n".join(text_parts).strip()
                    if not full_text:
                        continue

                    if len(full_text) > 1300:
                        legal_splitter = GeneralDocumentSplitter(chunk_size=1200, chunk_overlap=250)
                        sub_chunks = legal_splitter.split_text(full_text)
                        texts_to_add.extend(sub_chunks)
                        pids_to_add.extend([pid] * len(sub_chunks))
                    else:
                        texts_to_add.append(full_text)
                        pids_to_add.append(pid)

                print(f"已处理{file_path}，生成 {len(texts_to_add)} 个文本块")
                self.add_documents(texts_to_add, pids=pids_to_add, save_to_disk=save_to_disk)
                return



            
            
            # ======================
            # 2. TXT 文件处理 (法条库专用)
            # ======================
            if file_path.lower().endswith('.txt'):
                print(f"📜 检测到 TXT 文件，尝试按【一行一条法条】模式解析：{file_path}")
                
                texts_to_add = []
                empty_lines = 0
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        clean_line = line.strip()
                        
                        # 跳过空行
                        if not clean_line:
                            empty_lines += 1
                            continue
                        
                        # 🔥 关键策略：直接整行作为一个 Chunk，不做任何截断！
                        # 确保法条的完整性（标题 + 内容都在一行里）
                        texts_to_add.append(clean_line)
                
                if texts_to_add:
                    print(f"✅ [TXT] 解析成功！共读取 {len(texts_to_add)} 条法条 (跳过 {empty_lines} 个空行)")
                    print(f"   示例片段：{texts_to_add[0][:50]}...")
                    self.add_documents(texts_to_add, save_to_disk=save_to_disk)
                else:
                    print(f"⚠️ [TXT] 文件 {file_path} 似乎为空或格式不正确，未提取到内容。")
                
                return

            # ======================
            # 3. 其他文件 (PDF, Word 等) - 降级使用通用逻辑
            # ======================
            print(f"📄 处理非 TXT/JSON 文档，使用通用分块策略：{file_path}")
            structured_chunks = self.document_processor.process_document(file_path)
            texts_to_add = []
            
            for chunk in structured_chunks:
                full_text = chunk['full_text']
                # 只有对于非结构化长文才进行切割
                if len(full_text) > 500:
                    legal_splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
                    sub_chunks = legal_splitter.split_text(full_text)
                    texts_to_add.extend(sub_chunks)
                else:
                    texts_to_add.append(full_text)

            print(f"✅ [通用] 从文档中提取了 {len(structured_chunks)} 个块，生成 {len(texts_to_add)} 个文本块")
            if texts_to_add:
                self.add_documents(texts_to_add, save_to_disk=save_to_disk)

        except Exception as e:
            print(f"❌ 文档处理失败：{e}")
            import traceback
            traceback.print_exc()
            # 仅在严重错误时尝试 fallback
            self._fallback_add_documents(file_path, save_to_disk)

    def _fallback_add_documents(self, file_path: str, save_to_disk: bool = True):
        """回退到普通分块策略"""
        print(f"使用普通分块策略处理：{file_path}")

        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
        elif file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            print(f"不支持的文件格式：{file_path}")
            return

        pages = loader.load()
        documents = self.general_splitter.split_documents(pages)
        texts = [doc.page_content for doc in documents]
        self.add_documents(texts, save_to_disk=save_to_disk)


    def add_folder_documents(self, folder_path: str, save_to_disk: bool = True):
        """添加文件夹中的所有文档 (优化版：JSON 批量处理)"""
        supported_extensions = ('.pdf', '.doc', '.docx', '.txt', '.json')

        if not os.path.exists(folder_path):
            print(f"文件夹不存在：{folder_path}")
            return

        json_files = []
        other_files = []
        
        # 1. 先分类文件
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_extensions):
                file_path = os.path.join(folder_path, filename)
                if filename.lower().endswith('.json'):
                    json_files.append(file_path)
                else:
                    other_files.append(file_path)

        all_json_chunks = []
        all_json_pids = []
        
        # 2. 【优化点】阶段一：仅内存解析所有 JSON 文件，不构建索引
        if json_files:
            print(f"\n🚀 发现 {len(json_files)} 个 JSON 文件，启动高速批量模式...")
            for i, file_path in enumerate(json_files):
                try:
            
                    #基于字段的分块
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                
                    records = [data] if isinstance(data, dict) else data
                
                    for record in records:
                        doc_id = record.get('pid')
                        if not doc_id:
                            print(f"Warning: 跳过一条没有 pid 的记录: {record.get('id', 'unknown')}")
                            continue
                        text_parts = []
                        if 'fact' in record and record['fact'].strip():
                            text_parts.append("【案件事实】" + record['fact'])
                        if 'reason' in record and record['reason'].strip():
                            text_parts.append("【判决理由】" + record['reason'])
                        if 'result' in record and record['result'].strip():
                            text_parts.append("【判决结果】" + record['result'])
                        if 'charge' in record and record['charge']:
                            charges = "、".join(record['charge'])
                            text_parts.append(f"【涉及罪名】{charges}")
                        if 'article' in record and record['article']:
                            article_refs = "、".join([f"第{num}条" for num in record['article']])
                            text_parts.append(f"【引用法条】《中华人民共和国刑法》{article_refs}")
                        if 'qw' in record and record['qw'].strip():
                            text_parts.append("【裁判文书全文】" + record['qw'])

                        for text in text_parts:
                            if not text:
                                continue
                            if len(text) > 1300:
                                legal_splitter = GeneralDocumentSplitter(chunk_size=1200, chunk_overlap=200)
                                sub_chunks = legal_splitter.split_text(text)
                                all_json_chunks.extend(sub_chunks)
                                all_json_pids.extend([str(doc_id)]*len(sub_chunks))
                            else:
                                all_json_chunks.append(text)
                                all_json_pids.append(str(doc_id))
                       
                        
                    # 简单的进度打印
                    if (i + 1) % 1000 == 0:
                        print(f"   已解析 {i+1}/{len(json_files)} 个 JSON 文件 (当前累积块数：{len(all_json_chunks)})")
                except Exception as e:
                    print(f"解析 {file_path} 失败：{e}")
                    continue
            print(f"✅ JSON 解析完成，共获得 {len(all_json_chunks)} 个文本块。")

            # 3. 【优化点】阶段二：一次性将所有 JSON 块加入索引
            if all_json_chunks:
                print("⏳ 正在一次性构建 JSON 数据的向量和 BM25 索引 (GPU Full Speed)...")
                # 此时 vector_db 应该是 None (如果是全新构建)，直接走 from_embeddings 分支，速度极快
                self.add_documents(all_json_chunks, pids=all_json_pids, save_to_disk=False) 

        # 4. 处理其他格式文件 (PDF/Word 等)，保持原有逐个处理逻辑
        # 这些文件通常数量少，或者解析慢，逐个处理没问题
        if other_files:
            print(f"\n📄 开始处理 {len(other_files)} 个非 JSON 文件...")
            for file_path in other_files:
                # 这里调用原有的 add_file_documents，但确保 save_to_disk=False
                # 注意：你需要确保 add_file_documents 能正确处理非 JSON 逻辑
                # 如果 add_file_documents 里只有 JSON 逻辑，你需要把 PDF 解析逻辑移出来或补全
                self.add_file_documents(file_path, save_to_disk=False)

        # 5. 最终统一保存
        if save_to_disk:
            print("\n💾 所有文件处理完毕，正在保存最终索引...")
            # 移除 force_rebuild，因为 add_documents 内部已经正确添加了数据
            # 如果 BM25 实现依赖 force_rebuild 来 finalize，则保留，否则不需要
            # 大多数 BM25 实现 (如 rank-bm25) 在 add_documents 后直接可用，或者需要在初始化时构建
            # 假设你的 bm25_retriever 需要 finalize:
            if hasattr(self.bm25_retriever, 'force_rebuild'):
                 self.bm25_retriever.force_rebuild()
                 
            self.save_vector_db()
            self.bm25_retriever.save_index()
            print("🎉 全部完成！")

    def save_vector_db(self):
        """保存向量数据库到本地"""
        if self.vector_db is not None:
            self.vector_db.save_local(self.db_path)
            print(f"向量数据库已保存到：{self.db_path}")

    def load_vector_db(self):
        """从本地加载向量数据库"""
        try:
            self.vector_db = FAISS.load_local(
                self.db_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print(f"向量数据库已从 {self.db_path} 加载")
        except Exception as e:
            print(f"❌ 向量库加载失败：{e}")
            self.vector_db = None

    def hybrid_retrieve_documents(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Optional[str]]]:
        """混合检索：RRF融合向量检索 + BM25检索，返回 (text, score, pid)"""
        k = 60  # RRF常数
        retrieve_k = top_k * 5  # 多检索一些用于融合

        # 1. 向量检索（L2距离，越小越好，排序已由FAISS保证）
        vec_ranked = []  # [(text, pid), ...] 按相似度降序
        try:
            if self.vector_db is not None:
                vector_results = self.vector_db.similarity_search_with_score(query, k=retrieve_k)
                for doc, score in vector_results:
                    pid = doc.metadata.get("pid") if doc.metadata else None
                    vec_ranked.append((doc.page_content, pid))
                print(f"向量检索返回 {len(vec_ranked)} 个结果")
        except Exception as e:
            print(f"向量检索失败: {e}")

        # 2. BM25检索（分数越大越好，已排序）
        bm25_ranked = []
        try:
            bm25_results = self.bm25_retriever.search(query, top_k=retrieve_k)
            for doc, score, pid in bm25_results:
                bm25_ranked.append((doc, pid))
            print(f"BM25检索返回 {len(bm25_ranked)} 个结果")
        except Exception as e:
            print(f"BM25检索失败: {e}")

        # 3. RRF融合（按排名融合，不依赖分数值）
        fused = {}  # text -> (rrf_score, pid)
        for rank, (text, pid) in enumerate(bm25_ranked):
            score = self.bm25_weight / (k + rank + 1)
            fused[text] = (score, pid)
        for rank, (text, pid) in enumerate(vec_ranked):
            score = self.vector_weight / (k + rank + 1)
            if text in fused:
                old_score, old_pid = fused[text]
                fused[text] = (old_score + score, old_pid or pid)
            else:
                fused[text] = (score, pid)

        sorted_results = sorted(fused.items(), key=lambda x: x[1][0], reverse=True)
        final_results = [(doc, score, pid) for doc, (score, pid) in sorted_results[:top_k]]
        print(f"混合检索融合后返回 {len(final_results)} 个结果")

        return final_results


    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Optional[str]]]:
        """混合检索 (核心检索接口)，返回 (text, score, pid)"""

        enhanced_query = self.enhancer.enhance(query)

        hybrid_results = self.hybrid_retrieve_documents(enhanced_query, top_k=top_k)
        if not hybrid_results:
            print("混合检索未返回任何结果")
            return []

        print(f"混合检索返回 {len(hybrid_results)} 个最终结果")
        return hybrid_results[:top_k]


    def get_document_count(self) -> int:
        """获取向量数据库中的文档数量"""
        if self.vector_db is None:
            return 0
        return self.vector_db.index.ntotal if hasattr(self.vector_db.index, 'ntotal') else 0

    def get_bm25_document_count(self) -> int:
        """获取 BM25 索引中的文档数量"""
        return self.bm25_retriever.get_document_count()

    def get_retrieval_stats(self) -> dict:
        """获取检索统计信息"""
        return {
            "vector_documents": self.get_document_count(),
            "bm25_documents": self.bm25_retriever.get_document_count(),
            "reranker_enabled": self.reranker is not None
        }

    # ======================
    # 引用溯源：按需回源 + LRU 缓存
    # ======================

    def get_case_record(self, pid: str) -> dict | None:
        """根据 pid 读取案例 JSON 原始字段，供前端展示纪检案例详情。"""
        pid = str(pid)
        if pid in self._record_lru_cache:
            self._record_lru_cache.move_to_end(pid)
            return self._record_lru_cache[pid]

        if not self.source_dir:
            return None

        file_path = os.path.join(self.source_dir, f"{pid}.json")
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            if not isinstance(record, dict):
                return None

            self._record_lru_cache[pid] = record
            if len(self._record_lru_cache) > self._lru_max_size:
                self._record_lru_cache.popitem(last=False)
            return record
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 案例字段加载失败 (pid={pid}): {e}")
            return None

    def get_guidance_record(self, pid: str) -> dict | None:
        """根据 pid 读取办案规范 JSON 字段。"""
        return self.get_case_record(pid)

    def get_qa_record(self, pid: str) -> dict | None:
        """根据 pid 读取纪检监察业务问答 JSON 字段。"""
        return self.get_case_record(pid)

    @staticmethod
    def _build_full_text(record: dict) -> str:
        """从原始 JSON 记录重建完整文档文本。
        与 add_file_documents / add_folder_documents 中的 full_text 构建逻辑保持一致，
        但 qw 字段不截断，用于回源展示完整原文。"""
        text_parts = []
        if record.get('fact', '').strip():
            text_parts.append("【案件事实】" + record['fact'])
        if record.get('reason', '').strip():
            text_parts.append("【判决理由】" + record['reason'])
        if record.get('result', '').strip():
            text_parts.append("【判决结果】" + record['result'])
        if record.get('charge'):
            charges = "、".join(record['charge'])
            text_parts.append(f"【涉及罪名】{charges}")
        if record.get('article'):
            article_refs = "、".join([f"第{num}条" for num in record['article']])
            text_parts.append(f"【引用法条】《中华人民共和国刑法》{article_refs}")
        if record.get('qw', '').strip():
            text_parts.append("【裁判文书全文】" + record['qw'])  # 不截断，展示完整原文
        return "\n".join(text_parts).strip()

    def get_parent_document(self, pid: str) -> str | None:
        """根据 pid 从原始 JSON 文件按需加载完整父文档。

        基于 LeCaRD 数据集的命名约定 ({pid}.json)，
        通过 pid 直接推导文件路径，无需额外索引结构。
        查 LRU Cache → 拼接路径 → 读文件 → 重建文本 → 写入 Cache → 返回。

        参数:
          pid: 案例 ID (字符串形式，如 "0", "10210")
        返回:
          完整父文档文本，或 None（pid 无效 / 源文件缺失 / 源目录未配置）
        """
        # ① 查 LRU Cache（最快路径）
        if pid in self._lru_cache:
            self._lru_cache.move_to_end(pid)
            return self._lru_cache[pid]

        # ② 源目录未配置，无法回源
        if not self.source_dir:
            return None

        # ③ 基于命名约定拼接路径
        file_path = os.path.join(self.source_dir, f"{pid}.json")
        if not os.path.exists(file_path):
            return None

        # ④ 读取原始 JSON 并重建完整文档
        try:
            record = self.get_case_record(pid)
            if not record:
                return None
            full_text = self._build_full_text(record)
            if not full_text:
                return None

            # ⑤ 写入 LRU Cache（超过容量则淘汰最旧条目）
            self._lru_cache[pid] = full_text
            if len(self._lru_cache) > self._lru_max_size:
                self._lru_cache.popitem(last=False)

            return full_text

        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 回源加载失败 (pid={pid}): {e}")
            return None
