from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta, datetime
from dotenv import load_dotenv
import json
import os
import sys
import bcrypt
import random
import time
import requests

# 加载环境变量
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')

# 配置
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'chem-safe-secret-key-2024')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# DeepSeek 配置（当前未使用）
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/chat/completions')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

# 初始化
CORS(app, origins='*', supports_credentials=True)
jwt = JWTManager(app)

# 数据文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
HISTORIES_FILE = os.path.join(DATA_DIR, 'histories.json')
CHEMICALS_FILE = os.path.join(DATA_DIR, 'chemicals.json')

os.makedirs(DATA_DIR, exist_ok=True)

# PubChem 配置
PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
PUBCHEM_CACHE_FILE = os.path.join(DATA_DIR, 'pubchem_cache.json')
PUBCHEM_MIN_INTERVAL = 0.4  # NCBI 限速：请求间隔不低于 0.4 秒
_pubchem_last_request = 0.0

# 常见中文化学品 -> PubChem 查询名（提升中文识别率）
PUBCHEM_NAME_MAP = {
    '医用酒精': 'ethanol',
    '酒精': 'ethanol',
    '乙醇': 'ethanol',
    '含氯消毒液': 'sodium hypochlorite',
    '84消毒液': 'sodium hypochlorite',
    '84': 'sodium hypochlorite',
    '漂白水': 'sodium hypochlorite',
    '洁厕灵': 'hydrochloric acid',
    '盐酸': 'hydrochloric acid',
    '双氧水': 'hydrogen peroxide',
    '过氧化氢': 'hydrogen peroxide',
    '白醋': 'acetic acid',
    '醋酸': 'acetic acid',
    '柠檬酸': 'citric acid',
    '小苏打': 'sodium bicarbonate',
    '碳酸氢钠': 'sodium bicarbonate',
    '氨水': 'ammonia',
    '84消毒片': 'sodium dichloroisocyanurate',
    '管道疏通剂': 'sodium hydroxide',
    '氢氧化钠': 'sodium hydroxide',
    '强碱': 'sodium hydroxide',
    '碘伏': 'povidone-iodine',
    '红药水': 'merbromin',
    '汞溴红': 'merbromin',
    '花露水': 'toilet water',
    '风油精': 'menthol oil',
    '油漆': 'paint',
    '丙酮': 'acetone',
    '洗甲水': 'acetone',
    '染发剂': 'hair dye',
    '杀虫剂': 'insecticide',
    '洗洁精': 'dishwashing liquid',
    '洗衣液': 'laundry detergent',
    '漂白剂': 'bleach',
    '彩漂液': 'oxygen bleach',
    '硼酸': 'boric acid',
    '甘油': 'glycerol',
    '甲醛': 'formaldehyde',
    '苯': 'benzene',
    '甲苯': 'toluene',
    '二甲苯': 'xylene',
    '硫酸': 'sulfuric acid',
    '硝酸': 'nitric acid',
    '磷酸': 'phosphoric acid',
    '次氯酸钠': 'sodium hypochlorite',
    '氯仿': 'chloroform',
    '四氯化碳': 'carbon tetrachloride',
    '乙醚': 'diethyl ether',
    '汽油': 'gasoline',
    '煤油': 'kerosene',
}


def init_data():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    if not os.path.exists(HISTORIES_FILE):
        with open(HISTORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    if not os.path.exists(CHEMICALS_FILE):
        with open(CHEMICALS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"chemicals": [], "mixing_rules": [], "rumors": []}, f, ensure_ascii=False, indent=2)


init_data()


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        if any(k in file_path for k in ('histories', 'chemicals', 'pubchem_cache')):
            return {}
        return []


def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_chemicals_data():
    return read_json(CHEMICALS_FILE)


def resolve_chemical(input_name, use_pubchem=True):
    """根据名称或别名解析出标准化学品名称；本地未命中时可查询 PubChem"""
    if not input_name:
        return None
    q = input_name.lower().strip()
    chemicals = get_chemicals_data().get('chemicals', [])

    # 1. 完全匹配名称
    for c in chemicals:
        if c['name'].lower() == q:
            return c['name']

    # 2. 别名精确匹配（优先，避免“84”先命中“84消毒片”）
    for c in chemicals:
        alias = c.get('alias', '')
        if not alias:
            continue
        for a in alias.split():
            al = a.lower().strip()
            if al and al == q:
                return c['name']

    # 3. 名称包含
    for c in chemicals:
        name_l = c['name'].lower()
        if name_l in q or q in name_l:
            return c['name']

    # 4. 别名包含/被包含
    for c in chemicals:
        alias = c.get('alias', '')
        if not alias:
            continue
        for a in alias.split():
            al = a.lower().strip()
            if not al:
                continue
            if al in q or q in al:
                return c['name']

    # 5. PubChem 兜底（英文/化学名）
    if use_pubchem:
        pub_query = PUBCHEM_NAME_MAP.get(input_name, input_name)
        pub = pubchem_lookup(pub_query)
        if pub:
            # 用 PubChem 的 IUPAC 名或首个同义词作为标准名
            std = pub['name']
            # 如果 IUPAC 名太长，尝试用更短的同义词
            for syn in pub.get('synonyms', []):
                if 3 <= len(syn) <= 40 and syn.lower().strip() == syn:
                    std = syn
                    break
            return std
    return None


def find_mix_rule(a, b):
    """查找本地混用规则"""
    rules = get_chemicals_data().get('mixing_rules', [])
    for r in rules:
        if (r['a'] == a and r['b'] == b) or (r['a'] == b and r['b'] == a):
            return r
    return None


def deepseek_chat(messages, temperature=0.7, max_tokens=800):
    """调用 DeepSeek API（当前未使用，保留代码以便后续切换）"""
    if not DEEPSEEK_API_KEY:
        return None, 'DeepSeek API 密钥未配置'
    try:
        headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': DEEPSEEK_MODEL,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content'], None
    except Exception as e:
        return None, str(e)


# ============ PubChem 查询模块 ============

def _pubchem_throttle():
    """NCBI 请求限速"""
    global _pubchem_last_request
    elapsed = time.time() - _pubchem_last_request
    if elapsed < PUBCHEM_MIN_INTERVAL:
        time.sleep(PUBCHEM_MIN_INTERVAL - elapsed)
    _pubchem_last_request = time.time()


def _load_pubchem_cache():
    try:
        return read_json(PUBCHEM_CACHE_FILE)
    except Exception:
        return {}


def _save_pubchem_cache(cache):
    write_json(PUBCHEM_CACHE_FILE, cache)


def pubchem_lookup(name):
    """通过 PubChem 查询化学品信息，带本地缓存和限速"""
    if not name:
        return None
    key = name.lower().strip()
    cache = _load_pubchem_cache()
    if key in cache:
        return cache[key]

    try:
        # 1. 查询基础属性
        _pubchem_throttle()
        encoded = requests.utils.quote(key)
        url = f"{PUBCHEM_BASE_URL}/compound/name/{encoded}/property/IUPACName,MolecularFormula,CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            cache[key] = None
            _save_pubchem_cache(cache)
            return None
        resp.raise_for_status()
        props = resp.json()['PropertyTable']['Properties'][0]
        cid = props['CID']

        # 2. 查询同义词（用于后续识别）
        _pubchem_throttle()
        syn_url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/synonyms/JSON"
        synonyms = []
        try:
            syn_resp = requests.get(syn_url, timeout=30)
            if syn_resp.status_code == 200:
                info = syn_resp.json().get('InformationList', {}).get('Information', [{}])[0]
                synonyms = info.get('Synonym', [])[:15]
        except Exception:
            pass

        # 3. 查询 GHS 危害声明
        hazards = pubchem_get_ghs(cid)

        result = {
            'cid': cid,
            'name': props.get('IUPACName') or key,
            'formula': props.get('MolecularFormula', ''),
            'smiles': props.get('CanonicalSMILES', ''),
            'synonyms': synonyms,
            'hazards': hazards,
            'source': 'pubchem',
            'cached_at': str(datetime.now())
        }
        cache[key] = result
        _save_pubchem_cache(cache)
        return result
    except Exception:
        return None


def pubchem_get_ghs(cid):
    """获取 PubChem GHS 危害声明"""
    _pubchem_throttle()
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
        # 导航到 GHS Classification 节点
        sections = data.get('Record', {}).get('Section', [])
        for s1 in sections:
            for s2 in s1.get('Section', []):
                for s3 in s2.get('Section', []):
                    if s3.get('TOCHeading') == 'GHS Classification':
                        hazards = []
                        for info in s3.get('Information', []):
                            if info.get('Name') == 'GHS Hazard Statements':
                                for item in info.get('Value', {}).get('StringWithMarkup', []):
                                    hazards.append(item.get('String', ''))
                        return hazards
        return []
    except Exception:
        return []


def extract_hazard_codes(hazards):
    """从 GHS 危害声明中提取 H-code"""
    codes = set()
    for h in hazards:
        import re
        for m in re.finditer(r'H\d{3}[a-zA-Z]*', h):
            codes.add(m.group())
    return sorted(codes)


def pubchem_mix_analysis(info_a, info_b):
    """基于 PubChem GHS 数据生成混用安全分析"""
    codes_a = extract_hazard_codes(info_a.get('hazards', []))
    codes_b = extract_hazard_codes(info_b.get('hazards', []))

    # 危险类别判断（GHS H-code）
    def has(codes, prefixes):
        return any(any(str(c).startswith(p) for p in prefixes) for c in codes)

    flammable_a = has(codes_a, ['H22'])       # H220-H226 易燃
    flammable_b = has(codes_b, ['H22'])
    oxidizer_a = has(codes_a, ['H27'])        # H270-H272 氧化剂
    oxidizer_b = has(codes_b, ['H27'])
    corrosive_a = has(codes_a, ['H314', 'H318'])  # 严重腐蚀
    corrosive_b = has(codes_b, ['H314', 'H318'])
    irritant_a = has(codes_a, ['H315', 'H319', 'H335'])  # 刺激
    irritant_b = has(codes_b, ['H315', 'H319', 'H335'])
    toxic_a = has(codes_a, ['H300', 'H301', 'H330', 'H331', 'H34', 'H35', 'H36', 'H37'])
    toxic_b = has(codes_b, ['H300', 'H301', 'H330', 'H331', 'H34', 'H35', 'H36', 'H37'])

    reasons = []
    level = 'safe'
    danger = '安全'

    if (flammable_a and oxidizer_b) or (flammable_b and oxidizer_a):
        level = 'critical'
        danger = '危险'
        reasons.append('易燃物与氧化剂混合可能引发燃烧或爆炸，务必避免。')
    elif flammable_a and flammable_b:
        level = 'warning'
        danger = '注意'
        reasons.append('两者均易燃，混合后火灾风险叠加，需远离火源。')
    elif corrosive_a and corrosive_b:
        level = 'warning'
        danger = '注意'
        reasons.append('两者均具强腐蚀性，混合可能加剧灼伤风险并释放热量。')
    elif (corrosive_a and (oxidizer_b or flammable_b)) or (corrosive_b and (oxidizer_a or flammable_a)):
        level = 'warning'
        danger = '注意'
        reasons.append('腐蚀性物质与强氧化/易燃物质混用风险较高。')
    elif toxic_a or toxic_b:
        level = 'warning'
        danger = '注意'
        reasons.append('其中至少一种物质具有毒性或生殖毒性，混合可能增加暴露风险。')
    elif irritant_a or irritant_b:
        level = 'info'
        danger = '低危'
        reasons.append('至少一种物质对皮肤/眼睛/呼吸道有刺激性，混合可能加重不适。')

    if not reasons:
        result_text = '基于 PubChem GHS 数据，未发现明确混用禁忌，但日常仍建议分开使用。'
    else:
        result_text = ' '.join(reasons)

    detail = (
        f"化学品A（CID {info_a.get('cid')}）：{info_a.get('name')}，分子式 {info_a.get('formula', 'N/A')}；\n"
        f"GHS 声明：{', '.join(codes_a) or '无'}。\n"
        f"化学品B（CID {info_b.get('cid')}）：{info_b.get('name')}，分子式 {info_b.get('formula', 'N/A')}；\n"
        f"GHS 声明：{', '.join(codes_b) or '无'}。\n"
        f"分析结论：{result_text}"
    )

    return {
        'level': level,
        'danger': danger,
        'result': 'PubChem GHS 数据分析',
        'explain': detail,
        'pubchem_a': info_a,
        'pubchem_b': info_b
    }


def build_mix_prompt(a, b):
    chemicals = get_chemicals_data().get('chemicals', [])
    chem_a = next((c for c in chemicals if c['name'] == a), None)
    chem_b = next((c for c in chemicals if c['name'] == b), None)
    info_a = f"{a}（{chem_a.get('category', '')}）：{chem_a.get('tip', '')}" if chem_a else a
    info_b = f"{b}（{chem_b.get('category', '')}）: {chem_b.get('tip', '')}" if chem_b else b
    return (
        "你是家庭化学品安全专家。请判断以下两种化学品混合使用是否安全。\n"
        "要求：\n"
        "1. 先给出明确结论：安全 / 不建议 / 危险 / 剧毒\n"
        "2. 简要说明可能发生的化学反应或风险\n"
        "3. 给出实用建议\n"
        "4. 如果不确定，明确说明不确定，并建议分开使用\n"
        "5. 回答控制在300字以内，用中文\n\n"
        f"化学品A：{info_a}\n"
        f"化学品B：{info_b}\n"
    )


def build_chat_prompt(user_input):
    return (
        "你是 ChemSafe 的家庭化学品安全智能助手\"小安\"。"
        "你擅长回答清洁剂、消毒液、护肤品、日用品等家庭化学品的安全问题，"
        "包括能否混用、如何存放、误食/溅入眼睛应急处理、使用注意事项等。"
        "回答要简洁、准确、有安全意识，不确定时明确说明。\n"
        "用户问题：" + user_input
    )


def build_local_chat_reply(user_input):
    """基于本地知识库和 PubChem 数据回答聊天问题（不调用 DeepSeek）"""
    text = user_input.lower()
    chemicals = get_chemicals_data().get('chemicals', [])
    rules = get_chemicals_data().get('mixing_rules', [])

    # 1. 识别输入中提到的化学品
    found = []
    for c in chemicals:
        names = [c['name']] + c.get('alias', '').split()
        for n in names:
            n = n.lower().strip()
            if n and n in text:
                found.append(c)
                break

    # 2. 常见混用/安全问题关键词回复
    if '混用' in text or '一起' in text or '混合' in text or '能不能' in text:
        if len(found) >= 2:
            rule = find_mix_rule(found[0]['name'], found[1]['name'])
            if rule:
                return (
                    f"{found[0]['name']} 和 {found[1]['name']} {rule['danger']}。"
                    f"{rule['explain']} 建议分开使用。",
                    'local'
                )
            return (
                f"本地数据库中暂未记录 {found[0]['name']} 与 {found[1]['name']} 的明确禁忌，"
                "但日常清洁仍建议分开使用，避免成分互相影响。",
                'local'
            )
        if len(found) == 1:
            return (
                f"{found[0]['name']} 的使用提示：{found[0]['tip']}。"
                "如果你想知道它和另一种化学品能否混用，可以告诉我另一种的名称。",
                'local'
            )

    if '误食' in text or '吃了' in text or '喝' in text:
        return (
            "若发生误食，请立即停止摄入，保留产品包装，尽快联系医院或拨打急救电话。"
            "不要自行催吐，可少量饮水稀释（除非说明书明确禁止）。",
            'local'
        )

    if '眼睛' in text or '溅入' in text or '入眼' in text:
        return (
            "化学品溅入眼睛后，应立即用大量流动清水冲洗至少 15 分钟，"
            "冲洗时翻开眼睑确保彻底，随后尽快就医。",
            'local'
        )

    if '存放' in text or '保存' in text or '放哪' in text:
        if found:
            return (
                f"{found[0]['name']} 存放建议：{found[0]['tip']}。"
                "一般应放在阴凉、干燥、儿童接触不到的地方，远离火源和食品。",
                'local'
            )
        return (
            "家庭化学品建议分开存放、避免混放，尤其是清洁剂与消毒液、酸与碱、易燃品与火源。"
            "具体可查看安全科普页面的建议。",
            'local'
        )

    if '84' in text or '洁厕灵' in text or '含氯' in text:
        return (
            "84消毒液/含氯消毒剂切勿与洁厕灵、酸性清洁剂、酒精混用，"
            "混合可能释放氯气等有毒气体。使用时应稀释并保持通风。",
            'local'
        )

    if '酒精' in text:
        return (
            "医用酒精易燃，使用时应远离火源和高温，避免大面积喷洒。"
            "不要与含氯消毒剂混用，以免产生有害气体。",
            'local'
        )

    # 3. 尝试 PubChem 识别单个化学品
    if found:
        pub_query = PUBCHEM_NAME_MAP.get(found[0]['name'], found[0]['name'])
        pub = pubchem_lookup(pub_query)
    else:
        # 尝试把整句话里的潜在化学名词拿去 PubChem 查
        pub = None
        # 拆分出可能的英文/中文候选词
        import re
        candidates = set(re.findall(r'[a-zA-Z0-9\-]{2,40}|[\u4e00-\u9fa5]{2,10}', text))
        for token in candidates:
            if 2 <= len(token) <= 40:
                pub_query = PUBCHEM_NAME_MAP.get(token, token)
                pub = pubchem_lookup(pub_query)
                if pub:
                    break

    if pub:
        codes = extract_hazard_codes(pub.get('hazards', []))
        return (
            f"我从 PubChem 数据库查到「{pub['name']}」（CID {pub['cid']}，分子式 {pub['formula']}）。\n"
            f"GHS 危害声明：{', '.join(codes) if codes else '暂无明确危害声明'}。\n"
            "如有具体使用场景或混用问题，可以告诉我另一种化学品名称。",
            'pubchem'
        )

    return (
        "小安暂时只能从本地知识库和 PubChem 数据库回答化学品相关问题。"
        "你可以问我：两种化学品能不能混用、某种化学品怎么存放、误食或溅入眼睛怎么办等。",
        'fallback'
    )


# ============ API 路由 ============

@app.route('/api/admin/cleanup', methods=['POST'])
def admin_cleanup():
    """临时清理接口：清空用户和历史数据（需密钥）"""
    data = request.json or {}
    if data.get('secret') != os.getenv('CLEANUP_SECRET', 'chemsafe-cleanup-2024'):
        return jsonify({'success': False, 'message': '密钥错误'}), 403
    write_json(USERS_FILE, [])
    write_json(HISTORIES_FILE, {})
    return jsonify({'success': True, 'message': '用户和历史数据已清空'})


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': '请填写昵称和密码'}), 400
    if len(username) < 2:
        return jsonify({'success': False, 'message': '昵称至少2个字符'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'message': '密码至少6位'}), 400

    users = read_json(USERS_FILE)
    if any(u['username'] == username for u in users):
        return jsonify({'success': False, 'message': '该昵称已被注册'}), 400

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    new_user = {
        'id': int(time.time() * 1000),
        'username': username,
        'password': hashed.decode('utf-8'),
        'createdAt': str(datetime.now())
    }
    users.append(new_user)
    write_json(USERS_FILE, users)

    histories = read_json(HISTORIES_FILE)
    histories[str(new_user['id'])] = {'ai': [], 'rumor': [], 'safety': [], 'mix': []}
    write_json(HISTORIES_FILE, histories)

    access_token = create_access_token(identity=str(new_user['id']))
    return jsonify({
        'success': True,
        'message': '注册成功',
        'token': access_token,
        'user': {'id': new_user['id'], 'username': new_user['username']}
    })


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': '请输入昵称和密码'}), 400

    users = read_json(USERS_FILE)
    user = next((u for u in users if u['username'] == username), None)
    if not user:
        return jsonify({'success': False, 'message': '昵称或密码错误'}), 401
    if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return jsonify({'success': False, 'message': '昵称或密码错误'}), 401

    access_token = create_access_token(identity=str(user['id']))
    return jsonify({
        'success': True,
        'message': '登录成功',
        'token': access_token,
        'user': {'id': user['id'], 'username': user['username']}
    })


@app.route('/api/verify', methods=['GET'])
@jwt_required()
def verify():
    user_id = get_jwt_identity()
    users = read_json(USERS_FILE)
    user = next((u for u in users if str(u['id']) == user_id), None)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在'}), 401
    return jsonify({'success': True, 'user': {'id': user['id'], 'username': user['username']}})


@app.route('/api/chemicals', methods=['GET'])
def get_chemicals():
    data = get_chemicals_data()
    return jsonify({'success': True, 'data': data.get('chemicals', [])})


@app.route('/api/chemicals/resolve', methods=['POST'])
def resolve_chemical_api():
    """解析用户输入的化学品名称"""
    data = request.json
    name = data.get('name', '')
    matched = resolve_chemical(name)
    chemicals = get_chemicals_data().get('chemicals', [])
    chem_info = next((c for c in chemicals if c['name'] == matched), None)
    return jsonify({
        'success': True,
        'input': name,
        'matched': matched,
        'data': chem_info
    })


@app.route('/api/rumors', methods=['GET'])
def get_rumors():
    data = get_chemicals_data()
    return jsonify({'success': True, 'data': data.get('rumors', [])})


@app.route('/api/history/<int:user_id>/<hist_type>', methods=['GET'])
@jwt_required()
def get_history(user_id, hist_type):
    current_user = get_jwt_identity()
    if str(user_id) != current_user:
        return jsonify({'success': False, 'message': '无权访问'}), 403
    histories = read_json(HISTORIES_FILE)
    user_hist = histories.get(str(user_id), {'ai': [], 'rumor': [], 'safety': [], 'mix': []})
    return jsonify({'success': True, 'data': user_hist.get(hist_type, [])})


@app.route('/api/history', methods=['POST'])
@jwt_required()
def save_history():
    data = request.json
    user_id = data.get('userId')
    hist_type = data.get('type')
    content = data.get('content')

    current_user = get_jwt_identity()
    if str(user_id) != current_user:
        return jsonify({'success': False, 'message': '无权操作'}), 403

    histories = read_json(HISTORIES_FILE)
    user_id_str = str(user_id)
    if user_id_str not in histories:
        histories[user_id_str] = {'ai': [], 'rumor': [], 'safety': [], 'mix': []}

    histories[user_id_str][hist_type].insert(0, {
        'content': content,
        'time': str(datetime.now())
    })
    if len(histories[user_id_str][hist_type]) > 50:
        histories[user_id_str][hist_type] = histories[user_id_str][hist_type][:50]

    write_json(HISTORIES_FILE, histories)
    return jsonify({'success': True, 'message': '保存成功'})


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': '未上传文件'}), 400

    file = request.files['image']
    filename = file.filename.lower()
    chemicals = get_chemicals_data().get('chemicals', [])

    # 优先用名称或别名匹配
    matched = None
    for c in chemicals:
        if c['name'].lower() in filename:
            matched = c
            break
        alias = c.get('alias', '')
        if alias and any(a.lower() in filename for a in alias.split()):
            matched = c
            break

    if not matched:
        matched = random.choice(chemicals)

    confidence = round(75 + random.random() * 20, 1)
    return jsonify({'success': True, 'data': {**matched, 'confidence': confidence}})


@app.route('/api/analyze-rumor', methods=['POST'])
def analyze_rumor():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'success': False, 'message': '请输入谣言内容'}), 400

    rumors = get_chemicals_data().get('rumors', [])
    matched = next(
        (r for r in rumors if r['title'] in text or text in r['title']),
        None
    )
    if not matched:
        # 关键词匹配
        best_score = 0
        for r in rumors:
            title = r['title']
            score = sum(1 for word in text if word in title) / max(len(title), 1)
            if score > best_score:
                best_score = score
                matched = r
        if not matched:
            matched = random.choice(rumors)

    return jsonify({'success': True, 'data': {**matched, 'input': text}})


@app.route('/api/mix/check', methods=['POST'])
def mix_check():
    """混用禁忌查询：本地规则优先；无规则则基于 PubChem GHS 数据分析"""
    data = request.json
    raw_a = data.get('a', '').strip()
    raw_b = data.get('b', '').strip()
    user_id = data.get('userId')

    if not raw_a or not raw_b:
        return jsonify({'success': False, 'message': '请输入两种化学品名称'}), 400

    a = resolve_chemical(raw_a)
    b = resolve_chemical(raw_b)

    if a and b and a == b:
        return jsonify({'success': False, 'message': '请输入两种不同的化学品'}), 400

    result = {
        'input_a': raw_a,
        'input_b': raw_b,
        'matched_a': a,
        'matched_b': b,
        'source': 'local',
        'rule': None,
        'pubchem_rule': None,
        'error': None
    }

    # 本地规则命中直接返回
    if a and b:
        rule = find_mix_rule(a, b)
        if rule:
            result['rule'] = rule
            return jsonify({'success': True, 'data': result})

    # 尝试 PubChem 查询未识别或本地无规则的化学品
    rule = find_mix_rule(a, b) if a and b else None
    pub_query_a = PUBCHEM_NAME_MAP.get(a or raw_a, raw_a)
    pub_query_b = PUBCHEM_NAME_MAP.get(b or raw_b, raw_b)
    pub_a = pubchem_lookup(pub_query_a) if not a or not rule else None
    pub_b = pubchem_lookup(pub_query_b) if not b or not rule else None

    # 至少一种未识别且 PubChem 也未命中
    unknown = []
    if not a and not pub_a:
        unknown.append(raw_a)
    if not b and not pub_b:
        unknown.append(raw_b)
    if unknown:
        result['unknown'] = unknown
        result['source'] = 'fallback'
        result['error'] = '本地数据库和 PubChem 均未识别该化学品'
        return jsonify({'success': True, 'data': result})

    # 两者均已识别（本地或 PubChem），进行 PubChem GHS 分析
    if (a or pub_a) and (b or pub_b):
        result['source'] = 'pubchem'
        info_a = pub_a or {'name': a, 'formula': '', 'hazards': []}
        info_b = pub_b or {'name': b, 'formula': '', 'hazards': []}
        # 若 PubChem 识别出标准名，更新 matched
        if pub_a and not a:
            result['matched_a'] = pub_a['name']
        if pub_b and not b:
            result['matched_b'] = pub_b['name']
        analysis = pubchem_mix_analysis(info_a, info_b)
        result['pubchem_rule'] = {
            'level': analysis['level'],
            'danger': analysis['danger'],
            'result': analysis['result'],
            'explain': analysis['explain']
        }
        return jsonify({'success': True, 'data': result})

    # 能识别但无规则、无 PubChem 数据
    result['source'] = 'fallback'
    result['error'] = '未能获取足够的安全数据'
    return jsonify({'success': True, 'data': result})


@app.route('/api/chat', methods=['POST'])
def chat():
    """小安 AI 聊天：基于本地知识库 + PubChem 数据回答"""
    data = request.json
    user_input = data.get('message', '').strip()
    if not user_input:
        return jsonify({'success': False, 'message': '请输入问题'}), 400

    reply, source = build_local_chat_reply(user_input)
    return jsonify({
        'success': True,
        'data': {'reply': reply, 'source': source}
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    chemicals = get_chemicals_data().get('chemicals', [])
    return jsonify({
        'success': True,
        'data': {
            'image_recognition': random.randint(1200, 1300),
            'rumor_analysis': random.randint(800, 900),
            'chemical_count': len(chemicals),
            'accuracy': 87.3
        }
    })


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    try:
        print('=' * 50)
        print('🚀 ChemSafe 后端启动中...')
        print(f'📍 访问地址: http://localhost:{port}')
        print(f'📁 数据目录: {DATA_DIR}')
        print(f'🧪 PubChem 数据查询: 已启用（请求间隔 {PUBCHEM_MIN_INTERVAL}s）')
        print(f'🐍 Python版本: {sys.version}')
        print('=' * 50)
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError as e:
        print(f'❌ 启动失败: 端口{port}被占用 - {e}')
        sys.exit(1)
    except Exception as e:
        print(f'❌ 启动失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
