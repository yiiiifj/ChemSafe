from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta, datetime
from dotenv import load_dotenv
import json
import os
import re
import sys
import bcrypt
import random
import time
import threading
import requests
import tempfile
import smtplib
import secrets
import base64
import io
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image

# OCR 引擎：优先 RapidOCR（纯 pip 安装、自带模型、无需系统二进制，中文效果好），
# 其次回退到 pytesseract（需要系统另装 Tesseract 二进制）。
RAPIDOCR_AVAILABLE = False
try:
    from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    RAPIDOCR_AVAILABLE = True
except Exception:
    RapidOCR = None

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

OCR_AVAILABLE = RAPIDOCR_AVAILABLE or PYTESSERACT_AVAILABLE
if RAPIDOCR_AVAILABLE:
    OCR_ENGINE_NAME = 'rapidocr'
elif PYTESSERACT_AVAILABLE:
    OCR_ENGINE_NAME = 'pytesseract'
else:
    OCR_ENGINE_NAME = None

# 加载环境变量
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')

# 配置
_jwt_secret = (os.getenv('JWT_SECRET_KEY') or '').strip()
if not _jwt_secret:
    _jwt_secret = secrets.token_urlsafe(48)
    print('[warn] 未设置 JWT_SECRET_KEY，已生成临时随机密钥：进程重启后所有登录态失效。'
          '生产环境请在环境变量中配置固定密钥。')
app.config['JWT_SECRET_KEY'] = _jwt_secret
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# JSON 直接输出 UTF-8 中文，不再转义成 \uXXXX：中文内容为主的接口体积约减半，也方便调试时肉眼看。
if hasattr(app, 'json'):
    app.json.ensure_ascii = False

# DeepSeek 配置（文本模型）
# 注意 strip()：从 Render 面板粘贴的密钥常会带上末尾换行/空格，不 strip 会导致
# 请求头里出现非法字符（实测报 "Invalid ... in header value ... '\n'"）。
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '').strip()
DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/chat/completions').strip()
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat').strip()

# 视觉模型配置（图片识别专用）
# 注意：deepseek-chat 是文本模型，不支持图片输入，不能用作视觉识别。
# 需配置任意 OpenAI 兼容的多模态接口，例如：
#   通义千问  VISION_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions  VISION_MODEL=qwen-vl-max
#   智谱      VISION_API_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions             VISION_MODEL=glm-4v
#   豆包/OpenAI 同理。未配置时图片识别自动降级为 OCR + 文件名匹配。
VISION_API_URL = os.getenv('VISION_API_URL', '').strip()
VISION_API_KEY = os.getenv('VISION_API_KEY', '').strip()
VISION_MODEL = os.getenv('VISION_MODEL', '').strip()
VISION_ENABLED = bool(VISION_API_URL and VISION_API_KEY and VISION_MODEL)

# 邮件/SMTP 配置（用于邮箱找回密码）
SMTP_SERVER = os.getenv('SMTP_SERVER', '')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
SMTP_FROM = os.getenv('SMTP_FROM', SMTP_USER)
APP_BASE_URL = os.getenv('APP_BASE_URL', 'http://localhost:5000')

# 是否信任反向代理传来的 X-Forwarded-For（Render / nginx 等部署时为 1；
# 直连无代理时应设为 0，否则客户端可伪造 IP 绕过限流）
TRUST_PROXY = os.getenv('TRUST_PROXY', '1') == '1'

# 初始化
# 认证走 Authorization: Bearer 头而非 Cookie，因此不需要 supports_credentials，
# 且绝不应在携带凭据时使用通配来源。默认仅放开 /api/*，来源可用 CORS_ORIGINS 覆盖。
_cors_origins = [o.strip() for o in os.getenv('CORS_ORIGINS', '*').split(',') if o.strip()] or ['*']
CORS(app, resources={r'/api/*': {'origins': _cors_origins}})
jwt = JWTManager(app)


# ============ JWT 错误统一返回 JSON（前端 apiFetch 直接读 message 字段）============

@jwt.unauthorized_loader
def _jwt_missing(reason):
    return jsonify({'success': False, 'message': '请先登录后再使用该功能'}), 401


@jwt.expired_token_loader
def _jwt_expired(header, payload):
    return jsonify({'success': False, 'message': '登录已过期，请重新登录'}), 401


@jwt.invalid_token_loader
def _jwt_invalid(reason):
    return jsonify({'success': False, 'message': '登录凭证无效，请重新登录'}), 401


# ============ 简易内存限流 ============
# 目的：防止接口被刷导致 DeepSeek / 视觉模型费用失控、密码重置邮件被滥用。
# 说明：内存实现只在单进程内生效（当前 gunicorn --workers 1，恰好够用）；
#       若将来横向扩容，需要换成 Redis 等共享存储。
RATE_LIMITS = {
    '/api/login': (20, 300),            # 登录：防暴力破解
    '/api/register': (10, 300),
    '/api/forgot-password': (5, 3600),  # 找回密码：防刷邮件
    '/api/upload': (20, 300),           # 以下都花钱（视觉模型 / DeepSeek）
    '/api/chat': (20, 300),
    '/api/chat/product': (20, 300),
    '/api/analyze-rumor': (20, 300),
    '/api/mix/check': (30, 300),
    '/api/feedback': (60, 300),
}

_rate_buckets = {}


def _client_ip():
    """取客户端 IP。

    取 X-Forwarded-For 的**最后一段**：代理是「追加」而非「替换」该头，
    因此靠前的内容是客户端可伪造的，最后一段才是代理亲眼看到的真实地址。
    """
    if TRUST_PROXY:
        xff = request.headers.get('X-Forwarded-For', '')
        parts = [p.strip() for p in xff.split(',') if p.strip()]
        if parts:
            return parts[-1]
    return request.remote_addr or 'unknown'


@app.before_request
def _apply_rate_limit():
    rule = RATE_LIMITS.get(request.path)
    if not rule or request.method != 'POST':
        return None
    limit, window = rule
    key = (request.path, _client_ip())
    now = time.time()
    with _write_lock:
        # 桶数量保护：极端情况下清掉已过期的键，避免内存无限增长
        if len(_rate_buckets) > 5000:
            for k in [k for k, v in _rate_buckets.items() if not v or now - v[-1] > 3600]:
                _rate_buckets.pop(k, None)
        hits = [t for t in _rate_buckets.get(key, []) if now - t < window]
        if len(hits) >= limit:
            _rate_buckets[key] = hits
            retry = max(1, int(window - (now - hits[0])) + 1)
            resp = jsonify({'success': False,
                            'message': f'操作过于频繁，请在 {retry} 秒后再试'})
            resp.status_code = 429
            resp.headers['Retry-After'] = str(retry)
            return resp
        hits.append(now)
        _rate_buckets[key] = hits
    return None

# 数据文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
HISTORIES_FILE = os.path.join(DATA_DIR, 'histories.json')
CHEMICALS_FILE = os.path.join(DATA_DIR, 'chemicals.json')
RESET_TOKENS_FILE = os.path.join(DATA_DIR, 'reset_tokens.json')
FEEDBACK_FILE = os.path.join(DATA_DIR, 'feedback.json')

# 历史记录允许的类型白名单（防止非法 type 写入导致 KeyError）
VALID_HIST_TYPES = ('ai', 'rumor', 'safety', 'mix')

os.makedirs(DATA_DIR, exist_ok=True)

# PubChem 配置
PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
PUBCHEM_CACHE_FILE = os.path.join(DATA_DIR, 'pubchem_cache.json')
PUBCHEM_MIN_INTERVAL = 0.4  # NCBI 限速：请求间隔不低于 0.4 秒
_pubchem_last_request = 0.0

# DeepSeek 中文建议缓存（避免重复调用、节省费用）
ADVICE_CACHE_FILE = os.path.join(DATA_DIR, 'deepseek_advice_cache.json')


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
    '阿司匹林': 'aspirin',
    '乙酰水杨酸': 'aspirin',
    '布洛芬': 'ibuprofen',
    '对乙酰氨基酚': 'acetaminophen',
    '扑热息痛': 'acetaminophen',
    '水杨酸': 'salicylic acid',
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

    if not os.path.exists(RESET_TOKENS_FILE):
        with open(RESET_TOKENS_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)


init_data()


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        if any(k in file_path for k in ('histories', 'chemicals', 'pubchem_cache',
                                        'deepseek_advice_cache', 'reset_tokens')):
            return {}
        return []


# 写锁：串行化进程内的文件写入（当前部署 gunicorn --workers 1，单进程即全部）
_write_lock = threading.Lock()


def write_json(file_path, data):
    """原子写入：先写同目录临时文件，再 os.replace 覆盖。

    直接覆盖原文件的话，一旦写入过程中进程被杀 / 磁盘满，会留下半个 JSON，
    下次读取直接失败、数据全丢。os.replace 在同一文件系统上是原子操作，
    要么是旧内容、要么是新内容，不会出现中间态。
    """
    with _write_lock:
        directory = os.path.dirname(file_path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, file_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            raise


def _load_reset_tokens():
    try:
        return read_json(RESET_TOKENS_FILE)
    except Exception:
        return {}


def _save_reset_tokens(tokens):
    write_json(RESET_TOKENS_FILE, tokens)


def _cleanup_reset_tokens():
    """清理过期令牌"""
    tokens = _load_reset_tokens()
    now = datetime.now()
    cleaned = {k: v for k, v in tokens.items() if datetime.fromisoformat(v.get('expires')) > now}
    if len(cleaned) != len(tokens):
        _save_reset_tokens(cleaned)
    return cleaned


def _generate_reset_token(user_id, email):
    """生成带时效的重置令牌并持久化"""
    token = secrets.token_urlsafe(32)
    expires = (datetime.now() + timedelta(hours=1)).isoformat()
    tokens = _cleanup_reset_tokens()
    tokens[token] = {'user_id': user_id, 'email': email, 'expires': expires}
    _save_reset_tokens(tokens)
    return token


def _send_reset_email(email, token):
    """发送密码重置邮件，返回 (success, error_message)"""
    if not all([SMTP_SERVER, SMTP_USER, SMTP_PASSWORD]):
        return False, '邮件服务未配置'

    reset_url = f"{APP_BASE_URL}/?reset_token={token}"
    subject = 'ChemSafe 密码重置'
    body = (
        f"<p>您好，</p>"
        f"<p>您正在申请重置 ChemSafe 账号密码。请点击下方链接完成重置（1小时内有效）：</p>"
        f"<p><a href='{reset_url}'>{reset_url}</a></p>"
        f"<p>如非本人操作，请忽略此邮件。</p>"
        f"<p>ChemSafe · 家庭化学品安全智能助手</p>"
    )

    msg = MIMEMultipart()
    msg['From'] = SMTP_FROM or SMTP_USER
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html', 'utf-8'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM or SMTP_USER, email, msg.as_string())
        return True, None
    except Exception as e:
        return False, str(e)


def get_chemicals_data():
    return read_json(CHEMICALS_FILE)


def _pick_pubchem_display_name(pub, preferred=None):
    """从 PubChem 名称和同义词中挑选适合展示的名称（避开 CAS 号、SMILES、InChI）"""
    import re
    candidates = [pub.get('name', '')] + pub.get('synonyms', [])
    preferred_l = (preferred or '').lower().strip()

    def is_cas(s):
        return bool(re.match(r'^\d{1,7}-\d{2}-\d$', s))

    def is_smiles_or_inchi(s):
        return s.startswith('InChI=') or s.startswith('SMILES=') or ('=' in s and '@' in s)

    # 如果有查询词，优先返回与其完全匹配的同义词（保持 PubChem 原始大小写）
    if preferred_l:
        for c in candidates:
            if c.strip().lower() == preferred_l:
                return c.strip()

    best = pub.get('name', '')
    best_score = -999

    for c in candidates:
        c = c.strip()
        if not c or len(c) < 3 or len(c) > 50:
            continue
        if is_cas(c) or is_smiles_or_inchi(c):
            continue
        if re.match(r'^\d+$', c):
            continue

        score = 0
        # 偏好长度 8-12 的通用名，避免品牌名过短取胜
        score += max(0, 40 - abs(len(c) - 10))
        # 偏好首字母大写的常见名（如 Aspirin、Ibuprofen）
        if c[0].isupper() and not c.isupper():
            score += 15
        # 全小写常见名也可以
        elif c.islower():
            score += 8
        # 全大写化学名稍差
        if c.isupper():
            score -= 10
        # 含空格通常是正式名或通用名
        if ' ' in c:
            score += 3

        if score > best_score:
            best_score = score
            best = c

    return best


def _chemical_candidates(c):
    """某个化学品的全部可匹配字符串：名称 + 所有别名"""
    out = [c['name']]
    alias = c.get('alias') or ''
    out.extend(a for a in alias.split() if a)
    return out


def _candidate_collisions(candidate, chemicals, owner_name):
    """候选词的"碰撞数"：有多少个**其他**化学品的名称包含它。

    数字越小越专属。例："84" 只与「84消毒片」相撞（=1），
    而 "消毒液" 与「含氯消毒液」「消毒液喷雾」都相撞（=2）。
    用于在多个候选同时命中时择优——避免泛词抢答。
    """
    cand = candidate.lower()
    return sum(1 for x in chemicals
               if x['name'] != owner_name and cand in x['name'].lower())


def resolve_chemical(input_name, use_pubchem=True):
    """根据名称或别名解析出标准化学品名称；本地未命中时可查询 PubChem。

    核心原则：**宁可返回 None，也不要猜错**。
    安全提示是按化学品给的，猜错等于给出错误的安全建议，比"未识别"危险得多。
    因此对"类别词""泛词""被多个化学品共用的别名"一律返回 None，让前端提示用户说清楚。
    """
    if not input_name:
        return None
    q = input_name.lower().strip()
    if not q:
        return None
    chemicals = get_chemicals_data().get('chemicals', [])
    if not chemicals:
        return None

    # 1. 名称精确匹配
    for c in chemicals:
        if c['name'].lower() == q:
            return c['name']

    # 2. 类别词不解析成具体产品（"清洁剂""消毒液"是类别，不是某个产品）
    categories = {c['category'].lower() for c in chemicals if c.get('category')}
    if q in categories:
        return None

    # 3. 别名精确匹配：必须唯一
    alias_hits = {c['name'] for c in chemicals
                  if any(a.lower().strip() == q for a in _chemical_candidates(c)[1:])}
    if len(alias_hits) == 1:
        return next(iter(alias_hits))
    if len(alias_hits) > 1:
        return None

    # 4. 模糊匹配（包含关系）。太短的输入不猜：一个"水"字会同时命中"漂白水/双氧水/氨水"
    if len(q) < 2:
        return None

    # 若输入的词出现在 3 个以上化学品的名称里，说明它太泛（如"清洁剂"），不做名称包含匹配
    name_contains = [c['name'] for c in chemicals if q in c['name'].lower()]
    allow_name_contains = len(name_contains) < 3

    scored = []
    for c in chemicals:
        for cand in _chemical_candidates(c):
            cl = cand.lower().strip()
            if not cl or cl == q:
                continue
            if not (cl in q or q in cl):
                continue
            # q in name 方向若被判为"过泛"，跳过该候选
            if not allow_name_contains and cl == c['name'].lower() and q in cl:
                continue
            scored.append((_candidate_collisions(cl, chemicals, c['name']), -len(cl), c['name']))

    if scored:
        scored.sort()
        best_key = scored[0][:2]
        winners = {name for coll, neglen, name in scored if (coll, neglen) == best_key}
        # 最优候选并列时说明存在歧义 —— 不猜
        return next(iter(winners)) if len(winners) == 1 else None

    # 5. PubChem 兜底（英文/化学名）
    if use_pubchem:
        pub_query = PUBCHEM_NAME_MAP.get(input_name, input_name)
        pub = pubchem_lookup(pub_query)
        if pub:
            return _pick_pubchem_display_name(pub, preferred=pub_query)
    return None


def _equivalence_members(name):
    """返回与 name 同属一个等价组的全部名称（含自身）。

    等价组解决"同一个东西有多种叫法"的问题：数据里 84消毒液 / 84消毒片 / 漂白水
    是三个独立条目，如果禁忌规则只挂在其中一个上，用户换个叫法就查不到同一条禁忌——
    对安全类应用这是危险的（同一件危险的事，给出的答案却变弱了）。
    组内成员禁忌一致性的依据写在 data/chemicals.json 的 equivalence_groups 里。
    """
    data = get_chemicals_data()
    valid = {c['name'] for c in data.get('chemicals', [])}
    for g in data.get('equivalence_groups', []):
        members = [m for m in (g.get('members') or []) if m in valid]
        if name in members:
            return members
    return [name]


def _equivalence_group_id(name):
    """返回该化学品所属等价组的 id；不属于任何组时返回 None。"""
    for g in get_chemicals_data().get('equivalence_groups', []):
        if name in (g.get('members') or []):
            return g.get('id')
    return None


def find_mix_rule(a, b, with_origin=False):
    """查找本地混用规则。

    先查精确组合；未命中时在同一等价组内再查一次，
    避免用户把"含氯消毒液"说成"漂白水"就查不到同一条禁忌。
    with_origin=True 时返回 (规则, 实际命中的组合)，供前端说明判定依据。
    """
    index = {}
    for r in get_chemicals_data().get('mixing_rules', []):
        index.setdefault(tuple(sorted([r['a'], r['b']])), r)

    if not a or not b:
        return (None, None) if with_origin else None

    hit = index.get(tuple(sorted([a, b])))
    if hit:
        return (hit, (a, b)) if with_origin else hit

    for x in _equivalence_members(a):
        for y in _equivalence_members(b):
            if x == y:
                continue
            hit = index.get(tuple(sorted([x, y])))
            if hit:
                return (hit, (x, y)) if with_origin else hit

    return (None, None) if with_origin else None


def deepseek_chat(messages, temperature=0.7, max_tokens=800):
    """调用 DeepSeek API"""
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


def vision_recognize(image_base64, prompt='请识别这张化学品标签图片，返回产品名称、主要成分或类别，尽量简洁。', max_tokens=300):
    """调用 OpenAI 兼容的多模态视觉接口识别图片中的化学品信息。

    未配置视觉模型时返回 (None, '未配置视觉模型')，调用方应直接降级到 OCR，
    避免每次都发一次注定失败的请求。
    """
    if not VISION_ENABLED:
        return None, '未配置视觉模型（需设置 VISION_API_URL / VISION_API_KEY / VISION_MODEL）'
    try:
        headers = {
            'Authorization': f'Bearer {VISION_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': VISION_MODEL,
            'messages': [
                {'role': 'system', 'content': '你是家庭化学品安全识别助手，请根据图片内容识别化学品名称或主要成分，只输出最可能的名称，不要多余解释。'},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
                    ]
                }
            ],
            'max_tokens': max_tokens
        }
        resp = requests.post(VISION_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content'], None
    except Exception as e:
        return None, str(e)


def _load_advice_cache():
    try:
        return read_json(ADVICE_CACHE_FILE)
    except Exception:
        return {}


def _save_advice_cache(cache):
    write_json(ADVICE_CACHE_FILE, cache)


def _build_advice_cache_key(context, *identifiers):
    """根据场景和化学品标识生成缓存键"""
    parts = [context] + sorted(str(i).lower().strip() for i in identifiers if i)
    return '|'.join(parts)


def generate_pubchem_advice(pub_a, pub_b=None, context='single'):
    """
    基于 PubChem 数据调用 DeepSeek 生成中文安全建议。
    context: 'single'(单化学品) / 'mix'(混用分析) / 'chat'(聊天 fallback)
    失败时返回 None。
    """
    if not DEEPSEEK_API_KEY:
        return None

    cache = _load_advice_cache()
    cache_key = _build_advice_cache_key(
        context,
        pub_a.get('cid') or pub_a.get('name'),
        (pub_b or {}).get('cid') or (pub_b or {}).get('name')
    )
    if cache_key in cache:
        return cache[cache_key]

    def fmt_hazards(pub):
        hazards = pub.get('hazards', [])
        return ', '.join(hazards) if hazards else '暂无明确 GHS 危害声明'

    if context == 'mix' and pub_b:
        prompt = (
            "你是家庭化学品安全专家。请根据以下两种化学品的 PubChem GHS 数据，"
            "判断它们在日常生活中混用是否安全，并用中文给出结论、原因和防范建议。"
            "要求：1. 先给出明确结论（禁止混用/不建议混用/相对安全）；"
            "2. 用通俗易懂的中文解释风险；3. 给出 1-2 条实用建议；"
            "4. 不确定的信息不要编造。\n\n"
            f"化学品A：{pub_a.get('name', '未知')}，CID {pub_a.get('cid', 'N/A')}，"
            f"分子式 {pub_a.get('formula', 'N/A')}\n"
            f"GHS 危害声明：{fmt_hazards(pub_a)}\n\n"
            f"化学品B：{pub_b.get('name', '未知')}，CID {pub_b.get('cid', 'N/A')}，"
            f"分子式 {pub_b.get('formula', 'N/A')}\n"
            f"GHS 危害声明：{fmt_hazards(pub_b)}"
        )
    else:
        prompt = (
            "你是家庭化学品安全专家。请根据以下 PubChem 数据库信息，"
            "用中文简要说明该化学品在家庭日常场景中的安全使用、存放注意事项以及常见混用禁忌。"
            "要求：1. 语言简洁、通俗；2. 突出家庭场景中的实际风险；"
            "3. 给出 2-3 条具体建议；4. 不确定的信息不要编造。\n\n"
            f"化学品名称（PubChem）：{pub_a.get('name', '未知')}\n"
            f"CID：{pub_a.get('cid', 'N/A')}\n"
            f"分子式：{pub_a.get('formula', 'N/A')}\n"
            f"GHS 危害声明：{fmt_hazards(pub_a)}"
        )

    messages = [
        {'role': 'system', 'content': '你是一位专业的家庭化学品安全顾问，回答简洁、准确、有安全意识。'},
        {'role': 'user', 'content': prompt}
    ]
    reply, err = deepseek_chat(messages, temperature=0.5, max_tokens=600)
    if reply:
        cache[cache_key] = reply
        _save_advice_cache(cache)
        return reply
    print(f'DeepSeek 生成 PubChem 中文建议失败: {err}')
    return None


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


def _find_mentioned_chemicals(text, chemicals):
    """从一句话里找出"提到了哪些化学品"，按出现顺序返回。

    为什么不能简单按"别名出现在文本里"收集：同一物质在数据里有多个条目
    （例如 漂白水 既是独立条目、又是 含氯消毒液 的别名）。直接收集会把同一种东西
    算成两种，取前两个可能拿到同一物质的两个条目，导致回答错对象。
    这里做三件事：名称命中优先于别名命中 → 匹配更长的优先 → 按等价组去重。
    """
    flat = re.sub(r'\s+', '', (text or '').lower())
    hits = []
    for c in chemicals:
        best = None   # (候选串, 是否名称命中)
        for cand in _chemical_candidates(c):
            cand_f = re.sub(r'\s+', '', cand.lower())
            if not cand_f or cand_f not in flat:
                continue
            is_name = (cand_f == c['name'].lower())
            if (best is None
                    or (is_name and not best[1])
                    or (is_name == best[1] and len(cand_f) > len(best[0]))):
                best = (cand_f, is_name)
        if best:
            hits.append((best[0], best[1], c))

    hits.sort(key=lambda h: (not h[1], -len(h[0])))   # 名称命中优先，其次匹配更长

    kept, seen_groups = [], set()
    for _cand, _is_name, c in hits:
        gid = _equivalence_group_id(c['name'])
        if gid:
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
        kept.append(c)
    return kept


def build_local_chat_reply(user_input, history=None):
    """基于本地知识库和 PubChem 数据回答聊天问题（不调用 DeepSeek）"""
    text = user_input.lower()
    chemicals = get_chemicals_data().get('chemicals', [])

    # 1. 识别输入中提到的化学品
    # 注意：不能简单按"别名出现在文本里"收集——同一物质有多个条目（漂白水既是独立条目
    # 又是含氯消毒液的别名），那样会把一种东西算成两种，回答错对象。这里统一走去重逻辑。
    found = _find_mentioned_chemicals(user_input, chemicals)

    # 1b. 补上上一轮提到的化学品。
    #     用户常说"那它能和洁厕灵一起用吗"——当前句只出现了第二个对象（洁厕灵），
    #     第一个对象（"它"）在上文里。所以是"合并"而不是"当前句为空时才回看"。
    if history and len(found) < 2:
        for m in reversed(history):
            if m.get('role') != 'user':
                continue
            prev = _find_mentioned_chemicals(m.get('content') or '', chemicals)
            if not prev:
                continue
            existing = {c['name'] for c in found}
            for c in prev:
                if c['name'] not in existing:
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
        advice = generate_pubchem_advice(pub, context='chat')
        if advice:
            return (
                f"我从 PubChem 数据库查到「{pub['name']}」（CID {pub['cid']}，分子式 {pub['formula']}）。\n\n"
                f"{advice}",
                'pubchem+deepseek'
            )
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

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip().lower()

    if not username or not password:
        return jsonify({'success': False, 'message': '请填写昵称和密码'}), 400
    if len(username) < 2:
        return jsonify({'success': False, 'message': '昵称至少2个字符'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'message': '密码至少6位'}), 400
    if email and '@' not in email:
        return jsonify({'success': False, 'message': '邮箱格式不正确'}), 400

    users = read_json(USERS_FILE)
    if any(u['username'] == username for u in users):
        return jsonify({'success': False, 'message': '该昵称已被注册'}), 400
    if email and any(u.get('email') == email for u in users):
        return jsonify({'success': False, 'message': '该邮箱已被绑定'}), 400

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    new_user = {
        'id': int(time.time() * 1000),
        'username': username,
        'password': hashed.decode('utf-8'),
        'email': email,
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
        'user': {'id': new_user['id'], 'username': new_user['username'], 'email': email}
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
    return jsonify({'success': True, 'user': {'id': user['id'], 'username': user['username'], 'email': user.get('email', '')}})


@app.route('/api/chemicals', methods=['GET'])
def get_chemicals():
    data = get_chemicals_data()
    return jsonify({'success': True, 'data': data.get('chemicals', [])})


@app.route('/api/mixing_rules', methods=['GET'])
def get_mixing_rules():
    """混用禁忌规则表（前端"已知混用禁忌"列表依赖此接口）"""
    data = get_chemicals_data()
    return jsonify({'success': True, 'data': data.get('mixing_rules', [])})


@app.route('/api/chemicals/resolve', methods=['POST'])
@jwt_required()
def resolve_chemical_api():
    """解析用户输入的化学品名称；本地未命中时调用 PubChem 并用 DeepSeek 生成中文建议"""
    data = request.json
    name = data.get('name', '')
    matched = resolve_chemical(name)
    chemicals = get_chemicals_data().get('chemicals', [])
    chem_info = next((c for c in chemicals if c['name'] == matched), None)

    # 本地未命中但 PubChem 命中：补齐中文建议
    pubchem_info = None
    advice = None
    if matched and not chem_info:
        pub_query = PUBCHEM_NAME_MAP.get(name, name)
        pubchem_info = pubchem_lookup(pub_query)
        if pubchem_info:
            advice = generate_pubchem_advice(pubchem_info, context='single')

    return jsonify({
        'success': True,
        'input': name,
        'matched': matched,
        'data': chem_info,
        'pubchem_info': pubchem_info,
        'advice': advice
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
    if hist_type not in VALID_HIST_TYPES:
        return jsonify({'success': False, 'message': '无效的记录类型'}), 400
    histories = read_json(HISTORIES_FILE)
    user_hist = histories.get(str(user_id), {'ai': [], 'rumor': [], 'safety': [], 'mix': []})
    return jsonify({'success': True, 'data': user_hist.get(hist_type, [])})


@app.route('/api/history', methods=['POST'])
@jwt_required()
def save_history():
    data = request.get_json(silent=True) or {}
    user_id = data.get('userId')
    hist_type = data.get('type')
    content = data.get('content')

    current_user = get_jwt_identity()
    if str(user_id) != current_user:
        return jsonify({'success': False, 'message': '无权操作'}), 403

    if hist_type not in VALID_HIST_TYPES:
        return jsonify({'success': False, 'message': '无效的记录类型'}), 400
    if not isinstance(content, str) or not content.strip():
        return jsonify({'success': False, 'message': '缺少记录内容'}), 400
    content = content.strip()[:200]  # 限长，避免客户端塞入超大字符串

    histories = read_json(HISTORIES_FILE)
    user_id_str = str(user_id)
    if user_id_str not in histories:
        histories[user_id_str] = {'ai': [], 'rumor': [], 'safety': [], 'mix': []}
    if hist_type not in histories[user_id_str] or not isinstance(histories[user_id_str][hist_type], list):
        histories[user_id_str][hist_type] = []

    histories[user_id_str][hist_type].insert(0, {
        'content': content,
        'time': str(datetime.now())
    })
    if len(histories[user_id_str][hist_type]) > 50:
        histories[user_id_str][hist_type] = histories[user_id_str][hist_type][:50]

    write_json(HISTORIES_FILE, histories)
    return jsonify({'success': True, 'message': '保存成功'})


@app.route('/api/history/<int:user_id>/<hist_type>/<int:index>', methods=['DELETE'])
@jwt_required()
def delete_history_item(user_id, hist_type, index):
    """删除单条历史记录"""
    current_user = get_jwt_identity()
    if str(user_id) != current_user:
        return jsonify({'success': False, 'message': '无权操作'}), 403
    if hist_type not in VALID_HIST_TYPES:
        return jsonify({'success': False, 'message': '无效的记录类型'}), 400
    histories = read_json(HISTORIES_FILE)
    user_id_str = str(user_id)
    if user_id_str in histories and hist_type in histories[user_id_str]:
        items = histories[user_id_str][hist_type]
        if 0 <= index < len(items):
            items.pop(index)
            write_json(HISTORIES_FILE, histories)
            return jsonify({'success': True, 'message': '已删除'})
    return jsonify({'success': False, 'message': '记录不存在'}), 404


@app.route('/api/history/<int:user_id>/<hist_type>', methods=['DELETE'])
@jwt_required()
def clear_history(user_id, hist_type):
    """清空某类历史记录（保留供管理后台使用）"""
    current_user = get_jwt_identity()
    if str(user_id) != current_user:
        return jsonify({'success': False, 'message': '无权操作'}), 403
    if hist_type not in VALID_HIST_TYPES:
        return jsonify({'success': False, 'message': '无效的记录类型'}), 400
    histories = read_json(HISTORIES_FILE)
    user_id_str = str(user_id)
    if user_id_str in histories and hist_type in histories[user_id_str]:
        histories[user_id_str][hist_type] = []
        write_json(HISTORIES_FILE, histories)
    return jsonify({'success': True, 'message': '已清空'})


@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    """请求邮箱密码重置：发送带令牌的重置链接"""
    data = request.json
    email = data.get('email', '').strip().lower()
    if not email or '@' not in email:
        return jsonify({'success': False, 'message': '请输入有效邮箱'}), 400

    users = read_json(USERS_FILE)
    user = next((u for u in users if u.get('email', '').lower() == email), None)
    # 无论邮箱是否存在，都返回相同提示，避免枚举
    if not user:
        return jsonify({'success': True, 'message': '如果该邮箱已绑定账号，重置链接将发送至邮箱'})

    token = _generate_reset_token(user['id'], email)
    success, err = _send_reset_email(email, token)
    if not success:
        print(f'发送重置邮件失败: {err}')
        return jsonify({'success': False, 'message': '邮件发送失败，请检查邮件服务配置'}), 500

    return jsonify({'success': True, 'message': '重置链接已发送至邮箱，请查收（1小时内有效）'})


@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    """通过邮箱令牌重置密码"""
    data = request.json
    token = data.get('token', '').strip()
    new_password = data.get('newPassword', '').strip()
    if not token or not new_password:
        return jsonify({'success': False, 'message': '缺少令牌或新密码'}), 400
    if len(new_password) < 6:
        return jsonify({'success': False, 'message': '新密码至少6位'}), 400

    tokens = _cleanup_reset_tokens()
    record = tokens.get(token)
    if not record:
        return jsonify({'success': False, 'message': '重置链接已过期或无效'}), 400

    users = read_json(USERS_FILE)
    user = next((u for u in users if str(u['id']) == str(record['user_id'])), None)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在'}), 404

    hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    user['password'] = hashed.decode('utf-8')
    write_json(USERS_FILE, users)

    # 重置成功后使该令牌失效
    tokens.pop(token, None)
    _save_reset_tokens(tokens)

    return jsonify({'success': True, 'message': '密码已重置，请用新密码登录'})


@app.route('/api/account/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_account(user_id):
    """注销账号：删除用户及其历史记录"""
    current_user = get_jwt_identity()
    if str(user_id) != current_user:
        return jsonify({'success': False, 'message': '无权操作'}), 403

    users = read_json(USERS_FILE)
    users = [u for u in users if str(u['id']) != str(user_id)]
    write_json(USERS_FILE, users)

    histories = read_json(HISTORIES_FILE)
    histories.pop(str(user_id), None)
    write_json(HISTORIES_FILE, histories)

    return jsonify({'success': True, 'message': '账号已注销'})


def _match_chemical_by_text(text, chemicals):
    """根据文本匹配化学品名称或别名，返回最佳匹配"""
    result = _match_chemical_by_text_with_info(text, chemicals)
    return result['chemical']


def _match_chemical_by_text_with_info(text, chemicals):
    """从一段文本（OCR 输出 / 视觉模型输出 / 文件名）中匹配化学品。

    择优顺序：候选正好是该化学品名称 > 碰撞数少（越专属）> 匹配更长。
    例如 OCR 把「84消毒液」读成 "84 ... 消毒液" 时，"84"（只与「84消毒片」相撞）
    会胜过泛词 "消毒液"（与「含氯消毒液」「消毒液喷雾」都相撞），从而给出正确结果。
    """
    # OCR 常把产品名按行切断，先去掉全部空白再匹配："84\n消毒液" -> "84消毒液"
    text_flat = re.sub(r'\s+', '', (text or '').lower())
    best = None
    best_len = 0
    best_candidate = ''
    best_collide = None
    best_is_name = False

    for c in chemicals:
        for candidate in _chemical_candidates(c):
            cand = re.sub(r'\s+', '', candidate.lower())
            if not cand or cand not in text_flat:
                continue
            is_name = (cand == c['name'].lower())
            collide = _candidate_collisions(cand, chemicals, c['name'])
            better = False
            if best is None:
                better = True
            elif is_name != best_is_name:
                better = is_name
            elif collide != best_collide:
                better = collide < best_collide
            elif len(cand) > best_len:
                better = True
            if better:
                best = c
                best_len = len(cand)
                best_candidate = candidate
                best_collide = collide
                best_is_name = is_name

    # 证据强度（0~1）：取决于命中的候选词有多具体，而不是它出现在多长的文本里
    # （视觉模型常在名称后面补一句说明，用"长度占比"会无端压低可信度）
    if best is None:
        quality = 0.0
    else:
        if best_is_name:
            quality = 0.90
        elif best_len >= 4:
            quality = 0.80
        elif best_len == 3:
            quality = 0.65
        else:
            quality = 0.40          # 两字符的短别名，证据较弱
        if best_collide == 0:
            quality = min(1.0, quality + 0.10)    # 专属词，加分
        elif best_collide >= 2:
            quality = max(0.15, quality - 0.30)   # 泛词，明显减分
    return {
        'chemical': best,
        'matched_candidate': best_candidate,
        'match_len': best_len,
        'quality': quality
    }


def _calc_upload_confidence(source, quality):
    """根据识别来源和匹配证据强度计算置信度（同图同值，不含随机）。

    用**乘性**模型而不是简单相加：来源基础分再高，只要匹配证据弱，
    最终置信度就必须明显下降——否则 OCR 只匹配到两个字符也会显示 80%+。
    """
    base = {'vision': 0.95, 'ocr': 0.85, 'filename': 0.6}.get(source, 0.5)
    q = min(max(quality, 0.0), 1.0)
    return round(base * (0.4 + 0.6 * q) * 100, 1)


_rapidocr_engine = None


def _get_rapidocr_engine():
    """懒加载 RapidOCR 引擎（首次会加载模型，耗时数秒，只做一次）"""
    global _rapidocr_engine
    if _rapidocr_engine is None:
        _rapidocr_engine = RapidOCR()
    return _rapidocr_engine


def _warm_up_ocr_async():
    """启动后在后台线程预热 OCR 引擎。

    为什么需要：模型加载只在第一次识别时才发生。本地只要 0.6 秒，但在免费实例这种
    受限 CPU 上会拖到几十秒，叠加识别本身耗时后，**第一次上传会慢到接近超时阈值**
    （实测线上首次上传 83 秒，worker 超时是 120 秒）。把这段成本挪到启动阶段后，
    第一个用户的等待时间就只剩下识别本身。
    """
    if not RAPIDOCR_AVAILABLE or os.getenv('OCR_WARMUP', '1') != '1':
        return

    def _run():
        try:
            _get_rapidocr_engine()
            print('[warmup] OCR 引擎预热完成')
        except Exception as e:
            print(f'[warmup] OCR 引擎预热失败（不影响服务启动）: {e}')

    threading.Thread(target=_run, daemon=True).start()


_warm_up_ocr_async()


# OCR 前的图片最长边上限。手机照片常有 3000~4000px，直接送入 OCR 会让内存和耗时
# 随像素数急剧上升（在 512MB 的免费实例上可能直接把进程打挂，表现为 502）。
# 1024 实测峰值约 290MB、仍能识别产品标签；识别小字不清时可上调到 1280/1600。
OCR_MAX_SIDE = int(os.getenv('OCR_MAX_SIDE', '1024'))


def _prepare_image_for_ocr(image_path):
    """把待识别图片缩到 OCR_MAX_SIDE 以内，返回 (可用路径, 是否临时文件)。

    不缩放的风险：一张 2000x2000 的照片解码后就是 12MB，OCR 的检测阶段还会再放大，
    叠加 onnxruntime 的中间张量后内存会出现尖峰。缩到 1280 后像素数只有约 1/2.4。
    """
    try:
        img = Image.open(image_path)
        if max(img.size) <= OCR_MAX_SIDE:
            return image_path, False
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        img.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE), Image.Resampling.LANCZOS)
        fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        img.save(tmp_path, format='JPEG', quality=88)
        return tmp_path, True
    except Exception as e:
        print(f'OCR 图片预处理失败，改用原图: {e}')
        return image_path, False


def ocr_extract_text(image_path):
    """用可用的 OCR 引擎提取图片文字，返回 (text, engine_name)。

    优先 RapidOCR（免系统依赖、中文识别效果好），失败时回退 pytesseract。
    都不可用或都识别为空时返回 ('', None)。
    """
    prepared, is_tmp = _prepare_image_for_ocr(image_path)
    try:
        if RAPIDOCR_AVAILABLE:
            try:
                engine = _get_rapidocr_engine()
                result, _elapse = engine(prepared)
                if result:
                    lines = [item[1] for item in result if len(item) > 1 and item[1]]
                    text = '\n'.join(lines)
                    if text.strip():
                        return text, 'rapidocr'
            except Exception as e:
                print(f'RapidOCR 识别失败，尝试回退: {e}')

        if PYTESSERACT_AVAILABLE:
            try:
                image = Image.open(prepared)
                image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                text = pytesseract.image_to_string(image, lang='chi_sim+eng')
                if text.strip():
                    return text, 'pytesseract'
            except Exception as e:
                print(f'Tesseract 识别失败: {e}')

        return '', None
    finally:
        if is_tmp and os.path.exists(prepared):
            try:
                os.unlink(prepared)
            except Exception:
                pass


def _resize_to_base64(file_storage, max_size=(1024, 1024), quality=85):
    """将上传图片缩放、转 JPEG 后返回 base64 字符串"""
    try:
        image = Image.open(file_storage.stream)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f'图片预处理失败: {e}')
        return None


def _parse_vision_names(text):
    """把视觉模型返回的名称解析成列表。

    模型被要求返回 JSON 数组（最多 3 个）；解析失败时退回按顿号/逗号/换行拆分。
    支持多产品同框：一张照片里出现多个瓶子时不再只取一个。
    """
    if not text:
        return []
    t = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.S).strip()
    try:
        arr = json.loads(t)
        if isinstance(arr, list):
            names = [str(x).strip() for x in arr if str(x).strip()]
        elif isinstance(arr, str) and arr.strip():
            names = [arr.strip()]
        else:
            names = []
    except Exception:
        names = [p.strip().strip('。.，,"\'') for p in re.split(r'[、,，;；\n]+', t) if p.strip()]
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out[:3]


@app.route('/api/upload', methods=['POST'])
@jwt_required()
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': '未上传文件'}), 400

    file = request.files['image']
    filename = file.filename.lower()
    chemicals = get_chemicals_data().get('chemicals', [])
    matched = None
    source = 'none'
    quality = 0
    ocr_engine = None
    vision_error = None
    products = []   # 多产品同框时，全部命中结果（含第一个）

    # 1. 优先调用视觉模型识别标签内容（需配置 VISION_* 变量；未配置则直接走 OCR）
    if VISION_ENABLED:
        try:
            base64_image = _resize_to_base64(file)
            if base64_image:
                vision_text, err = vision_recognize(
                    base64_image,
                    prompt='这是一张家庭化学品的照片，图中可能有一个或多个产品。'
                           '请识别图中所有化学品/清洁用品的名称，以 JSON 数组返回，最多 3 个，'
                           '例如 ["84消毒液"] 或 ["84消毒液","洁厕灵"]。只输出 JSON 数组，不要解释。',
                    max_tokens=300
                )
                if err:
                    vision_error = err
                    print(f'Vision API 调用失败: {err}')
                elif vision_text:
                    seen_groups = set()
                    for nm in _parse_vision_names(vision_text):
                        info = _match_chemical_by_text_with_info(nm, chemicals)
                        if not info['chemical']:
                            continue
                        name = info['chemical']['name']
                        gid = _equivalence_group_id(name)
                        if gid:
                            if gid in seen_groups:
                                continue
                            seen_groups.add(gid)
                        elif name in [p['name'] for p in products]:
                            continue
                        products.append({
                            **info['chemical'],
                            'confidence': _calc_upload_confidence('vision', info['quality']),
                            'source': 'vision'
                        })
                    if products:
                        matched = next(x for x in chemicals if x['name'] == products[0]['name'])
                        source = 'vision'
                        # 第一个产品的质量用于整体置信度
                        q0 = _match_chemical_by_text_with_info(
                            _parse_vision_names(vision_text)[0] if _parse_vision_names(vision_text) else '',
                            chemicals)
                        quality = q0['quality']
                    else:
                        # 视觉模型返回了内容，但没匹配到任何已知化学品
                        vision_error = '视觉模型已返回内容，但未匹配到已知化学品；原始返回: ' + str(vision_text)[:120]
        except Exception as e:
            print(f'Vision 识别异常: {e}')

    # 2. Vision 未命中时回退到 OCR（RapidOCR 优先，其次 Tesseract）
    if not matched and OCR_AVAILABLE:
        tmp_path = None
        try:
            file.stream.seek(0)
            ext = os.path.splitext(filename)[1] or '.jpg'
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            ocr_text, ocr_engine = ocr_extract_text(tmp_path)
            if ocr_text:
                info = _match_chemical_by_text_with_info(ocr_text, chemicals)
                if info['chemical']:
                    matched = info['chemical']
                    source = 'ocr'
                    quality = info['quality']
        except Exception as e:
            print(f'OCR 识别失败: {e}')
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # 3. OCR 未命中时回退到文件名匹配
    if not matched:
        info = _match_chemical_by_text_with_info(filename, chemicals)
        if info['chemical']:
            matched = info['chemical']
            source = 'filename'
            quality = info['quality']

    # 4. 仍未命中则返回识别失败，不再随机返回
    if not matched:
        if not VISION_ENABLED and not OCR_AVAILABLE:
            hint = ('当前未配置视觉模型（VISION_API_KEY），本机也没有可用的 OCR 引擎，'
                    '只能按文件名匹配。请配置 VISION_API_KEY，或安装 rapidocr_onnxruntime / Tesseract。')
        elif not VISION_ENABLED:
            hint = '未配置视觉模型（VISION_API_KEY），已尝试 OCR 但未匹配到已知化学品，建议直接输入产品名称查询。'
        else:
            hint = '未能从图片中识别出已知化学品，建议直接输入产品名称查询。'
        return jsonify({
            'success': True,
            'data': {
                'recognized': False,
                'message': hint
            }
        })

    confidence = _calc_upload_confidence(source, quality)
    resp = {
        **matched,
        'recognized': True,
        'confidence': confidence,
        'source': source,
        'ocr_engine': ocr_engine if source == 'ocr' else None,
        'vision_error': vision_error
    }
    # 多产品同框：附上全部命中结果（第一个与顶层字段一致）
    if len(products) > 1:
        resp['products'] = products
    return jsonify({'success': True, 'data': resp})


# 谣言本地匹配阈值：>= 前者直接采用本地结论；>= 后者仅作为 AI 判断的提示
RUMOR_LOCAL_THRESHOLD = 0.35
RUMOR_HINT_THRESHOLD = 0.15


def _rumor_tokens(text):
    """把文本切成用于相似度比较的 token：英文/数字词 + 中文二元组（bigram）"""
    import re
    cleaned = re.sub(r'[^\w\u4e00-\u9fa5]+', '', (text or '').lower())
    tokens = set(re.findall(r'[a-z0-9]{2,}', cleaned))
    for i in range(len(cleaned) - 1):
        pair = cleaned[i:i + 2]
        if all('\u4e00' <= ch <= '\u9fa5' for ch in pair):
            tokens.add(pair)
    return tokens


def _rumor_similarity(query, title):
    """Dice 相似度（0~1）：同时考虑查询与标题两侧长度，
    避免像"84消毒液怎么用"这类短查询因被长标题包含而虚高。"""
    q = _rumor_tokens(query)
    t = _rumor_tokens(title)
    if not q or not t:
        return 0.0
    return 2 * len(q & t) / (len(q) + len(t))


def _match_rumor(text, rumors):
    """返回 (最相似的谣言, 相似度)；无候选返回 (None, 0.0)"""
    best, best_score = None, 0.0
    for r in rumors:
        title = r.get('title', '')
        if not title:
            continue
        if title in text or text in title:
            return r, 1.0
        score = _rumor_similarity(text, title)
        if score > best_score:
            best, best_score = r, score
    return best, best_score


@app.route('/api/analyze-rumor', methods=['POST'])
@jwt_required()
def analyze_rumor():
    data = request.json
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'success': False, 'message': '请输入谣言内容'}), 400

    rumors = get_chemicals_data().get('rumors', [])
    matched, score = _match_rumor(text, rumors)

    # 本地辟谣库命中：只有相似度足够高才直接采用，避免张冠李戴
    if matched and score >= RUMOR_LOCAL_THRESHOLD:
        confidence = int(round(min(0.98, 0.72 + score * 0.3) * 100))
        return jsonify({'success': True, 'data': {
            **matched,
            'input': text,
            'found': True,
            'confidence': confidence,
            'source': 'local'
        }})

    # 相似但未达阈值 / 完全未命中：交给 DeepSeek 实时判断（可用时）
    if DEEPSEEK_API_KEY:
        hint = ''
        if matched and score >= RUMOR_HINT_THRESHOLD:
            hint = (f'\n（本地辟谣库中最接近的说法是「{matched["title"]}」，'
                    '仅供参考，请独立判断用户说法是否等同于此。）')
        messages = [
            {'role': 'system', 'content': (
                '你是家庭化学品安全领域的辟谣助手。请判断用户给出的说法是否成立，'
                '用中文先给出明确结论（谣言/属实/部分正确），再简要说明科学依据。'
                '不确定时明确说明不确定，不要编造。控制在 200 字以内。'
            )},
            {'role': 'user', 'content': f'网传说法：{text}{hint}'}
        ]
        reply, err = deepseek_chat(messages, temperature=0.3, max_tokens=400)
        if reply:
            return jsonify({'success': True, 'data': {
                'title': text,
                'result': 'AI 分析结论',
                'explain': reply,
                'confidence': 70,
                'found': True,
                'source': 'deepseek',
                'input': text
            }})
        print(f'辟谣 DeepSeek 调用失败: {err}')

    # 最终未命中：明确告知未收录，绝不返回随机结果
    tip = '本地辟谣库暂未收录该说法。'
    if matched and score >= RUMOR_HINT_THRESHOLD:
        tip += f'最接近的已收录说法是「{matched["title"]}」，但相似度不足，未直接采用。'
    tip += '建议核对产品说明书，或咨询权威机构后再判断，切勿轻信未经验证的说法。'
    return jsonify({'success': True, 'data': {
        'title': text,
        'result': '暂未收录',
        'explain': tip,
        'confidence': 0,
        'found': False,
        'source': 'none',
        'input': text
    }})


@app.route('/api/mix/check', methods=['POST'])
@jwt_required()
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

    # 本地规则命中直接返回（支持等价组：换个产品名也能命中同一条禁忌）
    if a and b:
        rule, origin = find_mix_rule(a, b, with_origin=True)
        if rule:
            result['rule'] = rule
            if origin and tuple(sorted(origin)) != tuple(sorted([a, b])):
                result['rule']['matched_pair'] = list(origin)
                result['rule']['via_equivalence'] = True
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
        # 优先用 DeepSeek 生成自然中文解释，失败则保留 GHS 规则解释
        deepseek_explain = generate_pubchem_advice(info_a, info_b, context='mix')
        result['pubchem_rule'] = {
            'level': analysis['level'],
            'danger': analysis['danger'],
            'result': analysis['result'],
            'explain': deepseek_explain or analysis['explain']
        }
        return jsonify({'success': True, 'data': result})

    # 能识别但无规则、无 PubChem 数据
    result['source'] = 'fallback'
    result['error'] = '未能获取足够的安全数据'
    return jsonify({'success': True, 'data': result})


@app.route('/api/chat', methods=['POST'])
@jwt_required()
def chat():
    """小安 AI 聊天：优先调用 DeepSeek，失败时回退本地知识库。

    支持多轮上下文：前端把最近几轮对话一起送来。没有上下文时，
    "那它能和别的混用吗"这类指代根本无法解析。
    """
    data = request.get_json(silent=True) or {}
    user_input = (data.get('message') or '').strip()
    if not user_input:
        return jsonify({'success': False, 'message': '请输入问题'}), 400

    # 上下文：只取最近 6 条，单条截断，避免超长和滥用
    history = []
    for m in (data.get('history') or [])[-6:]:
        if not isinstance(m, dict):
            continue
        role = m.get('role')
        text = str(m.get('text') or '').strip()
        if role in ('user', 'bot') and text:
            history.append({'role': 'user' if role == 'user' else 'assistant',
                            'content': text[:1000]})

    if DEEPSEEK_API_KEY:
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是 ChemSafe 的家庭化学品安全智能助手"小安"。'
                    '你擅长回答清洁剂、消毒液、护肤品、日用品等家庭化学品的安全问题，'
                    '包括能否混用、如何存放、误食/溅入眼睛应急处理、使用注意事项等。'
                    '回答要简洁、准确、有安全意识，不确定时明确说明。'
                    '不要回答与化学品安全无关的问题。'
                )
            }
        ]
        messages.extend(history)
        messages.append({'role': 'user', 'content': user_input})
        reply, err = deepseek_chat(messages, temperature=0.6, max_tokens=800)
        if reply:
            return jsonify({
                'success': True,
                'data': {'reply': reply, 'source': 'deepseek'}
            })
        print(f'DeepSeek 调用失败，回退本地: {err}')

    reply, source = build_local_chat_reply(user_input, history=history)
    return jsonify({
        'success': True,
        'data': {'reply': reply, 'source': source}
    })


@app.route('/api/chat/product', methods=['POST'])
@jwt_required()
def chat_about_product():
    """针对已识别化学品进行 AI 问答"""
    data = request.json
    product = data.get('product', '').strip()
    question = data.get('question', '').strip()
    if not product or not question:
        return jsonify({'success': False, 'message': '请提供产品名称和问题'}), 400

    chemicals = get_chemicals_data().get('chemicals', [])
    chemical = next(
        (c for c in chemicals
         if c['name'] == product or product in c.get('alias', '').split()),
        None
    )

    context_lines = [f'产品名称：{product}']
    if chemical:
        context_lines.extend([
            f'类别：{chemical.get("category", "未知")}',
            f'安全等级：{chemical.get("safety", "未知")}',
            f'使用建议：{chemical.get("tip", "无")}'
        ])
    else:
        # 尝试 PubChem 补全信息
        resolved = resolve_chemical(product, use_pubchem=True)
        if resolved and resolved.lower() != product.lower():
            context_lines.append(f'英文名/PubChem：{resolved}')

    context = '\n'.join(context_lines)

    if DEEPSEEK_API_KEY:
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是 ChemSafe 的家庭化学品安全智能助手"小安"。'
                    '用户正在询问一个已识别的化学品，请基于下面提供的产品信息回答其功效、'
                    '安全性、使用方法、混用禁忌、注意事项等问题。回答简洁、准确、有安全意识，'
                    '不确定时明确说明，不要编造。'
                )
            },
            {
                'role': 'user',
                'content': f'{context}\n\n用户问题：{question}'
            }
        ]
        reply, err = deepseek_chat(messages, temperature=0.6, max_tokens=800)
        if reply:
            return jsonify({
                'success': True,
                'data': {'reply': reply, 'source': 'deepseek'}
            })
        print(f'DeepSeek 产品问答失败，回退本地: {err}')

    # 本地回退
    fallback = (
        f'关于 {product} 的已知信息：\n{context}\n\n'
        f'你的问题是：{question}\n'
        '当前 AI 服务暂不可用，建议参考产品说明书或咨询专业人士。'
    )
    return jsonify({
        'success': True,
        'data': {'reply': fallback, 'source': 'local'}
    })


@app.route('/api/feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    """记录用户对识别结果的反馈（真实落盘）。

    只做能力范围内的事：如实记录并给出累计条数。不做"自动训练模型"这类做不到的承诺。
    """
    data = request.get_json(silent=True) or {}
    product = (data.get('product') or '').strip()
    source = (data.get('source') or '').strip()
    correct = data.get('correct')

    if not product or not isinstance(correct, bool):
        return jsonify({'success': False, 'message': '缺少反馈内容'}), 400

    records = read_json(FEEDBACK_FILE)
    if not isinstance(records, list):
        records = []
    records.insert(0, {
        'product': product[:64],
        'source': source[:32],
        'correct': correct,
        'user_id': get_jwt_identity(),
        'time': str(datetime.now())
    })
    del records[1000:]  # 只保留最近 1000 条，避免文件无限增长
    write_json(FEEDBACK_FILE, records)

    return jsonify({'success': True, 'message': '反馈已记录', 'total': len(records)})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """首页统计：全部基于真实数据计算，不返回任何随机或写死的数字"""
    data = get_chemicals_data()
    chemicals = data.get('chemicals', [])
    rumors = data.get('rumors', [])
    rules = data.get('mixing_rules', [])
    users = read_json(USERS_FILE)
    histories = read_json(HISTORIES_FILE)
    if not isinstance(histories, dict):
        histories = {}
    feedback = read_json(FEEDBACK_FILE)
    if not isinstance(feedback, list):
        feedback = []

    def count_history(hist_type):
        total = 0
        for user_hist in histories.values():
            if isinstance(user_hist, dict) and isinstance(user_hist.get(hist_type), list):
                total += len(user_hist[hist_type])
        return total

    # 别名总数（去重）：知识库规模的一项指标
    aliases = set()
    for c in chemicals:
        for a in (c.get('alias') or '').split():
            if a.strip():
                aliases.add(a.strip())

    return jsonify({
        'success': True,
        'data': {
            'chemical_count': len(chemicals),
            'alias_count': len(aliases),
            'rumor_count': len(rumors),
            'rule_count': len(rules),
            'user_count': len(users) if isinstance(users, list) else 0,
            'recognition_total': count_history('ai'),
            'rumor_total': count_history('rumor'),
            'mix_total': count_history('mix'),
            'feedback_count': len(feedback)
        }
    })


@app.route('/healthz')
def healthz():
    """健康检查端点：供 Render 等平台探活使用，不读写任何数据文件。"""
    return jsonify({
        'status': 'ok',
        'vision_enabled': VISION_ENABLED,
        'ocr_engine': OCR_ENGINE_NAME
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
        print(f'👁️  视觉识别: ' + (f'已启用（{VISION_MODEL}）' if VISION_ENABLED else '未配置（设置 VISION_API_KEY 后启用）'))
        print(f'🔍 OCR 兜底: ' + (OCR_ENGINE_NAME or '不可用（安装 rapidocr_onnxruntime 后启用）'))
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
