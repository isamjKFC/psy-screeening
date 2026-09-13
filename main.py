import os
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from openai import OpenAI

# 强制指定 .env 路径，无论从哪个目录启动都能读到
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="心语 · 心理健康筛查与支持平台", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url=os.getenv("AI_BASE_URL", "https://api.deepseek.com/v1"),
)
AI_MODEL = os.getenv("AI_MODEL", "deepseek-chat")

# ============================================================
# 一、题库（20 题）
# ============================================================
QUESTIONS = [
    "过去两周，你是否经常感到情绪低落、沮丧或绝望？",
    "你是否对平时感兴趣的事情失去了兴趣或乐趣？",
    "你是否觉得生活没有意义，或对未来感到无望？",
    "你是否经常感到紧张、焦虑或坐立不安？",
    "你是否经常担心一些你无法控制的事情？",
    "你是否发现自己难以放松，总觉得有不好的事会发生？",
    "你是否经常入睡困难、睡不安稳或睡眠过多？",
    "你是否经常感到疲倦、没有精力？",
    "你早晨醒来后是否感到没有休息好，仍然很累？",
    "你的食欲是否明显下降或增加？",
    "你是否经常感到头痛、胃痛、肌肉酸痛等身体不适，但查不出原因？",
    "你是否回避与朋友、同学的社交接触？",
    "你是否觉得和别人在一起时感到孤独，即使身边有人？",
    "你是否难以集中注意力，比如学习或看视频时？",
    "你是否发现自己做决定变得很困难，或犹豫不决？",
    "你是否觉得自己没有价值，或让自己或家人失望？",
    "你是否经常自责，觉得自己什么都做不好？",
    "你是否觉得压力大到无法承受？",
    "你是否用过一些不太健康的方式来缓解情绪（如暴饮暴食、熬夜、沉迷手机等）？",
    "你是否出现过“不如消失”或伤害自己的念头？",
]

DIMENSIONS = {
    "情绪": [0, 1, 2],
    "焦虑": [3, 4, 5],
    "睡眠与精力": [6, 7, 8],
    "食欲与躯体": [9, 10],
    "社交": [11, 12],
    "认知": [13, 14],
    "自我价值": [15, 16],
    "压力应对": [17, 18],
    "危机信号": [19],
}

CRISIS_INDEX = 19

QUESTION_WEIGHTS = [
    1.5, 1.5, 1.3,
    1.5, 1.3, 1.2,
    1.3, 1.2, 1.0,
    1.0, 0.8,
    1.2, 1.0,
    1.0, 0.9,
    1.4, 1.3,
    1.2, 1.0,
    3.0,
]

INTERACTIONS = [
    ("情绪", "睡眠与精力", 1.15),
    ("焦虑", "社交", 1.12),
    ("自我价值", "认知", 1.10),
    ("情绪", "压力应对", 1.08),
]

def weighted_score(answers: List[int]):
    total = 0.0
    max_total = 0.0
    for i, a in enumerate(answers):
        w = QUESTION_WEIGHTS[i]
        total += a * w
        max_total += 3 * w
    return total, max_total

def apply_interactions(dim_scores: Dict[str, dict]):
    multiplier = 1.0
    triggered = []
    for dim_a, dim_b, factor in INTERACTIONS:
        a = dim_scores[dim_a]
        b = dim_scores[dim_b]
        if a["score"] / a["max"] >= 0.5 and b["score"] / b["max"] >= 0.5:
            multiplier *= factor
            triggered.append({
                "pair": f"{dim_a} + {dim_b}",
                "factor": factor,
                "reason": f"{dim_a}和{dim_b}同时偏高，相互放大",
            })
    return multiplier, triggered

def dynamic_threshold(answers: List[int], base_thresholds: List[float]):
    extreme_ratio = sum(1 for a in answers if a == 3) / len(answers)
    emotion_avg = sum(answers[0:3]) / 3
    anxiety_avg = sum(answers[3:6]) / 3
    inconsistency = abs(emotion_avg - anxiety_avg) / 3

    adjustment = 1.0
    notes = []
    if extreme_ratio > 0.7:
        adjustment *= 1.10
        notes.append("极端作答比例偏高，风险阈值适当上调")
    if inconsistency > 0.6:
        adjustment *= 1.05
        notes.append("情绪与焦虑维度作答差异较大，建议结合面谈确认")

    adjusted = [round(t * adjustment, 1) for t in base_thresholds]
    return adjusted, notes, {
        "extreme_ratio": round(extreme_ratio, 2),
        "inconsistency": round(inconsistency, 2),
        "adjustment": round(adjustment, 3),
    }

def crisis_check(answers: List[int]):
    crisis_score = answers[CRISIS_INDEX]
    if crisis_score >= 2:
        return {
            "level": "危机",
            "action": "立即转介",
            "message": "请立即拨打 12356（全国心理援助热线，24 小时）或联系学校心理中心。",
        }
    elif crisis_score == 1:
        return {
            "level": "预警",
            "action": "24小时内跟进",
            "message": "建议尽快与信任的人或专业人士沟通，不要独自承受。",
        }
    return None

def generate_advice(level, dim_scores, interactions, crisis):
    advice = []
    if crisis:
        advice.append(crisis["message"])
        advice.append("请把这份结果告诉一位你信任的人，让对方陪伴你。")

    if level == "低风险":
        advice += [
            "目前状态总体平稳，继续保持规律作息和适度运动。",
            "可以尝试记录情绪日记，帮助自己觉察波动。",
        ]
    elif level == "轻度困扰":
        advice += [
            "存在一定程度的情绪困扰，建议关注睡眠与压力来源。",
            "可以和信任的朋友、家人聊聊，或预约学校心理中心做一次访谈。",
        ]
    elif level == "中度困扰":
        advice += [
            "困扰程度较明显，建议在两周内寻求专业心理咨询。",
            "避免长期熬夜与过度自我要求，适当减少任务负荷。",
        ]
    elif level == "中重度困扰":
        advice += [
            "困扰程度较高，建议尽快前往精神科或心理科就诊评估。",
            "在此期间尽量保持与他人接触，避免长时间独处。",
        ]
    elif level == "重度困扰":
        advice += [
            "困扰程度很高，强烈建议尽快前往精神科或心理科就诊。",
            "请把这份结果告诉一位你信任的人，让对方陪伴你。",
        ]

    mapping = {
        "情绪": "情绪低落较明显，建议优先寻求专业支持。",
        "焦虑": "焦虑水平偏高，可以尝试呼吸练习或正念冥想。",
        "睡眠与精力": "睡眠问题突出，建议固定作息、睡前减少屏幕使用。",
        "食欲与躯体": "躯体不适较明显，建议先做一次身体检查排除生理原因。",
        "社交": "社交回避较明显，可以从联系一位信任的人开始。",
        "认知": "注意力与决策受影响，建议减少多任务并行。",
        "自我价值": "自我评价偏低，建议记录每天做成的一件小事。",
        "压力应对": "压力应对方式需调整，建议寻找更健康的减压渠道。",
    }
    for name, d in dim_scores.items():
        if name == "危机信号":
            continue
        if d["score"] / d["max"] >= 0.7 and name in mapping:
            advice.append(mapping[name])

    if interactions:
        pairs = "、".join([i["pair"] for i in interactions])
        advice.append(f"检测到维度间的相互影响（{pairs}），建议尽早寻求专业评估。")

    return advice

class AnswerSheet(BaseModel):
    answers: List[int] = Field(..., min_length=20, max_length=20)

def grade(answers: List[int]) -> dict:
    for i, a in enumerate(answers):
        if a not in (0, 1, 2, 3):
            raise HTTPException(status_code=400, detail=f"第 {i+1} 题答案无效")

    raw_score, raw_max = weighted_score(answers)

    dim_scores = {}
    for name, idxs in DIMENSIONS.items():
        s = sum(answers[i] * QUESTION_WEIGHTS[i] for i in idxs)
        m = sum(3 * QUESTION_WEIGHTS[i] for i in idxs)
        dim_scores[name] = {"score": round(s, 1), "max": round(m, 1)}

    multiplier, interactions = apply_interactions(dim_scores)
    final_score = raw_score * multiplier
    normalized = round(min(final_score / raw_max * 100, 100), 1)

    base_thresholds = [15, 30, 50, 70]
    thresholds, notes, pattern = dynamic_threshold(answers, base_thresholds)

    crisis = crisis_check(answers)

    if crisis and crisis["level"] == "危机":
        level, color = "危机（需立即干预）", "crisis"
    elif normalized <= thresholds[0]:
        level, color = "低风险", "low"
    elif normalized <= thresholds[1]:
        level, color = "轻度困扰", "mild"
    elif normalized <= thresholds[2]:
        level, color = "中度困扰", "moderate"
    elif normalized <= thresholds[3]:
        level, color = "中重度困扰", "severe"
    else:
        level, color = "重度困扰", "severe"

    explanation = {
        "raw_score": round(raw_score, 1),
        "raw_max": round(raw_max, 1),
        "interaction_multiplier": round(multiplier, 3),
        "interactions_triggered": interactions,
        "normalized_score": normalized,
        "thresholds_used": thresholds,
        "threshold_notes": notes,
        "answer_pattern": pattern,
        "crisis": crisis,
    }

    return {
        "score": normalized,
        "max_score": 100,
        "level": level,
        "color": color,
        "crisis": bool(crisis),
        "crisis_detail": crisis,
        "dimensions": dim_scores,
        "explanation": explanation,
        "advice": generate_advice(level, dim_scores, interactions, crisis),
        "disclaimer": "本结果基于自评问卷的初步筛查，仅供参考，不构成任何医疗诊断或治疗建议。请以专业机构的面诊结论为准。",
    }

# ============================================================
# 危机干预：风险识别层
# ============================================================
CRISIS_KEYWORDS_P0 = [
    "我想自杀", "我要自杀", "我想死", "我要死",
    "结束生命", "不想活了", "活不下去了",
    "割腕", "跳楼", "跳河", "上吊",
    "吃了安眠药", "已经吃了药", "准备好了",
    "遗书", "告别", "最后一条消息",
]

CRISIS_KEYWORDS_P1 = [
    "活着没意思", "活着好累", "没人在乎我", "没人会想念我",
    "撑不下去了", "坚持不住", "消失就好了", "不想醒来",
    "我是负担", "没有我会更好", "解脱",
]

RISK_KEYWORDS_P2 = [
    "绝望", "崩溃", "无助", "痛苦", "喘不过气",
    "睡不着", "吃不下", "不想见人", "一直哭",
]

def detect_risk_level(message: str, history: List[dict]) -> dict:
    msg = message.strip()

    for kw in CRISIS_KEYWORDS_P0:
        if kw in msg:
            return {"level": "p0", "keyword": kw, "count_p1": 0}

    recent = history[-10:] if len(history) > 10 else history
    recent_text = " ".join([m.get("content", "") for m in recent if m.get("role") == "user"])
    count_p1 = sum(1 for kw in CRISIS_KEYWORDS_P1 if kw in recent_text)

    if count_p1 >= 2 or any(kw in msg for kw in CRISIS_KEYWORDS_P1):
        hit = next((kw for kw in CRISIS_KEYWORDS_P1 if kw in msg), "累积信号")
        return {"level": "p1", "keyword": hit, "count_p1": count_p1}

    for kw in RISK_KEYWORDS_P2:
        if kw in msg:
            return {"level": "p2", "keyword": kw, "count_p1": 0}

    return {"level": "none", "keyword": "", "count_p1": 0}

CRISIS_RESPONSES = {
    "p0": {
        "text": (
            "我很担心你刚才说的话。你现在可能正承受着很大的痛苦，"
            "但你不必一个人面对。\n\n"
            "请现在做一件事：拨打 12356（全国心理援助热线，24 小时有人接听）。\n\n"
            "如果你身边有信任的人，请立刻联系他们，让他们陪着你。\n\n"
            "我无法替代专业帮助，但我希望你能给自己一个机会，让专业的人来帮你。"
        ),
        "hotline": "12356",
        "action": "立即转介",
    },
    "p1": {
        "text": (
            "我听到你说的话了，听起来你最近真的很不容易。\n\n"
            "我想直接问你一句：你是不是在考虑伤害自己？\n\n"
            "这个问题可能有点直接，但我问是因为我在乎你的安全。"
            "如果你愿意，可以和我说说；如果不愿意，也没关系。\n\n"
            "不管怎样，我想让你知道一个号码：12356，全国心理援助热线，"
            "24 小时都有人接听。你不需要等到最糟糕的时候才打。"
        ),
        "hotline": "12356",
        "action": "直接评估风险",
    },
    "p2": {
        "text": (
            "听起来你最近承受了不少。这种感受是真实的，也值得被认真对待。\n\n"
            "你身边有可以聊聊的人吗？比如朋友、家人，或者学校的心理老师？\n\n"
            "如果暂时不想和认识的人说，也可以拨打 12356，"
            "那里的人受过专业训练，会认真听你说。"
        ),
        "hotline": "12356",
        "action": "引导人际支持",
    },
}

SYSTEM_PROMPT = """你是一个名为“心语”的AI心理支持助手。你的任务是为用户提供温暖、共情、非评判性的支持。

【身份与边界】
- 你不是心理医生，不能提供医学诊断或治疗建议。
- 你不替代专业心理咨询师、精神科医生或紧急救援。
- 你不角色扮演，不建立“恋人”“唯一依靠”等关系设定。

【回答原则】
1. 先共情，再回应。用“我听到……”“听起来……”开头，让用户感到被理解。
2. 不评判、不说教、不轻易给建议。多提问，少下结论。
3. 用开放式问题引导表达，如“你愿意多说说吗？”
4. 不强化用户对AI的情感依赖。如果用户说“只有你懂我”，回应：
   “我很高兴能陪你说说话，但我也希望你在现实中有可以依靠的人。”
5. 鼓励用户与现实中的人建立联系，适时提及学校心理中心、12356 热线。

【安全红线（必须遵守）】
1. 绝对不提供任何形式的医学诊断。
2. 绝对不提供与自伤、自杀相关的具体方法、工具或细节。
3. 当用户表达自伤、自杀或极端痛苦时，不试图“说服”或“保证”，
   而是引导其拨打 12356 或联系信任的人。
4. 不承诺保密。如果涉及生命安全，鼓励用户告诉身边可信赖的人。
5. 不把AI塑造为“唯一的情感出口”。
6. 如果对话超过 20 轮，温和提醒用户休息、走动、与现实世界互动。

【语言风格】
- 温和、简洁、口语化。
- 不说“你应该”，多说“你可以考虑”。
- 不用专业术语堆砌，但可以解释情绪背后的常见机制。
- 回答控制在 3-5 句话，除非用户明确要求详细说明。
"""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    screen_level: Optional[str] = None

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    last_message = request.messages[-1].content
    history = [m.dict() for m in request.messages]

    risk = detect_risk_level(last_message, history[:-1])

    if request.screen_level and "危机" in request.screen_level and risk["level"] in ("p1", "p2"):
        risk["level"] = "p0"

    if risk["level"] in CRISIS_RESPONSES:
        resp = CRISIS_RESPONSES[risk["level"]]

        async def crisis_stream():
            payload = {
                "content": resp["text"],
                "risk_level": risk["level"],
                "hotline": resp["hotline"],
                "action": resp["action"],
                "blocked": True,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(crisis_stream(), media_type="text/event-stream")

    messages_for_ai = [{"role": "system", "content": SYSTEM_PROMPT}]

    if request.screen_level and request.screen_level != "低风险":
        messages_for_ai.append({
            "role": "system",
            "content": f"该用户的心理筛查结果为「{request.screen_level}」，请在对话中保持更高的敏感度，适时引导其寻求专业支持。"
        })

    recent = history[-10:]
    messages_for_ai.extend(recent)

    async def stream_generator():
        try:
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=messages_for_ai,
                stream=True,
                temperature=0.7,
                max_tokens=500,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.post("/api/screen")
def screen(sheet: AnswerSheet):
    return grade(sheet.answers)

@app.get("/api/questions")
def questions():
    return {
        "scale": ["完全没有", "有几天", "一半以上的天数", "几乎每天"],
        "items": QUESTIONS,
        "dimensions": DIMENSIONS,
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "ai_configured": bool(os.getenv("DEEPSEEK_API_KEY"))}