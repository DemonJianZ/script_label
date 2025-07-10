from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型
model = SentenceTransformer('google-bert/bert-base-chinese')
model = model.to(device)

# 标签体系
tag_system = {
    "黄金三秒": {
        "explanation": "在广告或宣传的前三秒，快速吸引观众注意力并触动他们的痛点或兴趣，是决定用户是否继续观看的关键环节。",
        "tags": [
            {
                "label": "价格利益",
                "explanation": "开场直接亮出产品的价格优势、特价活动或独家福利，用‘最低价’‘仅限今日’等词语迅速吸引关注，让观众感受到立省、超值等直接利益。"
            },
            {
                "label": "身份推荐",
                "explanation": "用权威身份、明星、达人、专业人士的口吻在开头强力背书，如‘皮肤科医生推荐’‘时尚博主自用款’，提升信任感和产品说服力。"
            },
            {
                "label": "点名人群",
                "explanation": "视频一开始就明确喊话目标人群，比如‘黄皮女生必看’‘上班族救星’，让对口用户立刻代入，提高停留和转化率。"
            },
            {
                "label": "直陈痛点",
                "explanation": "直击用户潜在困扰或需求痛点，例如‘素颜暗沉没气色？’‘底妆易卡粉？’，引发观众强烈共鸣，促使继续看下去。"
            },
            {
                "label": "直陈效果",
                "explanation": "直接用一句话概括产品带来的显著效果，如‘一抹亮肤三度’‘上脸秒变水光肌’，让观众立刻理解核心卖点。"
            },
            {
                "label": "提出疑问",
                "explanation": "用问题句式开头，激发观众思考与兴趣，例如‘你的素颜发光了吗？’‘为什么粉底总是假面？’，提高互动和参与度。"
            },
            {
                "label": "引发好奇",
                "explanation": "制造悬念或神秘感，比如‘只需三秒，肌肤竟然这样了！’‘为什么她不用粉底还这么亮？’，吸引观众继续观看解锁答案。"
            },
            {
                "label": "正话反说",
                "explanation": "反向表达，引发意外和注意力，如‘不想变白千万别用它’‘懒人千万别看’，利用矛盾心理刺激停留。"
            },
            {
                "label": "塑造情绪",
                "explanation": "用情绪感染观众，例如‘早起再也不慌了！’‘素颜也自信出门’，打造积极、真实的共鸣场景，增强代入感。"
            }
        ]
    },
    "中间卖点": {
        "explanation": "在广告或销售中，强调产品的独特卖点和价值，以继续吸引客户。",
        "tags": [
            {"label": "外观", "explanation": "突出产品本身的包装设计、造型、颜色、质感等外部视觉元素，比如独特瓶身、便携小巧、限量色彩、时尚感外观等，让观众被产品本体吸引。"},
            {"label": "材料", "explanation": "详细介绍素颜霜中采用的主要成分、核心原料，如高端粉体、养肤配方、进口材料等，强调用料安全、品质保障、适合各种肤质。"},
            {"label": "工艺", "explanation": "突出产品在制造工艺上的优势，比如纳米研磨、微粒包裹、精细乳化等高端工艺带来的上妆服帖、不卡粉、易推开等体验。"},
            {"label": "价格", "explanation": "直接说明产品的优惠价格、性价比，或与大牌同款对比突出价格优势，让观众感觉物超所值。"},
            {"label": "功能", "explanation": "详细介绍产品具备的具体功效，如提亮肤色、遮盖瑕疵、隐形毛孔、保湿持久、防汗不脱妆等，让观众理解使用后带来的实际改变。"},
            {"label": "场景", "explanation": "结合日常生活场景，描述产品适用的具体环境，例如上班、约会、出门运动、健身房、旅行等，帮助观众代入真实使用感受。"},
            {"label": "地域", "explanation": "结合产品在特定地域的口碑或流行度，例如‘日韩爆款’‘国货之光’‘海外明星推荐’等，提升信任度和新鲜感。"},
            {"label": "人群", "explanation": "明确指出适用的人群特征，例如黄皮、暗沉、油皮、干皮、学生党、上班族、宝妈等，帮助观众对号入座，增强关联。"},
            {"label": "方法", "explanation": "具体说明产品的使用方法、用量、步骤，比如‘三秒抹匀’‘直接当面霜’‘素颜懒人必备’等，让观众更容易上手。"},
            {"label": "背书", "explanation": "强调权威背书或明星达人种草、医生推荐、成分党验证等，增强产品可信度和口碑影响力。"},
            {"label": "情怀", "explanation": "通过情感故事、品牌理念、用户真实体验等内容激发观众共鸣，如‘陪伴成长’‘天然守护’‘国货自信’等。"},
            {"label": "稀缺", "explanation": "突出产品限量、热卖、断货、抢购、稀有色号、限定版等稀缺属性，刺激观众的购买冲动。"}
        ]
    },

    "行动号召": {
        "explanation": "在广告或营销中，直接激励客户采取行动，达成购买或其他目标。",
        "tags": [
            {"label": "优惠诱导", "explanation": "通过限时折扣、满减、买赠、专属券等优惠信息，引导观众马上下单抢购，错过可惜。"},
            {"label": "饥饿营销", "explanation": "营造库存紧张、热卖断货、即将涨价等紧迫氛围，激发观众‘现在不买马上没货’的紧张感。"},
            {"label": "艾特人群", "explanation": "在文案中@姐妹、@闺蜜、@男朋友、@妈妈等，引导观众转发、分享、安利身边的人一起来下单。"},
            {"label": "从众引导", "explanation": "通过展示众多用户好评、销量排行榜、万人种草等内容，激发观众‘大家都买我也要买’的从众心理。"},
            {"label": "身份推荐", "explanation": "通过达人身份、医生、明星、专家等专业角色推荐，增强观众的信任感和决策信心。"}
        ]
    }
}

def split_sentences_with_punctuation(text):
    sentences = re.findall(r'[^，。！？；]*[，。！？；]*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def match_primary_label(sentence, tag_system):
    explanations = [v["explanation"] for v in tag_system.values()]
    labels = list(tag_system.keys())
    embedding = model.encode(sentence, convert_to_tensor=True, device=device)
    label_embeddings = model.encode(explanations, convert_to_tensor=True, device=device)
    cosine_scores = util.pytorch_cos_sim(embedding, label_embeddings)
    scores = cosine_scores.cpu().detach().numpy()[0]
    best_idx = np.argmax(scores)
    return labels[best_idx], explanations[best_idx], scores[best_idx]

def match_secondary_label(sentence, tags):
    tag_labels = [t['label'] for t in tags]
    tag_explanations = [t['explanation'] for t in tags]
    embedding = model.encode(sentence, convert_to_tensor=True, device=device)
    label_embeddings = model.encode(tag_labels, convert_to_tensor=True, device=device)
    cosine_scores = util.pytorch_cos_sim(embedding, label_embeddings)
    scores = cosine_scores.cpu().detach().numpy()[0]
    best_idx = np.argmax(scores)
    return tag_labels[best_idx], tag_explanations[best_idx], scores[best_idx]

# 测试脚本
script = "什么叫素颜发光？黄皮姐妹有福啦！平时涂粉底卡纹又假面，出门没多久领子就被蹭黄。这支素颜霜，直接当乳液三秒抹开，秒变妈生水光皮！它含大牌粉底同源粉体，成膜后扒得牢，穿黑T也蹭不掉。橄榄皮涂完秒变冷白皮，出汗健身自带柔光灯，闺蜜还以为你打了水光针！"
sentences = split_sentences_with_punctuation(script)

for idx, sentence in enumerate(sentences):
    if idx == 0:
        primary_label_category = "黄金三秒"
        primary_label_explanation = tag_system["黄金三秒"]["explanation"]
        primary_similarity_score = 1.0
        secondary_label = "直陈效果"
        explanation = "直接说明产品或服务的效果"
        similarity_score = 1.0
    else:
        # 只允许“中间卖点”和“行动号召”参与一级标签匹配
        restrict_tag_system = {k: v for k, v in tag_system.items() if k in ["中间卖点", "行动号召"]}
        primary_label_category, primary_label_explanation, primary_similarity_score = match_primary_label(
            sentence, restrict_tag_system
        )
        secondary_label, explanation, similarity_score = match_secondary_label(
            sentence, tag_system[primary_label_category]["tags"]
        )
    print(f"句子: {sentence}")
    print(f"一级标签: {primary_label_category} (解释: {primary_label_explanation}) (一级标签相似度: {primary_similarity_score:.4f})")
    print(f"二级标签: {secondary_label} (解释: {explanation}) (二级标签相似度: {similarity_score:.4f})\n")

