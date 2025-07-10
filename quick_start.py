from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import torch

# 检查 CUDA 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # 输出使用的设备

# 加载预训练模型并将其移动到 GPU（如果可用）
model = SentenceTransformer('sentence-transformers/sentence-t5-xxl')
model = model.to(device)  # 将模型移动到 GPU 或 CPU

# 定义一级标签及二级标签体系和解释
tag_system = {
    "黄金三秒": {
        "explanation": "在广告或宣传的前三秒，吸引观众注意力并触动他们的痛点或兴趣。",
        "tags": [
            {"label": "价格利益", "explanation": "通过强调价格优势吸引客户购买"},
            {"label": "身份推荐", "explanation": "通过推荐身份提升客户购买意愿"},
            {"label": "点名人群", "explanation": "明确指出目标客户群体，提高相关性"},
            {"label": "直陈痛点", "explanation": "直接陈述客户痛点，引起共鸣"},
            {"label": "直陈效果", "explanation": "直接说明产品或服务的效果"},
            {"label": "提出疑问", "explanation": "通过提出问题引发客户兴趣"},
            {"label": "引发好奇", "explanation": "通过制造悬念或好奇心吸引客户关注"},
            {"label": "正话反说", "explanation": "通过反向表达方式引发注意"},
            {"label": "塑造情绪", "explanation": "通过情感调动激发客户购买欲望"}
        ]
    },
    "中间卖点": {
        "explanation": "在广告或销售中，强调产品的独特卖点和价值，以继续吸引客户。",
        "tags": [
            {"label": "外观", "explanation": "通过强调产品外观来吸引客户"},
            {"label": "材料", "explanation": "强调产品使用的材料和质量"},
            {"label": "工艺", "explanation": "突出产品的工艺和制造过程"},
            {"label": "价格", "explanation": "通过强调价格优势或优惠来吸引客户"},
            {"label": "功能", "explanation": "展示产品的功能和实用性"},
            {"label": "场景", "explanation": "通过应用场景的展示来吸引目标用户"},
            {"label": "地域", "explanation": "强调产品在特定地域的独特性或优势"},
            {"label": "人群", "explanation": "突出产品适用于特定人群的特点"},
            {"label": "方法", "explanation": "介绍使用产品或服务的具体方法"},
            {"label": "背书", "explanation": "通过权威背书增强产品信任度"},
            {"label": "情怀", "explanation": "通过情感营销引发客户的共鸣"},
            {"label": "稀缺", "explanation": "强调产品的稀缺性，提高购买欲望"}
        ]
    },
    "行动号召": {
        "explanation": "在广告或营销中，直接激励客户采取行动，达成购买或其他目标。",
        "tags": [
            {"label": "优惠诱导", "explanation": "通过优惠信息激励客户行动"},
            {"label": "饥饿营销", "explanation": "通过营造紧迫感诱导客户快速购买"},
            {"label": "艾特人群", "explanation": "通过艾特特定人群引导他们进行购买"},
            {"label": "从众引导", "explanation": "通过从众心理引导客户进行购买"},
            {"label": "身份推荐", "explanation": "通过身份推荐激发客户购买欲望"}
        ]
    }
}

# 分句函数，保留标点符号
def split_sentences_with_punctuation(text):
    # 使用正则表达式根据标点符号进行分句并保留标点符号
    sentences = re.findall(r'[^，。！？；]*[，。！？；]*', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # 去除多余空格并去掉空字符串
    return sentences

# 规则匹配函数
def match_label(script, labels):
    # 使用 SentenceTransformer 模型计算语义嵌入并将脚本和标签转移到 GPU
    script_embedding = model.encode(script, convert_to_tensor=True, device=device)
    label_embeddings = model.encode([label['label'] for label in labels], convert_to_tensor=True, device=device)

    # 计算相似度
    cosine_scores = util.pytorch_cos_sim(script_embedding, label_embeddings)

    # 将结果从 GPU 转移到 CPU，再转换为 NumPy 数组
    cosine_scores_cpu = cosine_scores.cpu().detach().numpy()

    # 找到最相似的标签索引
    best_label_idx = np.argmax(cosine_scores_cpu)

    # 返回最相似的标签和对应的解释
    best_label = labels[best_label_idx]
    return best_label['label'], best_label['explanation'], cosine_scores_cpu[0][best_label_idx]

# 测试脚本
script = "什么叫素颜发光？黄皮姐妹有福啦！平时涂粉底卡纹又假面，出门没多久领子就被蹭黄。这支素颜霜，直接当乳液三秒抹开，秒变妈生水光皮！它含大牌粉底同源粉体，成膜后扒得牢，穿黑T也蹭不掉。橄榄皮涂完秒变冷白皮，出汗健身自带柔光灯，闺蜜还以为你打了水光针！"

# 分句并保留标点符号
sentences = split_sentences_with_punctuation(script)

# 对每一句进行标签匹配
for idx, sentence in enumerate(sentences):
    # 获取所有一级标签和二级标签
    all_labels = sum([category['tags'] for category in tag_system.values()], [])

    # 如果是第一句，标签固定为“黄金三秒”
    if idx == 0:
        primary_label_category = "黄金三秒"
        primary_label_explanation = tag_system["黄金三秒"]["explanation"]
        primary_label = "直陈效果"  # 可选择默认的二级标签（例如：直陈效果）
        secondary_label = primary_label
        explanation = "直接说明产品或服务的效果"
        similarity_score = 1.0  # 第一条固定标签，设置为最高相似度
    else:
        primary_label, explanation, similarity_score = match_label(sentence, all_labels)
        # 获取一级标签和解释
        primary_label_category = [category for category, data in tag_system.items() if any(label['label'] == primary_label for label in data['tags'])][0]
        primary_label_explanation = tag_system[primary_label_category]["explanation"]
        secondary_label = primary_label

    print(f"句子: {sentence}")
    print(f"一级标签: {primary_label_category} (解释: {primary_label_explanation})")
    print(f"二级标签: {secondary_label} (解释: {explanation}) (相似度: {similarity_score:.4f})\n")
