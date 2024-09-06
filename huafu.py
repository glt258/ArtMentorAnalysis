import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.cm import ScalarMappable
import pandas as pd  # Import pandas


# 自定义颜色映射，渐变从灰白色到深绿色
cmap = plt.cm.get_cmap("Greens")  # 使用 Matplotlib 的 "Greens" 渐变色
gray_color = "#0f0f0f"  # 用于表示无数据的灰白色
norm = Normalize(vmin=0, vmax=1)

# 定义9个维度名称
dimensions = [
    "Realistic", "Deformation", "Imagination", "Color Richness",
    "Color Contrast", "Line Combination", "Line Texture",
    "Picture Organization", "Transformation"
]

# 1. 加载JSON数据
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# 2. 定义标准化函数
def normalize(value, min_value, max_value):
    if max_value - min_value == 0:
        return 0
    return (value - min_value) / (max_value - min_value)

# 3. 计算评分差异度（SD），并反转其值，让差异越小越接近1
def calculate_sd(score_data):
    total_diff = 0
    valid_rounds = 0

    for round_data in score_data:
        if round_data["round"] == 1:
            continue  # 跳过初始化的 round 1 数据
        gpt_score = round_data['data']['scores']['initGPTscore']
        user_score = round_data['data']['scores']['current']

        if gpt_score is not None and user_score is not None:
            try:
                gpt_score = float(gpt_score)
                user_score = float(user_score)
                score_diff = abs(gpt_score - user_score)
                total_diff += score_diff
                valid_rounds += 1
            except ValueError:
                continue

    if valid_rounds == 0:
        return np.nan  # 无数据则返回空值

    sd = total_diff / valid_rounds
    normalized_sd = 1 - normalize(sd, 0, 5)
    return normalized_sd

# 4. 计算SC (评分接受度)
def calculate_sc(score_data):
    gpt_scores = []
    user_scores = []

    for round_data in score_data:
        if round_data["round"] == 1:
            continue
        gpt_score = round_data['data']['scores']['initGPTscore']
        user_score = round_data['data']['scores']['current']
        if gpt_score is not None and user_score is not None:
            gpt_scores.append(float(gpt_score))
            user_scores.append(float(user_score))

    if len(gpt_scores) < 1 or len(user_scores) < 1:
        return np.nan  # 无数据则返回空值

    score_diffs = np.abs(np.array(gpt_scores) - np.array(user_scores))
    max_diff = 5.0
    normalized_diff = 1 - (np.mean(score_diffs) / max_diff)
    return normalize(normalized_diff, 0, 1)

# 5. 计算SV（评分波动度），并反转其值
def calculate_sv(score_data):
    user_scores = []
    for round_data in score_data:
        if round_data["round"] == 1:
            continue
        user_score = round_data['data']['scores']['current']
        if user_score is not None:
            user_scores.append(float(user_score))

    if len(user_scores) <= 0:
        return np.nan  # 无数据则返回空值

    sv = np.std(user_scores)
    normalized_sv = 1 - normalize(sv, 0, 5)
    return normalized_sv

# 6. 计算TAR（文本修改率），并反转其值
def calculate_tar(text_data):
    total_added = total_removed = total_gpt_length = 0
    for round_data in text_data:
        if round_data["round"] == 1:
            continue
        added = round_data['data']['Reviews']['added']
        removed = round_data['data']['Reviews']['removed']
        original = round_data['data']['Reviews']['original']
        total_added += len(added)
        total_removed += len(removed)
        total_gpt_length += len(original)

    if total_added + total_gpt_length > 0:
        tar = total_removed / (total_added + total_gpt_length)
        normalized_tar = 1 - normalize(tar, 0, 1)
        return normalized_tar
    return np.nan  # 无数据则返回空值

# 7. 计算文本相似度（TS）
def calculate_text_similarity(text_data, is_suggestion=False):
    gpt_texts = []
    user_texts = []
    for round_data in text_data:
        if round_data["round"] == 1:
            continue
        if is_suggestion:
            gpt_text = round_data['data']['suggestions']['original']
            user_text = round_data['data']['suggestions']['current']
        else:
            gpt_text = round_data['data']['Reviews']['original']
            user_text = round_data['data']['Reviews']['current']
        if gpt_text.strip() and user_text.strip():
            gpt_texts.append(gpt_text)
            user_texts.append(user_text)

    if not gpt_texts or not user_texts:
        return np.nan  # 无数据则返回空值

    vectorizer = CountVectorizer(analyzer='char').fit_transform([gpt_texts[-1], user_texts[-1]])
    vectors = vectorizer.toarray()
    if len(vectors) > 1:
        similarity = cosine_similarity(vectors)[0][1]
        return normalize(similarity, 0, 1)
    return np.nan  # 无数据则返回空值

# 生成圆角矩形
def draw_rounded_square(ax, color, x, y, size=0.9, radius=0.2):
    square = FancyBboxPatch((x, y), size, size,
                            boxstyle=f"round,pad=0,rounding_size={radius}",
                            facecolor=color, edgecolor='none')
    ax.add_patch(square)

# 生成华夫饼图
def plot_custom_waffle_chart(image_metrics, metric_name):
    n_rows = 4  # 固定每行4个格子
    n_cols = 5  # 固定每列5个格子

    values = [metrics[metric_name] for metrics in image_metrics]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    for i, value in enumerate(values):
        x = i % n_cols
        y = n_rows - 1 - i // n_cols
        color = gray_color if np.isnan(value) else cmap(norm(value))
        draw_rounded_square(ax, color, x, y)

    # 添加颜色渐变图例
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.04)
    cbar.set_label(metric_name)

    plt.show()

# 9. 批量处理文件并计算每张图片的指标
def process_directory(score_review_dir, suggestion_dir):
    image_metrics = []
    for image_num in range(1, 21):
        sc_values, sv_values, tar_values, ts_values, sd_values = [], [], [], [], []

        for dimension in dimensions:
            score_review_file = os.path.join(score_review_dir, f"{image_num}.jpg_{dimension}_score_Review.json")
            suggestion_file = os.path.join(suggestion_dir, f"{image_num}.jpg_{dimension}_suggestion.json")

            score_review_data = load_json_data(score_review_file)
            if score_review_data:
                sc_values.append(calculate_sc(score_review_data))
                sv_values.append(calculate_sv(score_review_data))
                tar_values.append(calculate_tar(score_review_data))
                sd_values.append(calculate_sd(score_review_data))
                ts_values.append(calculate_text_similarity(score_review_data))

            suggestion_data = load_json_data(suggestion_file)
            if suggestion_data:
                ts_values.append(calculate_text_similarity(suggestion_data, is_suggestion=True))

        if sc_values:
            avg_sc = np.nanmean(sc_values)
            avg_sv = np.nanmean(sv_values)
            avg_tar = np.nanmean(tar_values)
            avg_ts = np.nanmean(ts_values)
            avg_sd = np.nanmean(sd_values)

            # 打印每张图片的各个指标
            print(f"Image: {image_num}.jpg, SC: {avg_sc}, SV: {avg_sv}, TAR: {avg_tar}, TS: {avg_ts}, SD: {avg_sd}")

            image_metrics.append({
                "image": f"{image_num}.jpg",
                "SC": avg_sc,
                "SV": avg_sv,
                "TAR": avg_tar,
                "TS": avg_ts,
                "SD": avg_sd
            })

        # 将结果保存到 DataFrame
    df = pd.DataFrame(image_metrics)

    # 导出到 Excel 文件
    output_file = "image_metrics.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Metrics exported to {output_file}")

    # 生成各项指标的华夫饼图
    plot_custom_waffle_chart(image_metrics, "SC")
    plot_custom_waffle_chart(image_metrics, "SV")
    plot_custom_waffle_chart(image_metrics, "TAR")
    plot_custom_waffle_chart(image_metrics, "TS")
    plot_custom_waffle_chart(image_metrics, "SD")


# 主函数入口
if __name__ == "__main__":
    # 指定存放JSON文件的目录
    score_review_directory = "score_Review"
    suggestion_directory = "suggestion"

    process_directory(score_review_directory, suggestion_directory)
