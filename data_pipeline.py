# data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt

# ===================== 路径配置变量，只改这里即可 =====================
BASE_DIR = r"D:\dnmp\www\ai-image-classifier"
DATA_DIR = f"{BASE_DIR}\\data"

# 输入数据集
INPUT_CSV = f"{DATA_DIR}\\dataset.csv"

# 输出文件
OUT_TRAIN_CSV = f"{DATA_DIR}\\train.csv"
OUT_VAL_CSV = f"{DATA_DIR}\\val.csv"
OUT_TEST_CSV = f"{DATA_DIR}\\test.csv"
OUT_REPORT_TXT = f"{DATA_DIR}\\data_quality_report.txt"
BOXPLOT_PNG = f"{DATA_DIR}\\boxplot_outlier.png"
# =================================================================


def detect_outliers_zscore(series, threshold=3):
    """Z‑score 3σ异常检测，返回异常布尔mask与异常数量，适合正态分布数据"""
    z_scores = stats.zscore(series.dropna())
    mask_valid = np.abs(z_scores) <= threshold
    outlier_count = np.sum(np.abs(z_scores) > threshold)
    return mask_valid, outlier_count


def draw_boxplot(df_clean, save_path=None):
    """对所有数值列绘制箱线图，直观查看异常值分布，可选保存图片"""
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return
    plt.figure(figsize=(12, 6))
    df_clean[numeric_cols].boxplot()
    plt.title("Boxplot for numeric columns (outlier visualization)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    # ===================== Step1 加载数据与基础探索 =====================
    df = pd.read_csv(INPUT_CSV)

    print("========== 【Step1 基础数据探索】 ==========")
    print(f"数据集形状(行,列): {df.shape}")
    print("\n各列数据类型 dtypes：")
    print(df.dtypes)
    print("\n数值字段统计信息 describe：")
    print(df.describe())

    # ===================== Step2 缺失值识别、多种填充策略对比 =====================
    print("\n========== 【Step2 缺失值统计】 ==========")
    missing_count = df.isnull().sum()
    missing_rate = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        "缺失数量": missing_count,
        "缺失率(%)": missing_rate.round(2)
    })
    print(missing_df)

    # 策略1：直接删除含缺失样本
    df_drop_na = df.dropna()
    print(f"\n策略1 dropna 删除缺失后样本数：{df_drop_na.shape[0]}")

    # 策略2：数值列填充均值，类别列填充众数
    df_fill_mean_mode = df.copy()
    for col in df_fill_mean_mode.columns:
        if df_fill_mean_mode[col].dtype in [np.float64, np.int64]:
            df_fill_mean_mode[col] = df_fill_mean_mode[col].fillna(df_fill_mean_mode[col].mean())
        else:
            df_fill_mean_mode[col] = df_fill_mean_mode[col].fillna(df_fill_mean_mode[col].mode()[0])
    print(f"策略2 均值/众数填充后，剩余缺失：{df_fill_mean_mode.isnull().sum().sum()}")

    # 策略3：数值列填充中位数
    df_fill_median = df.copy()
    num_cols = df_fill_median.select_dtypes(include=[np.number]).columns
    df_fill_median[num_cols] = df_fill_median[num_cols].fillna(df_fill_median[num_cols].median())
    # 非数值列众数填充
    cat_cols = df_fill_median.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        df_fill_median[c] = df_fill_median[c].fillna(df_fill_median[c].mode()[0])
    print(f"策略3 中位数/众数填充后，剩余缺失：{df_fill_median.isnull().sum().sum()}")

    # ===================== Step3 检测删除重复记录 =====================
    print("\n========== 【Step3 重复数据处理】 ==========")
    dup_count = df_fill_median.duplicated().sum()
    print(f"检测到重复记录数量：{dup_count}")
    df_clean = df_fill_median.drop_duplicates().copy()
    print(f"去重完成，清洗后数据集大小：{df_clean.shape}")

    # ===================== Step3‑1 异常值检测 Z‑score(±3σ) + IQR + 箱线图可视化 =====================
    print("\n========== 【Step3‑1 异常值检测 Z‑score(±3σ) & IQR】 ==========")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    draw_boxplot(df_clean, save_path=BOXPLOT_PNG)
    print(f"箱线图已保存：{BOXPLOT_PNG}")

    zscore_outlier_info = {}
    iqr_outlier_info = {}

    for col in numeric_cols:
        # Z‑score 3σ异常检测（适合正态分布）
        _, z_out_cnt = detect_outliers_zscore(df_clean[col], threshold=3)
        zscore_outlier_info[col] = z_out_cnt

        # IQR四分位异常检测（通用，不要求正态）
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_out_cnt = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)].shape[0]
        iqr_outlier_info[col] = iqr_out_cnt

        print(f"{col:20s} | Z‑score(±3σ)异常：{z_out_cnt:4d} | IQR异常：{iqr_out_cnt:4d}")

    # ========== 可选：开启下面代码块，将 |z|>3 的异常样本剔除 ==========
    # keep_mask = np.ones(len(df_clean), dtype=bool)
    # for col in numeric_cols:
    #     mask_z, _ = detect_outliers_zscore(df_clean[col], threshold=3)
    #     keep_mask = keep_mask & mask_z
    # df_clean = df_clean[keep_mask].copy()
    # print(f"\n执行Z‑score异常过滤后数据集大小：{df_clean.shape}")

    # ===================== Step4 6:2:2 划分训练/验证/测试集 =====================
    print("\n========== 【Step4 数据集划分 6:2:2】 ==========")
    # 先分出60%训练集；剩余40%对半拆分：验证20%，测试20%
    train_df, temp_df = train_test_split(df_clean, test_size=0.4, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

    print(f"训练集 train: {train_df.shape}")
    print(f"验证集 val:   {val_df.shape}")
    print(f"测试集 test:  {test_df.shape}")

    train_df.to_csv(OUT_TRAIN_CSV, index=False)
    val_df.to_csv(OUT_VAL_CSV, index=False)
    test_df.to_csv(OUT_TEST_CSV, index=False)
    print(f"\n已输出：\n{OUT_TRAIN_CSV}\n{OUT_VAL_CSV}\n{OUT_TEST_CSV}")

    # ===================== Step5 生成数据质量报告 =====================
    print("\n========== 【Step5 数据质量报告】 ==========")
    report_lines = []
    report_lines.append("======= 数据质量报告 Data Quality Report =======\n")
    report_lines.append(f"原始数据集大小：{df.shape}\n")
    report_lines.append(f"清洗后数据集大小：{df_clean.shape}\n")
    report_lines.append(f"重复记录总数：{dup_count}\n")
    report_lines.append("\n---各列缺失统计---\n")
    report_lines.append(missing_df.to_string())

    report_lines.append("\n\n---异常值统计 Z‑score(±3σ，正态分布适用) | IQR(通用)---")
    for col in numeric_cols:
        report_lines.append(f"{col:20s} | Z‑score异常数：{zscore_outlier_info[col]:4d} | IQR异常数：{iqr_outlier_info[col]:4d}")

    report_lines.append("\n说明：")
    report_lines.append("1. Z‑score(|z|>3)：适合近似正态分布数据；超过3个标准差判定为异常。")
    report_lines.append("2. IQR 1.5*IQR：不要求正态分布，通用异常检测方法。")
    report_lines.append("3. 箱线图可视化文件：boxplot_outlier.png")
    report_lines.append("4. 本脚本仅统计异常，默认不删除异常样本，可打开注释执行剔除。")

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n✅ 数据质量报告已保存至 {OUT_REPORT_TXT}")


if __name__ == "__main__":
    main()
