import os
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class HyperParams:
    PREPROCESSED_DIR = "preprocessed_data"
    LANDUSE_TARGETS = [1, 2, 3, 4, 5, 6]
    MODEL_SAVE_DIR = "model_training/saved_models"
    STAT_SAVE_PATH = "model_training/data_stats.npy"
    SAMPLING_STATS_PATH = "model_training/sampling_stats.npy"

    TARGET_SAMPLING_RATE = 0.75
    SPECIAL_LU_RATES = {3: 0.8}

    NEIGHBORHOOD_SIZE = 7
    AUXILIARY_VARS = [
        'avg_radiance', 'blh', 'd2m', 'elevation', 'NDVI',
        'population_density', 'sp', 'str', 't2m', 'tp', 'u10', 'v10'
    ]
    DERIVED_VARS = [
        ('NDVI', 'elevation', 'product', 'NDVI_elevation_product'),
        ('t2m', 'd2m', 'ratio', 't2m_d2m_ratio'),
        ('sp', 'blh', 'product', 'sp_blh_product'),
        ('t2m', 'NDVI', 'product', 't2m_NDVI_product'),
        ('population_density', 'avg_radiance', 'product', 'pop_radiance_product')
    ]
    XCO2_MIN = 380.0
    XCO2_MAX = 440.0

    REQUIRED_SPATIO_TEMPORAL_FIELDS = ["time", "latitude", "longitude"]
    GENERATED_ST_FEATURES = [
        "time_ordinal_norm", "season_sin", "season_cos", "lat_norm", "lon_norm"
    ]
    ST_BRANCH_HIDDEN_DIM = 256
    ST_BRANCH_OUTPUT_DIM = 512

    VAL_RATIO = 0.2
    RANDOM_SEED = 42
    CONV1_CHANNELS = 96
    CONV2_CHANNELS = 192
    CONV3_CHANNELS = 384
    CONV4_CHANNELS = 512
    TRANSFORMER_HEADS = 8
    TRANSFORMER_LAYERS = 4
    FC1_UNITS = 256
    FC2_UNITS = 128
    DROPOUT_INIT = 0.1
    DROPOUT_FINAL = 0.3
    BATCH_SIZE = 128
    EPOCHS = 60
    LR_CNN = 1.5e-4
    LR_TRANSFORMER = 3e-4
    LR_FC = 5e-4
    LR_ST_BRANCH = 5e-4
    WEIGHT_DECAY = 1e-5
    SPATIAL_LOSS_WEIGHT = 0.03
    GRAD_PENALTY_WEIGHT = 0.07
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    EARLY_STOP_PATIENCE = 15
    EARLY_STOP_MIN_DELTA = 0.003
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataProcessor:
    @staticmethod
    def create_directories():
        os.makedirs(HyperParams.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(HyperParams.STAT_SAVE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(HyperParams.SAMPLING_STATS_PATH), exist_ok=True)
        print(f"使用设备: {HyperParams.DEVICE}")

    @staticmethod
    def _add_derived_features(neighborhood, landuse_type, sample_idx):
        required_raw_vars = set()
        for var1, var2, _, _ in HyperParams.DERIVED_VARS:
            required_raw_vars.add(var1)
            required_raw_vars.add(var2)

        missing_raw_vars = [v for v in required_raw_vars if v not in neighborhood]
        if missing_raw_vars:
            print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：缺失原始空间特征 {missing_raw_vars}")
            return None

        for var in required_raw_vars:
            if not isinstance(neighborhood[var], np.ndarray):
                print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：特征 {var} 不是numpy数组（类型：{type(neighborhood[var])}）")
                return None

        try:
            for var1, var2, op, feat_name in HyperParams.DERIVED_VARS:
                val1 = neighborhood[var1]
                val2 = neighborhood[var2]

                if val1.shape != val2.shape:
                    print(
                        f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：衍生特征 {feat_name} 原始特征形状不匹配（{var1}:{val1.shape}, {var2}:{val2.shape}）")
                    return None

                if op == "product":
                    derived_val = val1 * val2
                elif op == "ratio":
                    derived_val = np.divide(val1, val2, out=np.zeros_like(val1), where=val2 != 0)
                else:
                    print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：不支持的衍生特征操作 {op}（特征：{feat_name}）")
                    return None

                neighborhood[feat_name] = derived_val
            return neighborhood
        except Exception as e:
            print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：生成衍生特征失败 - {str(e)}")
            return None

    @staticmethod
    def _collect_all_timestamps(time_to_data):
        all_timestamps = []
        for time_str, samples in time_to_data.items():
            if samples:
                all_timestamps.append(time_str)
        unique_timestamps = sorted(list(set(all_timestamps)))
        return {ts: idx + 1 for idx, ts in enumerate(unique_timestamps)}

    @staticmethod
    def _generate_spatiotemporal_features(sample, time_ordinal_map, landuse_type, sample_idx):
        try:
            time_str = sample["time"]
            latitude = float(sample["latitude"])
            longitude = float(sample["longitude"])

            if time_str not in time_ordinal_map:
                print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：时间戳 {time_str} 不在有效时间映射中")
                return None

            total_timesteps = len(time_ordinal_map)
            time_ordinal = time_ordinal_map[time_str]
            sample["time_ordinal_norm"] = time_ordinal / total_timesteps

            if len(time_str) != 6:
                print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：时间格式错误（{time_str}），需为'YYYYMM'（如202309）")
                return None
            try:
                month = int(time_str[4:6])
                if not (1 <= month <= 12):
                    raise ValueError(f"月份 {month} 超出范围（1-12）")
            except ValueError as e:
                print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：时间解析失败 - {str(e)}")
                return None
            sample["season_sin"] = np.sin(2 * np.pi * month / 12)
            sample["season_cos"] = np.cos(2 * np.pi * month / 12)

            original_longitude = longitude
            if longitude > 180:
                longitude = longitude - 360
            elif longitude < -180:
                longitude = longitude + 360

            if not (-180 <= longitude <= 180):
                print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：转换后经度 {longitude}° 仍超出范围（-180~180°）")
                return None

            if not (-90 <= latitude <= 90):
                print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：纬度 {latitude}° 超出范围（-90~90°）")
                return None

            sample["lat_norm"] = (latitude + 90) / 180
            sample["lon_norm"] = (longitude + 180) / 360

            return sample
        except Exception as e:
            print(f"样本无效（landuse={landuse_type}, 索引={sample_idx}）：生成时空特征失败 - {str(e)}")
            return None

    @staticmethod
    def load_landuse_data():
        print("\n===== 开始加载多landuse数据（含详细错误日志） =====")
        print(f"当前工作目录: {os.getcwd()}")
        data_dir = HyperParams.PREPROCESSED_DIR
        print(f"数据目录绝对路径: {os.path.abspath(data_dir)}")

        if not os.path.exists(data_dir):
            print(f"错误: 数据目录不存在 - {os.path.abspath(data_dir)}")
            return {}, 0, 0, {}

        all_files = os.listdir(data_dir)
        print(f"数据目录中的所有文件: {all_files}")

        target_prefixes = [f"landuse_{lu}_" for lu in HyperParams.LANDUSE_TARGETS]
        landuse_files = [
            f for f in all_files
            if any(f.startswith(prefix) for prefix in target_prefixes) and f.endswith(".npy")
        ]

        print(f"找到符合条件的文件数量: {len(landuse_files)}")
        print(f"符合条件的文件列表: {landuse_files}")

        if not landuse_files:
            print(f"错误: 未找到landuse={HyperParams.LANDUSE_TARGETS}的文件")
            return {}, 0, 0, {}

        time_to_raw_samples = defaultdict(list)
        for file in sorted(landuse_files, key=lambda x: x.split("_")[-1].replace(".npy", "")):
            try:
                file_parts = file.split("_")
                if len(file_parts) < 3 or not file_parts[1].isdigit():
                    raise ValueError(f"文件名格式错误，需为'landuse_{{lu}}_时间戳.npy'（当前：{file}）")
                lu_str = file_parts[1]
                landuse_type = int(lu_str)
                if landuse_type not in HyperParams.LANDUSE_TARGETS:
                    print(f"警告: 文件{file}的landuse={landuse_type}不在目标列表中，跳过")
                    continue
            except (IndexError, ValueError) as e:
                print(f"警告: 解析文件{file}的landuse类型失败 - {str(e)}，跳过")
                continue

            time_str = file.split("_")[-1].replace(".npy", "")
            file_path = os.path.join(data_dir, file)
            print(f"\n处理文件: {file}（landuse={landuse_type}, 时间戳={time_str}）")

            if not os.path.exists(file_path):
                print(f"警告: 文件不存在 - {file_path}，跳过")
                continue
            if not os.access(file_path, os.R_OK):
                print(f"警告: 无文件读取权限 - {file_path}，跳过")
                continue

            try:
                data = np.load(file_path, allow_pickle=True).tolist()
                if not isinstance(data, list):
                    raise TypeError(f"文件内容不是列表（类型：{type(data)}），无法解析为样本集")

                valid_raw_samples = []
                for s_idx, sample in enumerate(data):
                    required_fields = ["neighborhood", "xco2"] + HyperParams.REQUIRED_SPATIO_TEMPORAL_FIELDS
                    missing_fields = [f for f in required_fields if f not in sample]

                    if not missing_fields:
                        sample["landuse_type"] = landuse_type
                        valid_raw_samples.append(sample)
                    else:
                        print(f"样本{s_idx}（landuse={landuse_type}）基础字段缺失: {missing_fields}，丢弃")

                time_to_raw_samples[time_str].extend(valid_raw_samples)
                print(
                    f"文件{file}处理完成：原始{len(data)} → 保留{len(valid_raw_samples)}个含基础字段的样本 | 时间戳{time_str}累计样本数：{len(time_to_raw_samples[time_str])}")
            except Exception as e:
                print(f"加载文件{file}失败: {str(e)}，跳过")
                continue

        time_ordinal_map = DataProcessor._collect_all_timestamps(time_to_raw_samples)
        if not time_ordinal_map:
            raise ValueError("没有有效时间戳，无法生成时间特征")
        print(f"\n时间序数映射生成完成：共{len(time_ordinal_map)}个时间片（从1开始编号）")

        time_to_data = {}
        total_valid = 0
        total_invalid = 0
        landuse_stats = {lu: 0 for lu in HyperParams.LANDUSE_TARGETS}

        for time_str, raw_samples in time_to_raw_samples.items():
            if not raw_samples:
                continue

            valid_samples = []
            xco2_values = []
            for s in raw_samples:
                try:
                    float_xco2 = float(s["xco2"])
                    if HyperParams.XCO2_MIN <= float_xco2 <= HyperParams.XCO2_MAX:
                        xco2_values.append(float_xco2)
                except (ValueError, TypeError):
                    continue

            if xco2_values:
                print(
                    f"时间片{time_str} XCO2统计: 均值={np.mean(xco2_values):.2f}, 标准差={np.std(xco2_values):.2f}, 范围=[{min(xco2_values):.2f}, {max(xco2_values):.2f}]")
            else:
                print(f"警告: 时间片{time_str}没有有效xco2值，跳过")
                continue

            for sample_idx, sample in enumerate(raw_samples):
                landuse_type = sample["landuse_type"]
                try:
                    xco2_val = sample["xco2"]
                    try:
                        float_xco2 = float(xco2_val)
                    except (ValueError, TypeError):
                        raise ValueError(f"xco2值不是有效数字（值：{xco2_val}）")

                    if np.isnan(float_xco2) or np.isinf(float_xco2):
                        raise ValueError(f"xco2是NaN/Inf")
                    if not (HyperParams.XCO2_MIN <= float_xco2 <= HyperParams.XCO2_MAX):
                        raise ValueError(
                            f"xco2超出范围（{HyperParams.XCO2_MIN}-{HyperParams.XCO2_MAX}，实际值：{float_xco2:.2f}）")
                    sample["xco2"] = float_xco2

                    updated_neighborhood = DataProcessor._add_derived_features(
                        sample["neighborhood"], landuse_type, sample_idx
                    )
                    if updated_neighborhood is None:
                        raise ValueError("衍生特征生成失败（见上文详细日志）")
                    sample["neighborhood"] = updated_neighborhood

                    sample = DataProcessor._generate_spatiotemporal_features(
                        sample, time_ordinal_map, landuse_type, sample_idx
                    )
                    if sample is None:
                        raise ValueError("时空特征生成失败（见上文详细日志）")

                    spatial_vars = HyperParams.AUXILIARY_VARS + [f[3] for f in HyperParams.DERIVED_VARS]
                    missing_spatial = [v for v in spatial_vars if v not in sample["neighborhood"]]
                    if missing_spatial:
                        raise ValueError(f"缺少空间特征：{missing_spatial}")

                    missing_st = [v for v in HyperParams.GENERATED_ST_FEATURES if v not in sample]
                    if missing_st:
                        raise ValueError(f"缺少时空特征：{missing_st}")

                    expected_shape = (HyperParams.NEIGHBORHOOD_SIZE, HyperParams.NEIGHBORHOOD_SIZE)
                    for var in spatial_vars:
                        if sample["neighborhood"][var].shape != expected_shape:
                            raise ValueError(f"特征{var}形状异常（预期{expected_shape}，实际{sample['neighborhood'][var].shape}）")

                    valid_samples.append(sample)
                    total_valid += 1
                    landuse_stats[landuse_type] += 1

                except Exception as e:
                    print(f"样本{sample_idx}（landuse={landuse_type}）无效: {str(e)}，丢弃")
                    total_invalid += 1

            time_to_data[time_str] = valid_samples
            print(
                f"时间片{time_str}处理完成: 原始{len(raw_samples)} → 有效{len(valid_samples)} → 无效{len(raw_samples) - len(valid_samples)}")

        print(f"\n===== 数据加载完成 =====")
        print(f"有效样本总数: {total_valid}")
        print(f"无效样本总数: {total_invalid}")
        print(f"各landuse有效样本分布: {landuse_stats}")

        missing_landuses = [lu for lu in HyperParams.LANDUSE_TARGETS if landuse_stats[lu] == 0]
        if missing_landuses:
            print(f"警告: 以下landuse类型无有效样本: {missing_landuses}")

        if total_valid == 0:
            raise ValueError("无有效样本，无法继续训练")

        return time_to_data, total_valid, total_invalid, landuse_stats

    @staticmethod
    def stratified_sample_by_time_landuse(time_to_data, random_seed=HyperParams.RANDOM_SEED):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        sampled_time_to_data = defaultdict(list)
        sampling_stats = {
            "original_total": 0,
            "sampled_total": 0,
            "time_landuse_stats": defaultdict(dict)
        }

        for time_str, samples in time_to_data.items():
            landuse_samples_in_time = defaultdict(list)
            for sample in samples:
                lu = sample["landuse_type"]
                landuse_samples_in_time[lu].append(sample)
                sampling_stats["original_total"] += 1

            for lu, lu_samples in landuse_samples_in_time.items():
                n_original = len(lu_samples)
                sampling_rate = HyperParams.SPECIAL_LU_RATES.get(lu, HyperParams.TARGET_SAMPLING_RATE)
                n_sampled = max(1, int(n_original * sampling_rate))
                sampled_indices = np.random.choice(n_original, size=n_sampled, replace=False)
                sampled_samples = [lu_samples[idx] for idx in sampled_indices]

                sampled_time_to_data[time_str].extend(sampled_samples)
                sampling_stats["sampled_total"] += n_sampled

                sampling_stats["time_landuse_stats"][f"{time_str}-LU{lu}"] = {
                    "original_count": n_original,
                    "sampled_count": n_sampled,
                    "sampling_rate": n_sampled / n_original if n_original != 0 else 0.0
                }

        actual_sampling_rate = sampling_stats["sampled_total"] / sampling_stats["original_total"] if sampling_stats[
                                                                                                         "original_total"] != 0 else 0.0
        sampling_stats["actual_sampling_rate"] = actual_sampling_rate

        print("\n===== 按时间片+Landuse分层抽样统计 =====")
        print(f"原总样本数: {sampling_stats['original_total']}")
        print(f"抽样后总样本数: {sampling_stats['sampled_total']}")
        print(f"目标抽样率: {HyperParams.TARGET_SAMPLING_RATE:.1%}, 实际抽样率: {actual_sampling_rate:.1%}")
        print(f"特殊Landuse抽样率: {HyperParams.SPECIAL_LU_RATES}")
        print("\n各时间片-Landuse单元抽样详情（前10个单元）:")
        time_landuse_items = list(sampling_stats["time_landuse_stats"].items())[:10]
        for unit, stats in time_landuse_items:
            print(f"  {unit}: 原{stats['original_count']} → 抽{stats['sampled_count']}（率{stats['sampling_rate']:.1%}）")
        if len(sampling_stats["time_landuse_stats"]) > 10:
            print(f"  ... 共{len(sampling_stats['time_landuse_stats'])}个单元，其余省略")

        np.save(HyperParams.SAMPLING_STATS_PATH, sampling_stats)
        print(f"\n抽样统计已保存至: {HyperParams.SAMPLING_STATS_PATH}")

        return sampled_time_to_data, sampling_stats

    @staticmethod
    def split_train_val(time_to_data, total_valid, landuse_stats):
        landuse_groups = {}
        for time_str, samples in time_to_data.items():
            for s in samples:
                lu = s["landuse_type"]
                if lu not in landuse_groups:
                    landuse_groups[lu] = []
                landuse_groups[lu].append(s)

        train_data = []
        val_data = []
        split_stats = {}

        for lu in HyperParams.LANDUSE_TARGETS:
            group_samples = landuse_groups.get(lu, [])
            if not group_samples:
                split_stats[lu] = {"训练样本数": 0, "验证样本数": 0, "原因": "无有效样本"}
                continue

            if len(group_samples) < 5:
                train_data.extend(group_samples)
                split_stats[lu] = {
                    "训练样本数": len(group_samples),
                    "验证样本数": 0,
                    "原因": f"样本太少（{len(group_samples)}个），全部分配到训练集"
                }
                continue

            train_split, val_split = train_test_split(
                group_samples,
                test_size=HyperParams.VAL_RATIO,
                random_state=HyperParams.RANDOM_SEED,
                stratify=[s["landuse_type"] for s in group_samples]
            )
            train_data.extend(train_split)
            val_data.extend(val_split)
            split_stats[lu] = {
                "训练样本数": len(train_split),
                "验证样本数": len(val_split),
                "验证比例": f"{len(val_split) / len(group_samples):.2%}"
            }

        print(f"\n数据拆分统计（按landuse类型）: {split_stats}")
        print(f"总训练样本: {len(train_data)}, 总验证样本: {len(val_data)}")

        np.save(HyperParams.STAT_SAVE_PATH, {
            "split_stats": split_stats,
            "total_train": len(train_data),
            "total_val": len(val_data),
            "total_valid": total_valid,
            "landuse_stats": landuse_stats
        })

        if len(train_data) == 0 or len(val_data) == 0:
            raise ValueError("数据拆分后训练集/验证集为空")

        return train_data, val_data

    @staticmethod
    def normalize_data(data, stats=None, is_train=True):
        spatial_vars = HyperParams.AUXILIARY_VARS + [
            feat_name for _, _, _, feat_name in HyperParams.DERIVED_VARS
        ]
        st_vars = HyperParams.GENERATED_ST_FEATURES

        inputs = []
        st_features = []
        labels = []
        landuse_types = []

        for sample in data:
            try:
                neighbor = np.stack([
                    sample["neighborhood"][var] for var in spatial_vars
                ], axis=0).astype(np.float32)
                if np.isnan(neighbor).any() or np.isinf(neighbor).any():
                    lu = sample["landuse_type"]
                    print(f"样本（landuse={lu}）标准化失败：空间特征包含NaN/Inf，丢弃")
                    continue

                st_feat = np.array([
                    sample[var] for var in st_vars
                ], dtype=np.float32)
                if np.isnan(st_feat).any() or np.isinf(st_feat).any():
                    lu = sample["landuse_type"]
                    print(f"样本（landuse={lu}）标准化失败：时空特征包含NaN/Inf，丢弃")
                    continue

                inputs.append(neighbor)
                st_features.append(st_feat)
                labels.append(sample["xco2"])
                landuse_types.append(sample["landuse_type"])

            except Exception as e:
                lu = sample.get("landuse_type", "未知")
                print(f"样本（landuse={lu}）标准化失败: {str(e)}，丢弃")
                continue

        inputs = np.array(inputs)
        st_features = np.array(st_features)
        labels = np.array(labels).reshape(-1, 1).astype(np.float32)
        landuse_types = np.array(landuse_types)
        print(f"标准化前 - 空间特征形状: {inputs.shape}, 时空特征形状: {st_features.shape}, 标签形状: {labels.shape}")

        if len(inputs) == 0 or len(labels) == 0:
            raise ValueError("标准化后无有效样本，无法继续训练")

        if is_train:
            spatial_means = inputs.mean(axis=(0, 2, 3), keepdims=True)
            spatial_stds = np.maximum(inputs.std(axis=(0, 2, 3), keepdims=True), 1e-6)

            st_means = st_features.mean(axis=0, keepdims=True)
            st_stds = np.maximum(st_features.std(axis=0, keepdims=True), 1e-6)

            label_mean = labels.mean(axis=0, keepdims=True)
            label_std = np.maximum(labels.std(axis=0, keepdims=True), 1e-6)

            stats = {
                "spatial_mean": spatial_means,
                "spatial_std": spatial_stds,
                "st_mean": st_means,
                "st_std": st_stds,
                "label_mean": label_mean,
                "label_std": label_std,
                "spatial_vars": spatial_vars,
                "st_vars": st_vars
            }
            print(f"计算标准化参数 - 空间特征均值形状: {spatial_means.shape}, 时空特征均值: {st_means.flatten()}")

            existing_stats = np.load(HyperParams.STAT_SAVE_PATH, allow_pickle=True).item()
            existing_stats["normalization"] = stats
            np.save(HyperParams.STAT_SAVE_PATH, existing_stats)

        normalized_inputs = (inputs - stats["spatial_mean"]) / stats["spatial_std"]
        normalized_st = (st_features - stats["st_mean"]) / stats["st_std"]
        normalized_labels = (labels - stats["label_mean"]) / stats["label_std"]

        return (normalized_inputs, normalized_st, normalized_labels, stats, landuse_types) if is_train else \
            (normalized_inputs, normalized_st, normalized_labels, landuse_types)

    @staticmethod
    def denormalize_label(preds, stats_path=HyperParams.STAT_SAVE_PATH):
        stats = np.load(stats_path, allow_pickle=True).item()["normalization"]
        label_mean = stats["label_mean"]
        label_std = stats["label_std"]

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().detach().numpy()

        preds_denorm = preds * label_std + label_mean
        return np.clip(preds_denorm, HyperParams.XCO2_MIN, HyperParams.XCO2_MAX)


class XCO2Dataset(Dataset):
    def __init__(self, inputs, st_features, labels, landuse_types):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.st_features = torch.tensor(st_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.landuse_types = landuse_types
        print(f"数据集初始化 - 样本数量: {len(self)}, 空间特征形状: {self.inputs.shape[1:]}, 时空特征维度: {self.st_features.shape[1]}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.st_features[idx], self.labels[idx], self.landuse_types[idx]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.attn_dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class XCO2HybridModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        spatial_vars = HyperParams.AUXILIARY_VARS + [f[3] for f in HyperParams.DERIVED_VARS]
        n_spatial_vars = len(spatial_vars)
        n_st_vars = len(HyperParams.GENERATED_ST_FEATURES)
        print(f"模型初始化 - 空间特征数量: {n_spatial_vars}, 时空特征数量: {n_st_vars}")

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(n_spatial_vars, HyperParams.CONV1_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(HyperParams.CONV1_CHANNELS),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(HyperParams.CONV1_CHANNELS, HyperParams.CONV2_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(HyperParams.CONV2_CHANNELS),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=1, stride=1),

            nn.Conv2d(HyperParams.CONV2_CHANNELS, HyperParams.CONV3_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(HyperParams.CONV3_CHANNELS),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(HyperParams.CONV3_CHANNELS, HyperParams.CONV4_CHANNELS, kernel_size=1, padding=0),
            nn.BatchNorm2d(HyperParams.CONV4_CHANNELS),
            nn.GELU()
        )

        self.transformer_input_dim = HyperParams.CONV4_CHANNELS
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(
                embed_dim=self.transformer_input_dim,
                num_heads=HyperParams.TRANSFORMER_HEADS,
                dropout=dropout_rate
            ) for _ in range(HyperParams.TRANSFORMER_LAYERS)]
        )

        self.st_branch = nn.Sequential(
            nn.Linear(n_st_vars, HyperParams.ST_BRANCH_HIDDEN_DIM),
            nn.GELU(),
            nn.BatchNorm1d(HyperParams.ST_BRANCH_HIDDEN_DIM),
            nn.Dropout(dropout_rate),

            nn.Linear(HyperParams.ST_BRANCH_HIDDEN_DIM, HyperParams.ST_BRANCH_HIDDEN_DIM * 2),
            nn.GELU(),
            nn.BatchNorm1d(HyperParams.ST_BRANCH_HIDDEN_DIM * 2),
            nn.Dropout(dropout_rate),

            nn.Linear(HyperParams.ST_BRANCH_HIDDEN_DIM * 2, HyperParams.ST_BRANCH_OUTPUT_DIM),
            nn.GELU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(self.transformer_input_dim, HyperParams.FC1_UNITS),
            nn.GELU(),
            nn.BatchNorm1d(HyperParams.FC1_UNITS),
            nn.Dropout(dropout_rate),

            nn.Linear(HyperParams.FC1_UNITS, HyperParams.FC2_UNITS),
            nn.GELU(),
            nn.BatchNorm1d(HyperParams.FC2_UNITS),
            nn.Dropout(dropout_rate),

            nn.Linear(HyperParams.FC2_UNITS, 1)
        )

        self.dropout_rate = dropout_rate

    def forward(self, x, st_feat):
        cnn_out = self.cnn_layers(x)
        transformer_in = cnn_out.view(cnn_out.size(0), -1, self.transformer_input_dim)
        transformer_out = self.transformer_blocks(transformer_in)
        spatial_feat = transformer_out.mean(dim=1)

        st_processed = self.st_branch(st_feat)

        fused_feat = spatial_feat + st_processed

        out = self.fc_layers(fused_feat)
        return out

    def update_dropout(self, epoch, total_epochs):
        new_dropout = HyperParams.DROPOUT_INIT + (HyperParams.DROPOUT_FINAL - HyperParams.DROPOUT_INIT) * (
                epoch / total_epochs)
        self.dropout_rate = new_dropout

        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_dropout

        for block in self.transformer_blocks:
            block.attn_dropout.p = new_dropout
            block.dropout.p = new_dropout
            block.attention.dropout = new_dropout


class Trainer:
    @staticmethod
    def spatial_smooth_loss(preds):
        preds_spatial = preds.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, HyperParams.NEIGHBORHOOD_SIZE, HyperParams.NEIGHBORHOOD_SIZE
        )
        horizontal_diff = torch.abs(preds_spatial[:, :, :, 1:] - preds_spatial[:, :, :, :-1])
        vertical_diff = torch.abs(preds_spatial[:, :, 1:, :] - preds_spatial[:, :, :-1, :])
        return (horizontal_diff.mean() + vertical_diff.mean()) / 2

    @staticmethod
    def gradient_penalty_loss(inputs, st_feat, model):
        inputs.requires_grad_(True)
        st_feat.requires_grad_(True)
        preds = model(inputs, st_feat)
        grad = torch.autograd.grad(
            outputs=preds,
            inputs=[inputs, st_feat],
            grad_outputs=torch.ones_like(preds),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        grad_norm = torch.norm(torch.cat([g.view(g.size(0), -1) for g in grad], dim=1), p=2, dim=1)
        return torch.mean((grad_norm - 1) ** 2)

    @staticmethod
    def combined_loss(preds, labels, inputs, st_feat, model, is_train=True):
        mse_loss = nn.MSELoss()(preds, labels)
        spatial_loss = Trainer.spatial_smooth_loss(preds)
        grad_loss = Trainer.gradient_penalty_loss(inputs, st_feat, model) if is_train else 0.0

        total_loss = (
                mse_loss
                + HyperParams.SPATIAL_LOSS_WEIGHT * spatial_loss
                + (HyperParams.GRAD_PENALTY_WEIGHT * grad_loss if is_train else 0.0)
        )

        return total_loss, mse_loss, spatial_loss

    @staticmethod
    def evaluate(model, dataloader, stats, device):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_lus = []

        with torch.no_grad():
            for inputs, st_feat, labels, lus in dataloader:
                inputs, st_feat, labels = inputs.to(device), st_feat.to(device), labels.to(device)
                preds = model(inputs, st_feat)

                loss, _, _ = Trainer.combined_loss(preds, labels, inputs, st_feat, model, is_train=False)
                total_loss += loss.item() * inputs.size(0)

                preds_denorm = DataProcessor.denormalize_label(preds)
                labels_denorm = DataProcessor.denormalize_label(labels)

                all_preds.extend(preds_denorm)
                all_labels.extend(labels_denorm)
                all_lus.extend(lus.numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        avg_mse = mean_squared_error(np.array(all_labels), np.array(all_preds))
        rmse = np.sqrt(avg_mse)
        r2 = r2_score(np.array(all_labels).flatten(), np.array(all_preds).flatten())

        lu_metrics = {}
        for lu in HyperParams.LANDUSE_TARGETS:
            lu_mask = np.array(all_lus) == lu
            if np.sum(lu_mask) == 0:
                lu_metrics[lu] = {"RMSE": None, "R²": None, "样本数": 0}
                continue
            lu_preds = np.array(all_preds)[lu_mask]
            lu_labels = np.array(all_labels)[lu_mask]
            lu_rmse = np.sqrt(mean_squared_error(lu_labels, lu_preds))
            lu_r2 = r2_score(lu_labels.flatten(), lu_preds.flatten())
            lu_metrics[lu] = {
                "RMSE": lu_rmse,
                "R²": lu_r2,
                "样本数": np.sum(lu_mask)
            }

        return avg_loss, avg_mse, rmse, r2, lu_metrics

    @staticmethod
    def train_model(train_loader, val_loader, train_stats):
        model = XCO2HybridModel(dropout_rate=HyperParams.DROPOUT_INIT).to(HyperParams.DEVICE)
        print(f"模型已加载到设备: {HyperParams.DEVICE}")

        optimizer = optim.Adam([
            {"params": model.cnn_layers.parameters(), "lr": HyperParams.LR_CNN},
            {"params": model.transformer_blocks.parameters(), "lr": HyperParams.LR_TRANSFORMER},
            {"params": model.st_branch.parameters(), "lr": HyperParams.LR_ST_BRANCH},
            {"params": model.fc_layers.parameters(), "lr": HyperParams.LR_FC}
        ], betas=HyperParams.ADAM_BETAS, eps=HyperParams.ADAM_EPS, weight_decay=HyperParams.WEIGHT_DECAY)

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=5,
            min_lr=[1e-8, 1e-8, 1e-8, 1e-8]
        )

        best_rmse = float('inf')
        best_r2 = -float('inf')
        best_epoch = 0
        no_improve_epochs = 0

        best_model_path = os.path.join(HyperParams.MODEL_SAVE_DIR, "best_model_with_spatiotemporal.pth")
        print(f"最优模型将保存至: {best_model_path}")

        print(f"\n开始训练 ({HyperParams.EPOCHS}轮):")
        print("=" * 220)
        print(
            f"轮次 | 训练损失 | 训练MSE | 训练RMSE | 训练R² | 验证损失 | 验证MSE | 验证RMSE(ppm) | 验证R² | Dropout | 学习率(CNN/Transformer/时空/FC)"
        )
        print("-" * 220)

        for epoch in range(1, HyperParams.EPOCHS + 1):
            model.update_dropout(epoch, HyperParams.EPOCHS)

            model.train()
            train_total_loss = 0.0
            train_preds = []
            train_labels = []

            for inputs, st_feat, labels, _ in tqdm(train_loader, desc=f"训练轮次 {epoch}/{HyperParams.EPOCHS}",
                                                   leave=False):
                inputs, st_feat, labels = inputs.to(HyperParams.DEVICE), st_feat.to(HyperParams.DEVICE), labels.to(
                    HyperParams.DEVICE)

                optimizer.zero_grad()
                preds = model(inputs, st_feat)
                loss, mse_loss, _ = Trainer.combined_loss(preds, labels, inputs, st_feat, model, is_train=True)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_total_loss += loss.item() * inputs.size(0)

                train_preds.extend(DataProcessor.denormalize_label(preds))
                train_labels.extend(DataProcessor.denormalize_label(labels))

            train_avg_loss = train_total_loss / len(train_loader.dataset)
            train_avg_mse = mean_squared_error(np.array(train_labels), np.array(train_preds))
            train_rmse = np.sqrt(train_avg_mse)
            train_r2 = r2_score(np.array(train_labels).flatten(), np.array(train_preds).flatten())

            val_loss, val_mse, val_rmse, val_r2, lu_metrics = Trainer.evaluate(model, val_loader, train_stats,
                                                                               HyperParams.DEVICE)

            lr_cnn = optimizer.param_groups[0]['lr']
            lr_transformer = optimizer.param_groups[1]['lr']
            lr_st = optimizer.param_groups[2]['lr']
            lr_fc = optimizer.param_groups[3]['lr']

            print(
                f"{epoch:4d} | {train_avg_loss:.4f} | {train_avg_mse:.4f} | {train_rmse:.4f} | {train_r2:.4f} | "
                f"{val_loss:.4f} | {val_mse:.4f} | {val_rmse:.4f} | {val_r2:.4f} | "
                f"{model.dropout_rate:.2f} | {lr_cnn:.6f}/{lr_transformer:.6f}/{lr_st:.6f}/{lr_fc:.6f}"
            )

            print("  按landuse验证结果: ", end="")
            for lu in HyperParams.LANDUSE_TARGETS:
                if lu_metrics[lu]["样本数"] > 0:
                    print(
                        f"landuse={lu}: RMSE={lu_metrics[lu]['RMSE']:.3f}, R²={lu_metrics[lu]['R²']:.3f}({lu_metrics[lu]['样本数']}个样本); ",
                        end="")
                else:
                    print(f"landuse={lu}: 无样本; ", end="")
            print()

            if (val_rmse < best_rmse - HyperParams.EARLY_STOP_MIN_DELTA) or (val_r2 > best_r2 + 0.01):
                best_rmse = val_rmse
                best_r2 = val_r2
                best_epoch = epoch
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  保存新的最优模型 (验证RMSE: {best_rmse:.4f} ppm, R²: {best_r2:.4f})")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= HyperParams.EARLY_STOP_PATIENCE:
                    print(f"  早停触发: 连续{no_improve_epochs}轮无显著改进")
                    break

            scheduler.step(val_rmse)

        print("=" * 220)
        print(f"训练完成! 最优模型(第{best_epoch}轮)已保存至: {best_model_path}")
        print(f"最优验证RMSE: {best_rmse:.4f} ppm, 最优验证R²: {best_r2:.4f}")


def main():
    print("===== XCO2混合模型训练程序（含分层抽样） =====")

    if torch.cuda.is_available():
        print("检测到CUDA设备，正在预热CUDA上下文...")
        try:
            torch.zeros(1).cuda()
            print("CUDA上下文初始化完成")
        except Exception as e:
            print(f"CUDA初始化警告: {str(e)}")
    else:
        print("未检测到CUDA设备，使用CPU训练")

    warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

    data_processor = DataProcessor()
    data_processor.create_directories()

    time_to_data, total_valid, _, landuse_stats = data_processor.load_landuse_data()

    print(f"\n开始按时间片+Landuse分层抽样（目标抽样率：{HyperParams.TARGET_SAMPLING_RATE:.1%}）...")
    time_to_data_sampled, sampling_stats = DataProcessor.stratified_sample_by_time_landuse(
        time_to_data=time_to_data,
        random_seed=HyperParams.RANDOM_SEED
    )

    total_valid_sampled = sampling_stats["sampled_total"]
    landuse_stats_sampled = defaultdict(int)
    for samples in time_to_data_sampled.values():
        for sample in samples:
            landuse_stats_sampled[sample["landuse_type"]] += 1
    print(f"抽样后各Landuse样本分布: {dict(landuse_stats_sampled)}")

    train_data, val_data = data_processor.split_train_val(
        time_to_data=time_to_data_sampled,
        total_valid=total_valid_sampled,
        landuse_stats=landuse_stats_sampled
    )

    train_inputs, train_st, train_labels, train_stats, train_lus = data_processor.normalize_data(train_data,
                                                                                                 is_train=True)
    val_inputs, val_st, val_labels, val_lus = data_processor.normalize_data(val_data, stats=train_stats, is_train=False)

    train_dataset = XCO2Dataset(train_inputs, train_st, train_labels, train_lus)
    val_dataset = XCO2Dataset(val_inputs, val_st, val_labels, val_lus)

    train_loader = DataLoader(
        train_dataset,
        batch_size=HyperParams.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(HyperParams.DEVICE.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=HyperParams.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(HyperParams.DEVICE.type == "cuda")
    )
    print(f"数据加载器创建完成 - 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    Trainer.train_model(train_loader, val_loader, train_stats)


if __name__ == "__main__":
    main()