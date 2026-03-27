import os
import numpy as np
import xarray as xr
import pickle
import glob
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')

class HyperParams:
    MODEL_SAVE_DIR = "model_training/saved_models"
    STAT_SAVE_PATH = "model_training/data_stats.npy"
    INPUT_DATA_DIR = "model_input_0.1deg"
    GRID_MAPPING_DIR = "grid_mapping"
    OUTPUT_NC_DIR = "model_output_nc"
    VALID_CACHE_PATH = "cache/u10_valid_indices.pkl"
    DEBUG_LOG_PATH = os.path.join(OUTPUT_NC_DIR, "inference_debug.log")

    SPATIAL_FEATURE_DIM = 17
    NEIGHBORHOOD_SIZE = 7
    SPATIAL_INPUT_SHAPE = (SPATIAL_FEATURE_DIM, NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE)

    GENERATED_ST_FEATURES = [
        "time_ordinal_norm", "season_sin", "season_cos", "lat_norm", "lon_norm"
    ]
    ST_BRANCH_HIDDEN_DIM = 256
    ST_BRANCH_OUTPUT_DIM = 512
    ST_FEATURE_DIM = len(GENERATED_ST_FEATURES)

    # 推理参数
    XCO2_MIN = 380.0
    XCO2_MAX = 440.0
    BATCH_INFER_SIZE = 2048
    DROPOUT_RATE = 0.2

    CONV1_CHANNELS = 96
    CONV2_CHANNELS = 192
    CONV3_CHANNELS = 384
    CONV4_CHANNELS = 512
    TRANSFORMER_HEADS = 8
    TRANSFORMER_LAYERS = 4
    FC1_UNITS = 256
    FC2_UNITS = 128


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
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
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.attn_dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.ffn_dropout(ffn_out))
        return x


class XCO2HybridModel(nn.Module):

    def __init__(self, dropout_rate=HyperParams.DROPOUT_RATE):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(HyperParams.SPATIAL_FEATURE_DIM, HyperParams.CONV1_CHANNELS, kernel_size=3, padding=1),
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
            nn.Linear(HyperParams.ST_FEATURE_DIM, HyperParams.ST_BRANCH_HIDDEN_DIM),
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

    def forward(self, x: torch.Tensor, st_feat: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn_layers(x)
        transformer_in = cnn_out.view(cnn_out.size(0), -1, self.transformer_input_dim)
        transformer_out = self.transformer_blocks(transformer_in)
        spatial_feat = transformer_out.mean(dim=1)

        st_processed = self.st_branch(st_feat)

        fused_feat = spatial_feat + st_processed

        out = self.fc_layers(fused_feat)
        return out


class DataProcessor:
    @staticmethod
    def init_log():
        os.makedirs(HyperParams.OUTPUT_NC_DIR, exist_ok=True)
        with open(HyperParams.DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(f"XCO2推理调试日志 - {datetime.now()}\n")
            f.write(f"设备: {HyperParams.DEVICE}\n\n")

    @staticmethod
    def log(message: str):
        with open(HyperParams.DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {message}\n")
        print(message)

    @staticmethod
    def load_stats() -> dict:
        DataProcessor.log(f"加载统计文件: {HyperParams.STAT_SAVE_PATH}")
        if not os.path.exists(HyperParams.STAT_SAVE_PATH):
            raise FileNotFoundError(f"统计文件缺失：{HyperParams.STAT_SAVE_PATH}")

        stats = np.load(HyperParams.STAT_SAVE_PATH, allow_pickle=True).item()
        if "normalization" not in stats:
            raise ValueError("统计文件格式错误：缺少'normalization'字段")

        norm_data = stats["normalization"]

        required_label_keys = ["label_mean", "label_std"]
        for key in required_label_keys:
            if key not in norm_data:
                raise ValueError(f"统计文件缺少关键标签参数: {key}")

        DataProcessor.log(f"标签归一化参数:")
        DataProcessor.log(f"  label_mean: {norm_data['label_mean']}")
        DataProcessor.log(f"  label_std: {norm_data['label_std']}")

        if np.isclose(norm_data["label_std"], 0, atol=1e-6):
            DataProcessor.log(f"警告：label_std接近0，可能导致反归一化错误")

        return norm_data

    @staticmethod
    def validate_feature_distribution(features: np.ndarray, feature_name: str):
        mean = features.mean()
        std = features.std()
        DataProcessor.log(f"{feature_name}特征分布 - 均值: {mean:.4f}, 标准差: {std:.4f}")

        if abs(mean) > 3:
            DataProcessor.log(f"警告：{feature_name}特征均值偏离0过大，可能未正确标准化")

        if std < 0.1 or std > 10:
            DataProcessor.log(f"警告：{feature_name}特征标准差异常，可能未正确标准化")

    @staticmethod
    def denormalize_xco2(preds: np.ndarray, stats: dict) -> np.ndarray:
        label_mean = stats["label_mean"]
        label_std = stats["label_std"]

        DataProcessor.log(f"模型原始输出 - 最小值: {preds.min():.4f}, 最大值: {preds.max():.4f}, 均值: {preds.mean():.4f}")

        preds_denorm = preds * label_std + label_mean

        DataProcessor.log(
            f"反归一化后 - 最小值: {preds_denorm.min():.4f}, 最大值: {preds_denorm.max():.4f}, 均值: {preds_denorm.mean():.4f}")

        clipped_low = np.sum(preds_denorm < HyperParams.XCO2_MIN)
        clipped_high = np.sum(preds_denorm > HyperParams.XCO2_MAX)
        total = preds_denorm.size
        DataProcessor.log(
            f"裁剪统计 - 低于{HyperParams.XCO2_MIN}: {clipped_low}个({clipped_low / total * 100:.2f}%), 高于{HyperParams.XCO2_MAX}: {clipped_high}个({clipped_high / total * 100:.2f}%)")

        return np.clip(preds_denorm, HyperParams.XCO2_MIN, HyperParams.XCO2_MAX)


def load_trained_model(model_path: str) -> Tuple[XCO2HybridModel, torch.device]:
    DataProcessor.log(f"加载模型: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件缺失：{model_path}")

    device = HyperParams.DEVICE
    model = XCO2HybridModel()

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        DataProcessor.log("模型权重加载成功")

        first_params = next(model.parameters()).data[:5]
        DataProcessor.log(f"模型前5个参数示例: {first_params.cpu().numpy()}")
    except Exception as e:
        DataProcessor.log(f"模型加载错误: {str(e)}")
        raise

    model.to(device)
    model.eval()
    DataProcessor.log(f"模型已部署到设备: {device}")
    return model, device


def get_grid_info() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    DataProcessor.log("获取网格信息...")

    valid_lat_idx, valid_lon_idx = np.array([]), np.array([])
    if os.path.exists(HyperParams.VALID_CACHE_PATH):
        with open(HyperParams.VALID_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        valid_lat_idx = cache["valid_lat_idx"]
        valid_lon_idx = cache["valid_lon_idx"]
        DataProcessor.log(f"加载有效点缓存: 原始数量={len(valid_lat_idx)}个")

        nc_files = glob.glob(os.path.join("merged_data", "*_merged.nc"))
        if not nc_files:
            raise FileNotFoundError("原始NC文件缺失：merged_data目录")
        with xr.open_dataset(nc_files[0]) as ds:
            lat_len = len(ds["lat"].values)
            lon_len = len(ds["lon"].values)

        valid_lat_mask = (valid_lat_idx >= 0) & (valid_lat_idx < lat_len)
        valid_lon_mask = (valid_lon_idx >= 0) & (valid_lon_idx < lon_len)
        total_valid_mask = valid_lat_mask & valid_lon_mask

        filtered_lat_idx = valid_lat_idx[total_valid_mask]
        filtered_lon_idx = valid_lon_idx[total_valid_mask]
        invalid_count = len(valid_lat_idx) - len(filtered_lat_idx)

        if invalid_count > 0:
            DataProcessor.log(f"警告：检测到{invalid_count}个非法索引，已过滤")
            invalid_lat = valid_lat_idx[~total_valid_mask]
            invalid_lon = valid_lon_idx[~total_valid_mask]
            DataProcessor.log(f"非法索引示例（lat）: {invalid_lat[:5]}..." if len(invalid_lat) > 0 else "")
            DataProcessor.log(f"非法索引示例（lon）: {invalid_lon[:5]}..." if len(invalid_lon) > 0 else "")

        valid_lat_idx, valid_lon_idx = filtered_lat_idx, filtered_lon_idx
        DataProcessor.log(f"过滤后有效点数量: {len(valid_lat_idx)}个")

    nc_files = glob.glob(os.path.join("merged_data", "*_merged.nc"))
    if not nc_files:
        raise FileNotFoundError("原始NC文件缺失：merged_data目录")
    with xr.open_dataset(nc_files[0]) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        DataProcessor.log(f"经纬度网格 - 纬度: {len(lat)}个点（索引0~{len(lat) - 1}）, 经度: {len(lon)}个点（索引0~{len(lon) - 1}）")

    grid_mask = np.zeros((len(lat), len(lon)), dtype=bool)
    if len(valid_lat_idx) > 0:
        grid_mask[valid_lat_idx, valid_lon_idx] = True
        DataProcessor.log(f"有效点占比: {np.sum(grid_mask) / grid_mask.size * 100:.2f}%")

    return lat, lon, grid_mask, (valid_lat_idx, valid_lon_idx)


def batch_inference(model: XCO2HybridModel, spatial_inputs: np.ndarray, st_inputs: np.ndarray,
                    device: torch.device) -> np.ndarray:
    n_samples = spatial_inputs.shape[0]
    predictions = []

    DataProcessor.log(f"推理输入形状 - 空间特征: {spatial_inputs.shape}, 时空特征: {st_inputs.shape}")
    if spatial_inputs.shape[0] != st_inputs.shape[0]:
        raise ValueError(f"样本数量不匹配 - 空间特征: {spatial_inputs.shape[0]}, 时空特征: {st_inputs.shape[0]}")

    with tqdm(total=n_samples, desc="批量推理") as pbar:
        for i in range(0, n_samples, HyperParams.BATCH_INFER_SIZE):
            batch_end = min(i + HyperParams.BATCH_INFER_SIZE, n_samples)
            batch_spatial = torch.tensor(spatial_inputs[i:batch_end], dtype=torch.float32).to(device)
            batch_st = torch.tensor(st_inputs[i:batch_end], dtype=torch.float32).to(device)

            if i == 0:
                DataProcessor.log(f"第一个批次空间特征示例: {batch_spatial[0, 0, :2, :2].cpu().numpy()}")
                DataProcessor.log(f"第一个批次时空特征示例: {batch_st[0, :].cpu().numpy()}")

            with torch.no_grad():
                batch_pred = model(batch_spatial, batch_st)

                if i == 0:
                    DataProcessor.log(f"第一个批次模型输出示例: {batch_pred[:5].cpu().numpy().flatten()}")

            predictions.append(batch_pred.cpu().numpy())
            pbar.update(batch_end - i)

    all_preds = np.concatenate(predictions, axis=0)
    DataProcessor.log(f"所有样本模型输出 - 最小值: {all_preds.min():.4f}, 最大值: {all_preds.max():.4f}, 均值: {all_preds.mean():.4f}")

    return all_preds


def map_pred_to_grid(preds: np.ndarray, valid_indices: Tuple[np.ndarray, np.ndarray],
                     grid_shape: Tuple[int, int]) -> np.ndarray:
    valid_lat_idx, valid_lon_idx = valid_indices
    n_lat, n_lon = grid_shape

    pred_count = len(preds)
    valid_count = len(valid_lat_idx)

    if pred_count != valid_count:
        DataProcessor.log(f"警告：预测数与有效点数不匹配（{pred_count} vs {valid_count}），将忽略不匹配的点")
        min_count = min(pred_count, valid_count)
        valid_lat_idx = valid_lat_idx[:min_count]
        valid_lon_idx = valid_lon_idx[:min_count]
        preds = preds[:min_count]

    DataProcessor.log(f"映射预测到网格 - 预测数量: {len(preds)}, 有效点数量: {len(valid_lat_idx)}")

    grid_pred = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    grid_pred[valid_lat_idx, valid_lon_idx] = preds.squeeze()

    valid_preds = grid_pred[~np.isnan(grid_pred)]
    DataProcessor.log(f"网格映射后有效预测 - 数量: {len(valid_preds)}, 最小值: {valid_preds.min():.4f}, 最大值: {valid_preds.max():.4f}")

    return grid_pred


def main_inference():

    DataProcessor.init_log()
    DataProcessor.log("===== 开始XCO2模型推理 =====")

    try:

        stats = DataProcessor.load_stats()
        model, device = load_trained_model(
            os.path.join(HyperParams.MODEL_SAVE_DIR, "best_model_with_spatiotemporal.pth")
        )
        lat, lon, grid_mask, valid_indices = get_grid_info()
        grid_shape = (len(lat), len(lon))

        input_files = glob.glob(os.path.join(HyperParams.INPUT_DATA_DIR, "combined_input_*.npy"))
        if not input_files:
            raise FileNotFoundError(f"未找到预处理输入文件：{HyperParams.INPUT_DATA_DIR}")
        input_files.sort(key=lambda x: os.path.basename(x).split("_")[-1].split(".")[0])
        DataProcessor.log(f"找到{len(input_files)}个输入文件，开始处理...")

        for input_file in input_files:
            filename = os.path.basename(input_file)
            time_str = filename.split("_")[-1].split(".")[0]
            time_coord = np.datetime64(f"{time_str[:4]}-{time_str[4:]}-01")
            DataProcessor.log(f"\n===== 处理文件: {filename}（{time_str[:4]}年{time_str[4:]}月） =====")

            input_data = np.load(input_file, allow_pickle=True)
            DataProcessor.log(f"输入数据形状: {input_data.shape}")

            if input_data.shape[1] != 2:
                raise ValueError(f"输入格式错误：需空间+时空特征（实际shape：{input_data.shape}）")

            spatial_inputs = np.array([x for x in input_data[:, 0]], dtype=np.float32)
            st_inputs = np.array([x for x in input_data[:, 1]], dtype=np.float32)
            DataProcessor.log(f"特征形状 - 空间: {spatial_inputs.shape}, 时空: {st_inputs.shape}")

            DataProcessor.validate_feature_distribution(spatial_inputs, "空间")
            DataProcessor.validate_feature_distribution(st_inputs, "时空")

            DataProcessor.log(f"开始推理（设备：{device}）")
            preds = batch_inference(model, spatial_inputs, st_inputs, device)

            preds_denorm = DataProcessor.denormalize_xco2(preds, stats)

            debug_pred_path = os.path.join(HyperParams.OUTPUT_NC_DIR, f"debug_preds_{time_str}.npy")
            np.save(debug_pred_path, preds_denorm)
            DataProcessor.log(f"调试：原始反归一化结果已保存至 {debug_pred_path}")

            grid_pred = map_pred_to_grid(preds_denorm, valid_indices, grid_shape)

            nc_path = os.path.join(HyperParams.OUTPUT_NC_DIR, f"xco2_pred_{time_str}.nc")
            ds = xr.Dataset(
                data_vars={
                    "xco2_pred": (("time", "lat", "lon"), [grid_pred]),
                    "valid_mask": (("lat", "lon"), grid_mask.astype(int))
                },
                coords={"time": [time_coord], "lat": lat, "lon": lon},
                attrs={
                    "title": "XCO2 Prediction (CNN-Transformer with Spatiotemporal Features)",
                    "pred_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "unit": "ppm",
                    "xco2_range": f"{HyperParams.XCO2_MIN}~{HyperParams.XCO2_MAX}"
                }
            )
            ds.to_netcdf(nc_path, mode="w", format="NETCDF4",
                         encoding={"xco2_pred": {"zlib": True}, "valid_mask": {"zlib": True}})
            ds.close()
            DataProcessor.log(f"NC文件已保存：{os.path.abspath(nc_path)}")

        DataProcessor.log(f"\n===== 所有文件处理完成！输出目录：{os.path.abspath(HyperParams.OUTPUT_NC_DIR)} =====")

    except Exception as e:
        DataProcessor.log(f"执行失败：{str(e)}")
        raise


if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            torch.cuda.init()
            print(f"CUDA初始化：{torch.cuda.get_device_name(0)}")
        main_inference()
    except Exception as e:
        print(f"执行失败：{str(e)}")
        raise
