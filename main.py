import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

# --------------------
# 数据处理
# --------------------

# 请根据您的实际数据文件路径和格式进行调整
# 以下是示例路径和文件名，请确保这些文件存在并且格式正确

# 节点特征文件路径
node_features_path = r"C:\Users\user0\Desktop\labels\feature重排列_processed1.xlsx"
# 邻接矩阵文件路径
adj_matrix_path = r"C:\Users\user0\Desktop\labels\daona(nomalization).xlsx"
# 阻抗矩阵文件路径
impedance_matrix_path = r"C:\Users\user0\Desktop\labels\impedance_matrix.xlsx"
# 标签文件路径
labels_path = r"C:\Users\user0\Desktop\labels\20labels0.06.xlsx"

# 检查文件是否存在
for file_path in [node_features_path, adj_matrix_path, impedance_matrix_path, labels_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到：{file_path}，请检查文件路径和名称。")

# 读取节点特征和邻接矩阵
node_features = pd.read_excel(node_features_path, header=None).values  # [样本数, 节点数 * 特征数]
adj_matrix = pd.read_excel(adj_matrix_path, header=None).values  # [节点数, 节点数]
impedance_matrix = pd.read_excel(impedance_matrix_path, header=None).values  # [节点数, 节点数]
labels = pd.read_excel(labels_path, header=None).values  # [样本数, 标签数]

# 获取样本数和节点数
num_samples = node_features.shape[0]
num_nodes = adj_matrix.shape[0]
num_node_features = int(node_features.shape[1] / num_nodes)
num_labels = labels.shape[1]

# 将节点特征重塑为 [样本数, 节点数, 特征数]
node_features = node_features.reshape(num_samples, num_nodes, num_node_features)

# 标签标准化处理
labels_normalized = 1 / (1 + np.exp(-4 * labels))
labels_normalized = torch.tensor(labels_normalized, dtype=torch.float32)


# 反归一化函数
def invert_normalization(normalized_values, scale=4, epsilon=1e-15):
    normalized_values = np.clip(normalized_values, epsilon, 1 - epsilon)
    result = -np.log(1 / normalized_values - 1) / scale
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


# 设置一个小的正数，避免阻抗为零
epsilon = 1e-8

# 处理阻抗矩阵
impedance_matrix_processed = impedance_matrix.copy()
impedance_matrix_processed[impedance_matrix_processed == 0] = epsilon
max_impedance = 1e6
impedance_matrix_processed[adj_matrix == 0] = max_impedance

# 对阻抗矩阵进行归一化处理
finite_impedances = impedance_matrix_processed[np.isfinite(impedance_matrix_processed)]
min_impedance = finite_impedances.min()
max_impedance = finite_impedances.max()
impedance_matrix_normalized = (impedance_matrix_processed - min_impedance) / (max_impedance - min_impedance + epsilon)


# 构建图数据
def create_graph_data(node_features_sample, adj_matrix, impedance_matrix, label_sample, impedance_threshold=0.1):
    data = Data()
    data.x = torch.tensor(node_features_sample, dtype=torch.float32)  # [节点数, 特征数]

    edge_index_local = []
    edge_index_global = []
    edge_attr_local = []
    edge_attr_global = []

    num_nodes = adj_matrix.shape[0]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] != 0 and i != j:
                impedance = impedance_matrix[i, j]
                # 添加局部边（直接邻居）
                edge_index_local.append([i, j])
                edge_attr_local.append([impedance])
                # 判断是否为高阻抗边
                if impedance >= impedance_threshold:
                    # 添加为全局边
                    edge_index_global.append([i, j])
                    edge_attr_global.append([impedance])

    # 如果没有全局边，避免张量为空
    if len(edge_index_global) == 0:
        edge_index_global = torch.empty((2, 0), dtype=torch.long)
        edge_attr_global = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index_global = torch.tensor(edge_index_global, dtype=torch.long).t().contiguous()  # [2, E_global]
        edge_attr_global = torch.tensor(edge_attr_global, dtype=torch.float32)  # [E_global, 1]

    # 如果没有局部边，避免张量为空
    if len(edge_index_local) == 0:
        edge_index_local = torch.empty((2, 0), dtype=torch.long)
        edge_attr_local = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index_local = torch.tensor(edge_index_local, dtype=torch.long).t().contiguous()  # [2, E_local]
        edge_attr_local = torch.tensor(edge_attr_local, dtype=torch.float32)  # [E_local, 1]

    data.edge_index_local = edge_index_local
    data.edge_attr_local = edge_attr_local

    data.edge_index_global = edge_index_global
    data.edge_attr_global = edge_attr_global

    data.y = torch.tensor(label_sample, dtype=torch.float32).reshape(-1, num_labels)  # [1, 标签数]

    return data


# 创建图列表
graphs = []
for i in range(num_samples):
    graph = create_graph_data(
        node_features_sample=node_features[i],
        adj_matrix=adj_matrix,
        impedance_matrix=impedance_matrix_normalized,
        label_sample=labels_normalized[i],
        impedance_threshold=0.1 
    )
    graphs.append(graph)

# 划分数据集 (80%训练集, 10%验证集, 10%测试集)
train_graphs, temp_graphs = train_test_split(graphs, test_size=0.20, random_state=42)
val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.50, random_state=42)

# 创建数据加载器
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)


# %%
# --------------------
# 模型定义
# --------------------
class GlobalAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GlobalAttentionLayer, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)  # 将输入映射到 out_channels 维度
        self.att = nn.Linear(out_channels * 2 + 1, 1)  # 注意力权重层
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        x_transformed = self.linear(x)  # [N, out_channels]
        propagated_x = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr)
        return propagated_x

    def message(self, x_i, x_j, edge_attr, index):
        alpha_input = torch.cat([x_i, x_j, edge_attr], dim=1)  # [E, 2 * out_channels + 1]
        alpha = self.leaky_relu(self.att(alpha_input))  # [E, 1]
        alpha = softmax(alpha, index)  # [E, 1]
        return x_j * alpha


class CombinedModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(CombinedModel, self).__init__()

        # 局部GAT层
        self.local_gat1 = GATConv(in_channels, hidden_channels, heads=6, concat=True, dropout=0.5)
        self.local_gat2 = GATConv(hidden_channels * 6, hidden_channels, heads=6, concat=False, dropout=0.5)

        # 全局注意力层
        self.global_att1 = GlobalAttentionLayer(in_channels, hidden_channels // 2)
        self.global_att2 = GlobalAttentionLayer(hidden_channels // 2, hidden_channels)
        self.global_att3 = GlobalAttentionLayer(hidden_channels, hidden_channels * 2)
        self.global_att4 = GlobalAttentionLayer(hidden_channels * 2, hidden_channels)
        # 添加线性变换层，当没有边时使用
        self.local_transform = nn.Linear(in_channels, hidden_channels)
        self.global_transform = nn.Linear(in_channels, hidden_channels)
        # 全连接层，增加更多的层
        self.fc1 = nn.Linear(hidden_channels * 2, 128)  # 输入为 hidden_channels * 2，输出为 128
        self.fc2 = nn.Linear(128, 64)  # 新增全连接层，从 128 到 64
        self.fc3 = nn.Linear(64, 32)  # 新增全连接层，从 64 到 32
        self.fc4 = nn.Linear(32, out_channels)  # 最后的输出层，输出为目标标签的维度
        self.dropout = nn.Dropout(0.5)  # 增加 dropout 防止过拟合
        self.elu = nn.ELU()  # 使用 ELU 激活函数

        # 定义可学习的局部和全局权重
        self.local_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.global_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, data):
        x = data.x  # [N, in_channels]

        # 局部GAT
        if data.edge_index_local.numel() > 0:
            x_local = F.elu(self.local_gat1(x, data.edge_index_local))
            x_local = F.elu(self.local_gat2(x_local, data.edge_index_local))
        else:
            x_local = F.elu(self.local_transform(data.x))

        # 全局注意力
        if data.edge_index_global.numel() > 0:
            x_global = F.elu(self.global_att1(x, data.edge_index_global, data.edge_attr_global))
            x_global = F.elu(self.global_att2(x_global, data.edge_index_global, data.edge_attr_global))
            x_global = F.elu(self.global_att3(x_global, data.edge_index_global, data.edge_attr_global))
            x_global = F.elu(self.global_att4(x_global, data.edge_index_global, data.edge_attr_global))

        else:
            x_global = F.elu(self.global_transform(data.x))

        # 拼接局部和全局特征，并且可学习权重应用于拼接的不同部分
        x_combined = torch.cat([self.local_weight * x_local, self.global_weight * x_global], dim=1)

        # 图级别的表示
        x_pooled = global_mean_pool(x_combined, data.batch)

        # 全连接层，加入 ELU 激活函数和 Dropout 层
        x_fc = self.elu(self.fc1(x_pooled))
        x_fc = self.dropout(x_fc)
        x_fc = self.elu(self.fc2(x_fc))
        x_fc = self.dropout(x_fc)
        x_fc = self.elu(self.fc3(x_fc))
        out = self.fc4(x_fc)

        return out


# --------------------
# 模型训练
# --------------------



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = CombinedModel(in_channels=num_node_features, hidden_channels=128, out_channels=num_labels).to(device)


optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)

# 设置学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

# 设置损失函数
criterion = nn.MSELoss()


# 训练循环
num_epochs = 70
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for data in train_loader:

        data = data.to(device)

        # 模型前向传播，计算损失并反向传播
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 打印训练损失
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss:.4f}")


    # 更新学习率调度器
    scheduler.step(total_loss)

    # 验证模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.to(device))
            val_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}")

# 测试模型
model.eval()
test_loss = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.to(device))
        test_loss += loss.item()

print(f"Test Loss: {test_loss:.4f}")


# 评估指标计算
def mean_arctangent_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true + 1e-15)))) * 100


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape = np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100
    return smape


# 评估指标计算
model.eval()
labels_test_numpy = []
predictions_numpy = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        predictions = model(data)
        labels_test_numpy.append(data.y.cpu().numpy())
        predictions_numpy.append(predictions.cpu().numpy())

# 转换为 NumPy 数组
labels_test_numpy = np.vstack(labels_test_numpy)
predictions_numpy = np.vstack(predictions_numpy)

# 反归一化预测值和标签
predictions_original_scale = invert_normalization(predictions_numpy)
labels_test_original_scale = invert_normalization(labels_test_numpy)

# 计算评估指标
test_rmse_original = root_mean_squared_error(labels_test_original_scale, predictions_original_scale)
test_smape_original = symmetric_mean_absolute_percentage_error(labels_test_original_scale, predictions_original_scale)

print(
    f'Test RMSE (Original Scale): {test_rmse_original:.5f}, Test SMAPE (Original Scale): {test_smape_original:.2f}%')
