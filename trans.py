# 环境要求：仅安装 Jittor + nkyolo（无 PyTorch）
import jittor as jt
import numpy as np
from nkyolo.models.yolo.model import YOLO

# 1. 创建 nkyolo 的 DetectionModel 实例（需与原模型结构完全一致！）
# 关键：nc（类别数）、网络结构（backbone/neck/head）必须和 ultralytics 模型一致
nk_model = YOLO("yolov5s.yaml", task="detect", verbose=True)
tmpmodel = nk_model.model

# 2. 加载步骤1导出的权重（二选一即可）
# 方式2：加载 .npz 格式权重（无 PyTorch 依赖，推荐）
npz_dict = np.load("model_weights.npz")
for name, param in tmpmodel.named_parameters():
    if name in npz_dict:
        # 直接加载 numpy 数组转为 Jittor 张量
        weight_np = npz_dict[name]
        jt_weight = jt.array(weight_np)
        param.assign(jt_weight)
    else:
        print(f"警告：权重 {name} 未在 nkyolo 模型中找到")

print("权重加载成功，nkyolo 模型创建完成")

# 3. 将 nkyolo 模型保存为 pkl 文件（实现最终转换目标）
import pickle

# 注意：Jittor 模型保存为 pkl 时，建议先切换到 eval 模式
tmpmodel.eval()

# 1. 加载 PyTorch 环境导出的非 model 参数
with open("non_model_params.pkl", "rb") as f:
    non_model_params = pickle.load(f)

# 2. 准备 tmpmodel 参数（若已导出为文件，需加载；若未导出，直接使用步骤2的 model_param）
# 若未导出，直接使用：
model_param = {"model": tmpmodel}

# 3. 合并为总字典（核心：将 model_param 和 non_model_params 合并）
total_dict = {}
total_dict.update(model_param)  # 加入 tmpmodel 相关参数（键：model 或 model_state_dict）
total_dict.update(non_model_params)  # 加入非 model 参数（覆盖同名键，若有冲突需调整）

# 4. 验证合并结果（可选）
print("合并后总字典的所有键：", total_dict.keys())
print("模型参数类型：", type(total_dict["model"]))  # 输出：<class 'nkyolo.YOLO'>
print("非 model 参数示例：", {k: v for k, v in non_model_params.items() if k != list(non_model_params.keys())[0]})

# 5. Pickle 保存为 .pkl 文件
with open("final_merged_model.pkl", "wb") as f:
    pickle.dump(total_dict, f)

print("合并完成！最终文件：final_merged_model.pkl")

# 保存模型实例到 pkl 文件
with open("nkyolo_model.pkl", "wb") as f:
    pickle.dump(tmpmodel, f)