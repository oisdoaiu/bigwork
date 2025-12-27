# 环境要求：仅安装 Jittor + nkyolo（无 PyTorch）
import pickle

# 1. 加载 PyTorch 环境导出的非 model 参数
with open("non_model_params.pkl", "rb") as f:
    non_model_params = pickle.load(f)

# 2. 准备 tmpmodel 参数（若已导出为文件，需加载；若未导出，直接使用步骤2的 model_param）
# 若未导出，直接使用：
# model_param = {"model": tmpmodel}
# 若已导出为 tmpmodel_param.pkl，加载：
with open("tmpmodel_param.pkl", "rb") as f:
    model_param = pickle.load(f)

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