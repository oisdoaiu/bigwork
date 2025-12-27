import torch
import pickle

def pt_to_pkl(pt_file_path, pkl_file_path):
    """
    将.pt文件转换为.pkl文件
    参数:
        pt_file_path: .pt文件的路径
        pkl_file_path: 输出的.pkl文件路径
    """
    try:
        # 加载.pt文件，设置weights_only=False
        data = torch.load(pt_file_path, weights_only=False)        
        # 将数据保存为.pkl文件
        if "model" in data:
            del data["model"]
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(data, f)
        print(data)
        print(f"成功将 {pt_file_path} 转换为 {pkl_file_path}")
        
    except Exception as e:
        print(f"转换失败: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的.pt文件路径和要保存的.pkl文件路径
    pt_file = "yolov5su.pt"
    pkl_file = "yolov5su.pkl"
    
    pt_to_pkl(pt_file, pkl_file)