import pickle
import inspect

def find_ultralytics_deps(obj, parent_key="", deps_list=None):
    """
    递归遍历对象，查找所有依赖 ultralytics 的内容
    参数:
        obj: 要检查的对象（字典、列表、实例等）
        parent_key: 父级键名（用于定位依赖项的路径）
        deps_list: 存储依赖项的列表（递归传递）
    返回:
        deps_list: 所有 ultralytics 依赖项的路径和信息
    """
    if deps_list is None:
        deps_list = []
    
    # 场景1：对象是字典，遍历所有键值对
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            find_ultralytics_deps(value, current_key, deps_list)
    
    # 场景2：对象是列表/元组，遍历所有元素
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            current_key = f"{parent_key}[{idx}]" if parent_key else f"[{idx}]"
            find_ultralytics_deps(item, current_key, deps_list)
    
    # 场景3：对象是类实例，检查其类路径是否包含 ultralytics
    else:
        # 获取对象的类信息
        obj_type = type(obj)
        # 跳过基础数据类型（int/str/float等，无模块依赖）
        if obj_type in (int, str, float, bool, bytes, None.__class__):
            return deps_list
        
        # 获取类的模块路径（关键：判断是否包含 ultralytics）
        module_name = obj_type.__module__
        if "ultralytics" in module_name:
            # 收集依赖信息：路径、类型、模块
            dep_info = {
                "path": parent_key,
                "object_type": str(obj_type),
                "module": module_name,
                "object_repr": str(obj)[:100]  # 截取对象简要信息
            }
            deps_list.append(dep_info)
        
        # 额外检查：实例的 __dict__（对象的属性字典，可能嵌套 ultralytics 实例）
        if hasattr(obj, "__dict__"):
            find_ultralytics_deps(obj.__dict__, f"{parent_key}.__dict__", deps_list)
    
    return deps_list

# 核心执行代码
if __name__ == "__main__":
    pkl_file = "yolov5su.pkl"
    
    # 1. 在含 ultralytics 的环境中加载 pkl 文件
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        print(f"成功加载 {pkl_file}")
    except Exception as e:
        print(f"加载 pkl 失败：{e}")
        exit(1)
    
    # 2. 递归查找 ultralytics 依赖项
    ultralytics_deps = find_ultralytics_deps(data)
    
    # 3. 输出排查结果
    if not ultralytics_deps:
        print("✅ pkl 文件中未找到 ultralytics 依赖项")
    else:
        print(f"\n❌ 共找到 {len(ultralytics_deps)} 个 ultralytics 依赖项：")
        for idx, dep in enumerate(ultralytics_deps, 1):
            print(f"\n{idx}. 依赖路径：{dep['path']}")
            print(f"   对象类型：{dep['object_type']}")
            print(f"   依赖模块：{dep['module']}")
            print(f"   对象简要信息：{dep['object_repr']}")