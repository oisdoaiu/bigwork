# 1. 手动构造完全相同的框坐标（用官方的真实框+预测框）
import jittor as jt
from nkyolo.utils.metrics import box_iou

# 统一输入：使用官方的 box1（真实框）和 box2（预测框）
common_box1_jt = jt.array([[230.15, 138.29, 578.41, 382.07]])
common_box2_jt = jt.array([[241.71, 138.79, 598.32, 392.35]])

# 2. 分别调用你的 box_iou 和官方的 box_iou
your_iou = box_iou(common_box1_jt, common_box2_jt)

# 3. 打印对比结果
print("你的 box_iou 输出：", your_iou.cpu().numpy())