from nkyolo.models.yolo.model import YOLO

model = YOLO("yolov5s.yaml", task="detect", verbose=True)


train_results = model.train(
    data="coco8.yaml",  # 数据集配置文件路径
    epochs=300,  # 训练周期数
    imgsz=640,  # 训练图像尺寸
    device="cpu",  # 运行设备 (例如 'cpu', 0, [0,1,2,3])
)

# metrics = model.val()

# results = model("path/to/image.jpg")  # Predict on an image
# results[0].show()  # Display results

# Export the model to ONNX format for deployment
# path = model.export(format="onnx")  # Returns the path to the exported model