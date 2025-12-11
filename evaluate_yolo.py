from ultralytics import YOLO

model = YOLO("best.pt")

print("\n🔍 Evaluating YOLO model on test set...\n")

metrics = model.val(data="data.yaml")   # Your YOLO dataset YAML

print("\n --- YOLO RESULTS ---")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("----------------------")
