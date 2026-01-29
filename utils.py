import os

from tqdm import tqdm

def yolo_to_xyxy(label, img_w, img_h):
    cls, x, y, w, h = label
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return int(cls), [x1, y1, x2, y2]

def execute_experiment(dataset: list, model, ) -> float:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    import torch

    metric = MeanAveragePrecision(iou_thresholds=[0.5]) # Imposta mAP@0.5

    for sample in tqdm(dataset, desc="Inferenza", unit="img"):
        image = sample["image"]
        labels = sample["labels"]

        h, w, _ = image.shape

        # ---------- GT ----------
        gt_boxes = []
        gt_labels = []

        for lab in labels:
            cls, box = yolo_to_xyxy(lab, w, h)
            gt_boxes.append(box)
            gt_labels.append(cls)

        target = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_labels, dtype=torch.int64)
        }

        # ---------- PRED ----------
        results = model(image, conf=0.001, verbose=False)

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            preds = {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        else:
            preds = {
                "boxes": r.boxes.xyxy.cpu(),
                "scores": r.boxes.conf.cpu(),
                "labels": r.boxes.cls.cpu().long()
            }

        metric.update([preds], [target])

    result = metric.compute()

    return result["map_50"]




def load_model(path_model: str):
    from ultralytics import YOLO
    model = YOLO(path_model)
    return model

def load_dataset(image_path: str, label_path: str) -> list:
    import cv2
    dataset = []

    image_files = os.listdir(image_path)

    for img_name in image_files:
        img_full_path = os.path.join(image_path, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_full_path = os.path.join(label_path, label_name)

        # carica immagine
        img = cv2.imread(img_full_path)
        if img is None:
            continue

        # carica label YOLO
        labels = []
        if os.path.exists(label_full_path):
            with open(label_full_path, "r") as f:
                for line in f:
                    labels.append(
                        list(map(float, line.strip().split()))
                    )
        # ogni riga: [class, x, y, w, h]

        dataset.append({
            "image_name": img_name,
            "image": img,
            "labels": labels
        })

    return dataset