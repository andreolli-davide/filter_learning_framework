import os

from tqdm import tqdm

def yolo_to_xyxy(label, img_w, img_h):
    cls, x, y, w, h = label
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return int(cls), [x1, y1, x2, y2]


def execute_experiment(dataset: list, model) -> float:
    import tempfile
    import cv2
    import os

    with tempfile.TemporaryDirectory() as tmp:
        images_dir = os.path.join(tmp, "images")
        os.makedirs(images_dir, exist_ok=True)

        # salva immagini temporanee
        for sample in dataset:
            image = sample["image"]
            img_name = sample["image_name"]

            tmp_img_path = os.path.join(images_dir, img_name)
            cv2.imwrite(tmp_img_path, image)

        # crea yaml
        yaml_path = create_yaml_config(tmp)

        # valida
        results = model.val(
            data=yaml_path,
            split='test',
            save_json=False,
            save_hybrid=False,
            plots=False,
            verbose=False
        )

        return results.box.map50


def create_yaml_config(tmp_dir: str):
    import yaml
    import os

    CLASS_NAMES = [
        "schizont",
        "gametocyte",
        "ring",
        "trophozoite"
    ]

    config = {
        "path": tmp_dir,
        "test": "images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }

    yaml_path = os.path.join(tmp_dir, "config.yaml")

    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    return yaml_path


def load_model(path_model: str):
    from ultralytics import YOLO
    model = YOLO(path_model)
    return model

def load_dataset(image_path: str, path_labels: str) -> list:
    import cv2
    dataset = []

    image_files = os.listdir(image_path)

    for img_name in image_files:
        img_full_path = os.path.join(image_path, img_name)

        # carica immagine
        img = cv2.imread(img_full_path)
        if img is None:
            continue

        dataset.append({
            "image_name": img_name,
            "image": img,
            "path_labels": path_labels
        })

    return dataset