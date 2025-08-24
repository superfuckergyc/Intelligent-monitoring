from ultralytics import RTDETR
import cv2
import numpy as np
from pathlib import Path
import argparse

def nms(boxes, scores, iou_thres=0.5):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_j = (boxes[idxs[1:],2]-boxes[idxs[1:],0])*(boxes[idxs[1:],3]-boxes[idxs[1:],1])
        ovr = inter / (area_i + area_j - inter + 1e-9)
        idxs = idxs[1:][ovr <= iou_thres]
    return keep

def infer_image_tiled(image_path, out_path, weights, imgsz=1280, conf=0.2, iou=0.5, tile=1024, overlap=256, device="0"):
    model = RTDETR(weights)
    im = cv2.imread(str(image_path))
    H, W = im.shape[:2]

    stride = tile - overlap
    all_boxes, all_scores, all_cls = [], [], []

    for y in range(0, max(1, H - overlap), stride):
        for x in range(0, max(1, W - overlap), stride):
            x2, y2 = min(x + tile, W), min(y + tile, H)
            tile_img = im[y:y2, x:x2]
            res = model.predict(source=tile_img, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
            if res.boxes is None or res.boxes.xyxy.numel() == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            sc = res.boxes.conf.cpu().numpy()
            cl = res.boxes.cls.cpu().numpy().astype(int)

            xyxy[:, [0,2]] += x
            xyxy[:, [1,3]] += y

            all_boxes.append(xyxy)
            all_scores.append(sc)
            all_cls.append(cl)

    if not all_boxes:
        print("No detections.")
        cv2.imwrite(str(out_path), im)
        return

    boxes = np.concatenate(all_boxes, 0)
    scores = np.concatenate(all_scores, 0)
    clses = np.concatenate(all_cls, 0)

    final_boxes, final_scores, final_cls = [], [], []
    for c in np.unique(clses):
        idx = np.where(clses == c)[0]
        keep = nms(boxes[idx], scores[idx], iou_thres=0.5)
        final_boxes.append(boxes[idx][keep])
        final_scores.append(scores[idx][keep])
        final_cls.append(np.full(len(keep), c, dtype=int))

    final_boxes = np.concatenate(final_boxes, 0)
    final_scores = np.concatenate(final_scores, 0)
    final_cls = np.concatenate(final_cls, 0)

    for (x1, y1, x2, y2), sc, c in zip(final_boxes, final_scores, final_cls):
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(im, f"{c}:{sc:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imwrite(str(out_path), im)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="runs/detect/train/weights/best.pt")
    ap.add_argument("--image", required=True, help="path/to/large_image.jpg")
    ap.add_argument("--out", default="out.jpg")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    infer_image_tiled(
        image_path=Path(args.image),
        out_path=Path(args.out),
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        tile=args.tile,
        overlap=args.overlap,
        device=args.device
    )
