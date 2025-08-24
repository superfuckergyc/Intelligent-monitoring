from ultralytics import RTDETR
import argparse

def main(opt):
    model = RTDETR(opt.weights)
    model.predict(
        source=opt.source,      # 图像/文件夹/视频
        imgsz=opt.imgsz,
        conf=opt.conf,
        iou=opt.iou,
        save=True,              # 输出到 runs/detect/predict*
        device=opt.device
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="runs/detect/train/weights/best.pt")
    ap.add_argument("--source", required=True, help="path/to/image_or_dir_or_video")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="0")
    opt = ap.parse_args()
    main(opt)
