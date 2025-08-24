from ultralytics import RTDETR
import argparse

def main(opt):
    model = RTDETR(opt.weights)  # 训练得到的 best.pt
    metrics = model.val(
        data=opt.data,
        imgsz=opt.imgsz,
        conf=opt.conf,   # 可适当低一点，提高小目标召回
        iou=opt.iou
    )
    print(metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="runs/detect/train/weights/best.pt")
    ap.add_argument("--data", default="data/shiprs.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    opt = ap.parse_args()
    main(opt)
