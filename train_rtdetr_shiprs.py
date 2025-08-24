from ultralytics import RTDETR
import argparse
import time

def main(opt):
    # 初始化模型，Ultralytics 会自动下载权重
    model = RTDETR(opt.model)

    # 总训练开始时间
    total_start = time.time()

    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        workers=opt.workers,
        device=opt.device,
        optimizer="adamw",
        lr0=opt.lr0, lrf=opt.lrf, cos_lr=True,
        weight_decay=0.05,
        patience=40,
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.3,  # 减少HSV增强
        flipud=0.0, fliplr=0.3,  # 减少水平翻转概率
        mosaic=0.3, mixup=0.0,  # 减少Mosaic和禁用Mixup
        val=True, plots=True, save=True,
        save_period=10,    # 每10轮保存权重
        amp=True           # 启用 FP16
    )

    total_time = time.time() - total_start
    total_hours = total_time / 3600
    print(f"\n✅ 训练完成！总耗时约 {total_hours:.2f} 小时（估算）")
    print(results)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rtdetr-r18.pt", help="选择 RT-DETR 模型（Ultralytics 会自动下载）")
    ap.add_argument("--data", default="data/shiprs.yaml", help="数据集 YAML 文件")
    ap.add_argument("--epochs", type=int, default=100, help="训练总轮数")
    ap.add_argument("--imgsz", type=int, default=512, help="输入图像尺寸")
    ap.add_argument("--batch", type=int, default=1, help="批大小")
    ap.add_argument("--workers", type=int, default=4, help="Dataloader 并行线程数")
    ap.add_argument("--device", default="0", help="训练设备，GPU或CPU")
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--resume", action="store_true", help="是否从上次权重继续训练")
    opt = ap.parse_args()
    main(opt)
