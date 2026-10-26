import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="runs/video_inference")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {weights}")
    if not source.is_file():
        raise FileNotFoundError(f"Video tidak ditemukan: {source}")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf harus antara 0 dan 1")

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    output_path = output_dir / f"{source.stem}_detected.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Gagal membuat video output.")

    frame_count = 0
    total_inference_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            start = time.perf_counter()

            kwargs = {
                "source": frame,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "verbose": False,
            }
            if args.device:
                kwargs["device"] = args.device

            result = model.predict(**kwargs)[0]
            annotated = result.plot()

            elapsed = time.perf_counter() - start
            total_inference_time += elapsed
            frame_count += 1

            inference_fps = 1.0 / max(elapsed, 1e-9)

            cv2.putText(
                annotated,
                f"Inference FPS: {inference_fps:.1f}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            writer.write(annotated)

            if args.show:
                cv2.imshow("ASV YOLOv8 Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 100 == 0:
                print(f"Frame diproses: {frame_count}")

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print("\n=== Ringkasan ===")
    print(f"Frame diproses: {frame_count}")
    print(f"Waktu inferensi: {total_inference_time:.2f} detik")

    if total_inference_time > 0:
        print(
            "Rata-rata inferensi: "
            f"{frame_count / total_inference_time:.2f} FPS"
        )

    print(f"Video hasil: {output_path.resolve()}")


if __name__ == "__main__":
    main()