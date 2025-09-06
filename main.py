from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load model YOLO
model = YOLO("yolo11m.pt")

# Load video
cap = cv2.VideoCapture("video/Recording 2025-09-06 130657.mp4")

DISPLAY_WIDTH  = 760
DISPLAY_HEIGHT = 760

# Vạch đếm NGANG
line_y = DISPLAY_HEIGHT * 2 // 3 + 60
left_b = DISPLAY_WIDTH // 3 + 70

# Biến đếm
in_count, out_count = 0, 0
trajectories = {}
counted_ids = set()

# Thống kê loại xe riêng IN và OUT
vehicle_in_stats = defaultdict(int)
vehicle_out_stats = defaultdict(int)

# Map để gán ID hiển thị khi xe chạm vạch
display_id_map = {}
next_display_id = 1

TOL = 6  # tolerance

# ✅ Thiết lập ghi video đầu ra
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_result.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (DISPLAY_WIDTH, DISPLAY_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    results = model.track(frame, persist=True, tracker="botsort.yaml")

    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.int().cpu().tolist()
        clss  = results[0].boxes.cls.int().cpu().tolist()
        names = model.names

        for box, tracker_id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            label = names[int(cls)]

            prev = None
            if tracker_id not in trajectories:
                trajectories[tracker_id] = []
            else:
                if len(trajectories[tracker_id]) > 0:
                    prev = trajectories[tracker_id][-1]
            trajectories[tracker_id].append((cx, cy))

            # Vẽ trajectory
            pts = trajectories[tracker_id]
            for j in range(1, len(pts)):
                cv2.line(frame, pts[j - 1], pts[j], (0, 0, 255), 2)

            # Kiểm tra chạm vạch
            if tracker_id not in counted_ids:
                crossed = False
                if prev is not None:
                    prev_y = prev[1]
                    if (prev_y - line_y) * (cy - line_y) <= 0:
                        crossed = True
                    elif abs(cy - line_y) <= TOL and abs(prev_y - line_y) > TOL:
                        crossed = True
                else:
                    if abs(cy - line_y) <= TOL:
                        crossed = True

                if crossed:
                    if tracker_id not in display_id_map:
                        display_id_map[tracker_id] = next_display_id
                        next_display_id += 1
                    disp_id = display_id_map[tracker_id]

                    if cx < left_b:  # OUT
                        out_count += 1
                        vehicle_out_stats[label] += 1
                        color = (0, 0, 255)
                    else:            # IN
                        in_count += 1
                        vehicle_in_stats[label] += 1
                        color = (0, 255, 0)

                    counted_ids.add(tracker_id)

                    # Highlight khung và show ID+label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(frame, f"ID {disp_id} {label}", (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                if tracker_id in display_id_map:
                    disp_id = display_id_map[tracker_id]
                    color = (0, 0, 255) if cx < left_b else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(frame, f"ID {disp_id} {label}", (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Vẽ vạch đếm ---
    cv2.line(frame, (0, line_y), (left_b, line_y), (0, 0, 255), 3)                # OUT
    cv2.line(frame, (left_b, line_y), (DISPLAY_WIDTH, line_y), (0, 255, 0), 3)    # IN

    # --- Tạo khung panel thống kê ---
    panel_w, panel_h = 260, 340
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (50, 50, 50), -1)  # nền xám
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # --- Hiển thị số liệu ---
    cv2.putText(frame, "Traffic Stats", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame, f"OUT: {out_count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, f"IN : {in_count}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # OUT details
    y_offset = 130
    cv2.putText(frame, "OUT details:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    for i, (k, v) in enumerate(sorted(vehicle_out_stats.items(), key=lambda x: -x[1])[:6]):
        cv2.putText(frame, f"{k}: {v}", (20, y_offset + 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # IN details
    y_offset_in = y_offset + 25 + len(vehicle_out_stats) * 22 + 20
    cv2.putText(frame, "IN details:", (10, y_offset_in),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    for i, (k, v) in enumerate(sorted(vehicle_in_stats.items(), key=lambda x: -x[1])[:6]):
        cv2.putText(frame, f"{k}: {v}", (20, y_offset_in + 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # --- Xuất video & hiển thị ---
    out.write(frame)
    cv2.imshow("Traffic Detection - Saved Output", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
