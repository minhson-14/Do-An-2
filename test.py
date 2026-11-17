"""
Giai đoạn 1 + 2 + 3 + 4 - Camera ToF MaixSense A010
(Tối ưu hóa chống Lag + Ghi chú vật thể rõ ràng + Đề xuất hướng đi bằng giọng nói)
Giai đoạn 1: Đọc dữ liệu depth map 100x100 qua UART/USB.
Giai đoạn 2: Chia depth map thành 3 vùng, tính khoảng cách, đề xuất hướng đi.
Giai đoạn 3: Nhận diện loại chướng ngại vật (Tường, Cột, Hộp) và ghi chú rõ ràng.
Giai đoạn 4: Đọc to đề xuất hướng đi bằng giọng nói (text-to-speech), chạy trên
             luồng (thread) riêng để KHÔNG làm giật/chậm hình ảnh hiển thị.

Yêu cầu cài thêm cho giọng nói (offline, không cần internet, dùng SAPI5 có sẵn trên Windows):
    pip install pyttsx3
"""

import serial
import numpy as np
import cv2
import time
import threading
import queue
import sys

def _try_import_pyttsx3(auto_install=True):
    """
    Thử import pyttsx3. Nếu thiếu và auto_install=True, TỰ ĐỘNG cài đặt bằng
    đúng `sys.executable` (chính là Python đang chạy file test.py này) rồi thử
    import lại 1 lần -> đảm bảo cài đúng phiên bản Python đang dùng, không lo
    bị lẫn với 'pip' của một bản Python khác trên máy.
    """
    try:
        import pyttsx3
        return pyttsx3, None
    except Exception as e:
        if not auto_install:
            return None, e

        import subprocess

        def _pip_install():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyttsx3"])

        print(f"Đang tự động cài 'pyttsx3' vào Python: {sys.executable} ...")
        try:
            _pip_install()
        except Exception:
            # Một số bản Python (đặc biệt bản mới/cài tối giản) không có sẵn 'pip'
            # -> tự bootstrap pip bằng module ensurepip có sẵn trong Python, rồi thử cài lại.
            try:
                print("Python này chưa có 'pip' -> đang tự cài pip (ensurepip) ...")
                subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
                _pip_install()
            except Exception as e2:
                return None, e2

        try:
            import pyttsx3
            print("Đã cài xong 'pyttsx3'.")
            return pyttsx3, None
        except Exception as e3:
            return None, e3


pyttsx3, _TTS_IMPORT_ERROR = _try_import_pyttsx3(auto_install=True)
TTS_AVAILABLE = pyttsx3 is not None

# ====== CẤU HÌNH ======
PORT = "COM3"           # ĐỔI THÀNH CỔNG COM THỰC TẾ CỦA BẠN
BAUDRATE = 115200       # Giữ nguyên tốc độ này
FPS = 2                 # ĐÃ GIẢM XUỐNG 2 ĐỂ CHỐNG LAG (Băng thông tối đa ~1.4 FPS)
WIDTH, HEIGHT = 100, 100
UNIT = 0                 # 0 = auto; 1..10 = tuyến tính (mm)
HEADER = b"\x00\xFF"
END_BYTE = 0xDD
METADATA_LEN = 16
SAFE_THRESHOLD_MM = 800  # 80cm

# ====== CẤU HÌNH GIỌNG NÓI ======
VOICE_RATE = 175          # Tốc độ đọc (chữ/phút), tăng/giảm tùy sở thích
VOICE_REPEAT_SEC = 3.0    # Nếu hướng đi KHÔNG đổi, vẫn nhắc lại mỗi X giây (đặc biệt hữu ích cho cảnh báo)


# ====== MODULE ĐỌC GIỌNG NÓI (chạy nền, không chặn vòng lặp chính) ======
_voice_queue = queue.Queue(maxsize=2)  # chỉ giữ tối đa vài câu đang chờ, tránh nói dồn ứ
_voice_thread = None


def _voice_worker():
    """
    Chạy trong 1 thread riêng biệt: pyttsx3.runAndWait() là hàm CHẶN (blocking),
    nếu gọi trực tiếp trong vòng lặp chính sẽ làm treo/giật hình ảnh mỗi khi đọc.
    Bằng cách chạy trong thread riêng + giao tiếp qua Queue, vòng lặp xử lý ảnh
    hoàn toàn không bị ảnh hưởng bởi tốc độ đọc giọng nói.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", VOICE_RATE)
    while True:
        text = _voice_queue.get()
        if text is None:  # Tín hiệu để dừng thread khi thoát chương trình
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Lỗi TTS: {e}")


def start_voice_engine():
    """Khởi động thread đọc giọng nói. Gọi 1 lần trước vòng lặp chính."""
    global _voice_thread
    if not TTS_AVAILABLE:
        print(f"Cảnh báo: không dùng được 'pyttsx3' ({_TTS_IMPORT_ERROR}) -> tắt tính năng đọc giọng nói.")
        print(f"Python đang chạy chương trình này: {sys.executable}")
        print(f'Cài đặt đúng vào Python này bằng lệnh: "{sys.executable}" -m pip install pyttsx3')
        return
    _voice_thread = threading.Thread(target=_voice_worker, daemon=True)
    _voice_thread.start()


def speak_async(text):
    """
    Đưa câu cần đọc vào hàng đợi, KHÔNG chờ đọc xong (non-blocking).
    Nếu hàng đợi đầy (nghĩa là giọng nói đang đọc chậm hơn tốc độ khung hình mới),
    bỏ câu cũ nhất và chỉ giữ câu mới nhất -> tránh bị đọc dồn dập/trễ so với thực tế.
    """
    if not TTS_AVAILABLE or _voice_thread is None:
        return
    try:
        _voice_queue.put_nowait(text)
    except queue.Full:
        try:
            _voice_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _voice_queue.put_nowait(text)
        except queue.Full:
            pass


def stop_voice_engine():
    """Dừng thread đọc giọng nói một cách an toàn khi thoát chương trình."""
    if _voice_thread is not None:
        try:
            _voice_queue.put_nowait(None)
        except queue.Full:
            pass
        _voice_thread.join(timeout=1.0)


def send_at(ser, cmd):
    ser.write(f"AT+{cmd}\r".encode())
    time.sleep(0.1)


def pixel_to_mm(pixel_value, unit=UNIT):
    if unit == 0:
        return (pixel_value.astype(np.float32) / 5.1) ** 2
    else:
        return pixel_value.astype(np.float32) * unit


def read_frame(ser, buf_holder):
    """
    Đọc và parse 1 packet ảnh depth.
    ĐÃ TỐI ƯU: Tự động xả bớt buffer nếu dữ liệu tồn đọng quá nhiều để chống lag.
    """
    waiting = ser.in_waiting

    # Nếu dữ liệu tồn đọng quá nhiều (trễ quá 1 frame), xả bớt đi
    if waiting > 15000:
        ser.read(waiting - 1024)
        waiting = 1024

    chunk = ser.read(max(waiting, 1024))
    buf_holder.extend(chunk)

    idx = buf_holder.find(HEADER)
    if idx == -1:
        # Không tìm thấy header: chỉ giữ lại phần đuôi (phòng trường hợp header bị cắt đôi)
        if len(buf_holder) > 4096:
            del buf_holder[:-4096]
        return None

    if idx > 0:
        del buf_holder[:idx]

    needed = 2 + 2 + METADATA_LEN + (WIDTH * HEIGHT) + 2
    if len(buf_holder) < needed:
        return None

    # int.from_bytes chấp nhận bytearray/memoryview trực tiếp -> không cần bytes() copy
    packet_len = int.from_bytes(buf_holder[2:4], byteorder="little")
    total_len = 2 + 2 + packet_len + 2

    if len(buf_holder) < total_len:
        return None

    # Dùng memoryview để tránh copy dữ liệu 2 lần (chỉ copy 1 lần khi tạo frame bên dưới)
    mv = memoryview(buf_holder)
    tail_byte = mv[4 + packet_len + 1]
    image_data = bytes(mv[4 + METADATA_LEN: 4 + packet_len])

    ok = (tail_byte == END_BYTE and len(image_data) == WIDTH * HEIGHT)
    del mv
    del buf_holder[:total_len]

    if not ok:
        return None

    frame = np.frombuffer(image_data, dtype=np.uint8).reshape((HEIGHT, WIDTH))
    return frame


def compute_zone_averages(distance_mm):
    h, w = distance_mm.shape
    third = w // 3
    left_zone = distance_mm[:, 0:third]
    mid_zone = distance_mm[:, third:2 * third]
    right_zone = distance_mm[:, 2 * third:w]

    avg_left = float(np.median(left_zone))
    avg_mid = float(np.median(mid_zone))
    avg_right = float(np.median(right_zone))
    return avg_left, avg_mid, avg_right


def decide_direction(avg_left, avg_mid, avg_right):
    TIE_MARGIN_MM = 150
    if avg_mid >= 2000:
        best_direction = "GIỮA"
        best_distance = avg_mid
    else:
        if abs(avg_left - avg_right) < TIE_MARGIN_MM:
            if max(avg_left, avg_right) < SAFE_THRESHOLD_MM:
                best_direction = "GIỮA"
                best_distance = avg_mid
            else:
                best_direction = "TRÁI" if avg_left >= avg_right else "PHẢI"
                best_distance = max(avg_left, avg_right)
        else:
            best_direction = "TRÁI" if avg_left > avg_right else "PHẢI"
            best_distance = max(avg_left, avg_right)

    # Trả về 2 giá trị: (1) chữ hiển thị trên màn hình - có số mm chi tiết,
    # (2) câu ngắn gọn để đọc bằng giọng nói - bỏ số mm vì đọc số mm liên tục sẽ rối tai
    if best_distance < SAFE_THRESHOLD_MM:
        display_text = f"CẢNH BÁO: Vật cản gần (< {SAFE_THRESHOLD_MM}mm) - DỪNG LẠI"
        voice_text = "Cảnh báo, vật cản phía trước, dừng lại"
    else:
        display_text = f"Đề xuất: {best_direction} (~{best_distance:.0f}mm)"
        voice_map = {"TRÁI": "Rẽ trái", "PHẢI": "Rẽ phải", "GIỮA": "Đi thẳng"}
        voice_text = voice_map[best_direction]
    return display_text, voice_text


OBSTACLE_THRESHOLD_MM = 2000  # Ngưỡng cố định để coi là chướng ngại vật
DEPTH_BIN_MM = 300           # Độ dày mỗi "lớp" độ sâu dùng để tách vật thể liền kề nhưng khác khoảng cách
MIN_CONTOUR_AREA = 40        # Diện tích tối thiểu (điểm ảnh) của 1 vật thể để không bị coi là nhiễu
MIN_VALID_MM = 50            # Bỏ qua điểm ảnh gần 0mm (không phản xạ được / nhiễu cảm biến)
_reconstruction_small_buf = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)  # Buffer tái sử dụng, tránh cấp phát mỗi frame


def classify_obstacles(distance_mm):
    """
    Phân tích hình dáng Depth Map, phân loại chướng ngại vật.
    ĐÃ CẬP NHẬT: Vẽ hình khối trước, phóng to, sau đó mới ghi chú chữ to rõ ràng.
    ĐÃ TỐI ƯU: Ngưỡng mm cố định (không phụ thuộc distance_mm.max() mỗi frame -> ổn định
    hơn và bớt phép tính), tái sử dụng buffer ảnh nền thay vì cấp phát mới mỗi lần gọi.
    ĐÃ NÂNG CẤP PHÂN TÁCH: Phân đoạn theo từng LỚP ĐỘ SÂU (depth slicing) rồi mới tìm
    connected components trong từng lớp, thay vì gộp toàn bộ vùng < 2000mm thành 1
    ngưỡng nhị phân duy nhất. Nhờ vậy các vật thể ở khoảng cách khác nhau nhưng nằm
    sát nhau trong khung hình (ví dụ 1 cái hộp đứng trước 1 bức tường) sẽ được tách
    thành các khối riêng biệt, mỗi khối có nhãn + khoảng cách (mm) cụ thể, thay vì
    bị gộp chung thành 1 mảng trắng lớn ghi nhãn "VAT THE (OBJ)" như trước.
    """
    # 1. Tái sử dụng ảnh nền đen nhỏ (100x100) để chứa các khối màu
    reconstruction_small = _reconstruction_small_buf
    reconstruction_small.fill(0)

    detections = []  # Lưu thông tin: x, y, w, h, label, color

    # Loại bỏ điểm ảnh không hợp lệ (0mm hoặc quá gần do nhiễu/không phản xạ được)
    valid = distance_mm > MIN_VALID_MM

    # 2. PHÂN ĐOẠN THEO LỚP ĐỘ SÂU (depth slicing):
    # Thay vì gộp toàn bộ vùng "< 2000mm" thành 1 ngưỡng duy nhất (khiến nhiều vật
    # thể ở các khoảng cách khác nhau nhưng nằm sát nhau trong ảnh bị dính thành
    # 1 khối trắng lớn duy nhất), ta chia không gian độ sâu thành nhiều "lớp" dày
    # DEPTH_BIN_MM. Trong CÙNG một lớp, connectedComponents sẽ tách các vùng
    # không liền kề thành các vật thể riêng biệt -> phân tách rõ ràng và chi tiết hơn.
    num_bins = int(np.ceil(OBSTACLE_THRESHOLD_MM / DEPTH_BIN_MM))

    for i in range(num_bins):
        lo = i * DEPTH_BIN_MM
        hi = min((i + 1) * DEPTH_BIN_MM, OBSTACLE_THRESHOLD_MM)
        layer_mask = (valid & (distance_mm >= lo) & (distance_mm < hi)).astype(np.uint8) * 255
        # Bỏ qua sớm nếu lớp gần như trống (đỡ phải gọi connectedComponentsWithStats vô ích)
        if cv2.countNonZero(layer_mask) < MIN_CONTOUR_AREA:
            continue

        # connectedComponents: tách các vùng rời rạc trong cùng 1 lớp độ sâu
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(layer_mask, connectivity=8)

        for lbl in range(1, num_labels):  # nhãn 0 luôn là nền, bỏ qua
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < MIN_CONTOUR_AREA:  # Bỏ qua nhiễu
                continue

            x = stats[lbl, cv2.CC_STAT_LEFT]
            y = stats[lbl, cv2.CC_STAT_TOP]
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]

            component_mask = (labels[y:y + h, x:x + w] == lbl)
            aspect_ratio = float(w) / h if h > 0 else 0

            # "extent" = diện tích thực tế / diện tích bounding box, dùng thay cho
            # solidity (convex hull). Tính thẳng từ area/w/h đã có sẵn trong stats,
            # KHÔNG cần gọi cv2.findContours + cv2.convexHull cho từng vật thể nữa
            # (đây là phần tốn CPU nhất trước đây vì phải gọi nhiều lần mỗi frame,
            # nhân với số lớp độ sâu x số vật thể -> gây lag). extent cho kết quả
            # tương đương solidity với các khối gần như trục-thẳng (box/tường/cột).
            extent = float(area) / (w * h) if (w * h) > 0 else 0

            # Khoảng cách trung bình thực tế của riêng vật thể này (dùng mean thay
            # vì median để tránh sắp xếp mảng, rẻ hơn một chút)
            avg_dist = float(distance_mm[y:y + h, x:x + w][component_mask].mean())

            # Logic phân loại vật thể (kèm khoảng cách cụ thể của từng vật thể)
            if w > 40 and h > 40 and extent > 0.8:
                label = f"TUONG {avg_dist:.0f}mm"
                color = (255, 0, 0)  # Xanh dương
            elif aspect_ratio < 0.4 or aspect_ratio > 2.5:
                label = f"COT {avg_dist:.0f}mm"
                color = (0, 255, 255)  # Vàng
            elif extent > 0.85:
                label = f"HOP {avg_dist:.0f}mm"
                color = (0, 255, 0)  # Xanh lá
            else:
                label = f"VAT THE {avg_dist:.0f}mm"
                color = (255, 255, 255)  # Trắng

            # Vẽ khối màu đặc lên ảnh nhỏ (chỉ tô đúng hình dạng vật thể, không tô cả bounding box)
            layer_crop = reconstruction_small[y:y + h, x:x + w]
            layer_crop[component_mask] = color
            detections.append((x, y, w, h, label, color))

    # 3. Phóng to ảnh lên 450x450 (Lúc này chữ sẽ không bị nhòe)
    reconstruction_large = cv2.resize(reconstruction_small, (450, 450), interpolation=cv2.INTER_NEAREST)

    # 4. Ghi chú (Annotate) chữ to, rõ ràng lên ảnh lớn
    scale_factor = 450 / WIDTH # Hệ số phóng to (4.5)

    for (x, y, w, h, label, color) in detections:
        # Chuyển tọa độ từ ảnh nhỏ sang ảnh lớn
        x1, y1 = int(x * scale_factor), int(y * scale_factor)
        x2, y2 = int((x + w) * scale_factor), int((y + h) * scale_factor)

        # Vẽ viền hộp bao quanh vật thể
        cv2.rectangle(reconstruction_large, (x1, y1), (x2, y2), color, 2)

        # Tính toán kích thước chữ
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_w, text_h = text_size

        # Đặt chữ bên trên hộp. Nếu sát mép trên quá thì đẩy xuống dưới
        if y1 - text_h - 10 > 0:
            text_y = y1 - 10
        else:
            text_y = y1 + text_h + 10

        text_x = x1

        # Vẽ nền đen phía sau chữ để dễ đọc
        cv2.rectangle(reconstruction_large, (text_x, text_y - text_h - 5), (text_x + text_w, text_y + 5), (0, 0, 0), -1)

        # Ghi chữ (Ghi chú loại vật thể)
        cv2.putText(reconstruction_large, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return reconstruction_large


def main():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        time.sleep(0.5)

        # Cấu hình Camera
        send_at(ser, "ISP=1")
        send_at(ser, "BINN=1")
        send_at(ser, f"UNIT={UNIT}")
        send_at(ser, f"FPS={FPS}")
        send_at(ser, "DISP=6")

        print(f"Đang đọc dữ liệu ở {FPS} FPS... Nhấn 'q' trên cửa sổ hiển thị để thoát.")
    except Exception as e:
        print(f"Lỗi kết nối Camera: {e}")
        print("Hãy kiểm tra lại cổng COM và cáp kết nối.")
        return

    from collections import deque
    history_left = deque(maxlen=5)
    history_mid = deque(maxlen=5)
    history_right = deque(maxlen=5)
    read_buffer = bytearray()

    start_voice_engine()
    last_voice_text = None
    last_voice_time = 0.0

    try:
      while True:
        frame = read_frame(ser, read_buffer)
        if frame is None:
            # Không bận vòng lặp 100% CPU khi chưa có frame mới
            time.sleep(0.005)
            continue

        distance_mm = pixel_to_mm(frame)

        # 1. Tính toán hướng đi (Giai đoạn 2)
        raw_left, raw_mid, raw_right = compute_zone_averages(distance_mm)
        history_left.append(raw_left)
        history_mid.append(raw_mid)
        history_right.append(raw_right)
        avg_left = sum(history_left) / len(history_left)
        avg_mid = sum(history_mid) / len(history_mid)
        avg_right = sum(history_right) / len(history_right)

        suggestion, voice_text = decide_direction(avg_left, avg_mid, avg_right)

        # 1b. Đọc đề xuất bằng giọng nói (Giai đoạn 4)
        # Chỉ đọc lại khi: (a) hướng đi vừa đổi -> đọc ngay, hoặc
        # (b) hướng đi không đổi nhưng đã quá VOICE_REPEAT_SEC giây -> nhắc lại
        # (đặc biệt quan trọng cho cảnh báo "dừng lại" cần lặp lại đều đặn).
        now = time.time()
        if voice_text != last_voice_text or (now - last_voice_time) >= VOICE_REPEAT_SEC:
            speak_async(voice_text)
            last_voice_text = voice_text
            last_voice_time = now

        # 2. Nhận diện chướng ngại vật (Giai đoạn 3 - AI)
        obstacle_map = classify_obstacles(distance_mm)

        # ================= HIỂN THỊ MÀN HÌNH 1: RGB / DEPTH VIEW =================
        inverted = 255 - frame
        colored_depth = cv2.applyColorMap(inverted, cv2.COLORMAP_JET)
        colored_depth = cv2.resize(colored_depth, (450, 450), interpolation=cv2.INTER_NEAREST)

        third_px = 450 // 3
        cv2.line(colored_depth, (third_px, 0), (third_px, 450), (255, 255, 255), 1)
        cv2.line(colored_depth, (2 * third_px, 0), (2 * third_px, 450), (255, 255, 255), 1)

        cv2.putText(colored_depth, f"{avg_left:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(colored_depth, f"{avg_mid:.0f}", (third_px + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(colored_depth, f"{avg_right:.0f}", (2 * third_px + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(colored_depth, suggestion, (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ================= HIỂN THỊ MÀN HÌNH 2: OBSTACLE RECONSTRUCTION =================
        cv2.putText(obstacle_map, "AI OBSTACLE DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Hiển thị cả 2 cửa sổ
        cv2.imshow("1. RGB View (Depth Map) + Navigation", colored_depth)
        cv2.imshow("2. AI Reconstruction (Obstacle Type)", obstacle_map)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        cv2.destroyAllWindows()
        stop_voice_engine()


if __name__ == "__main__":
    main()
