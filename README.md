# Đồ Án 2 — Thiết bị hỗ trợ di chuyển cho người khiếm thị

Thiết bị dùng camera ToF **MaixSense A010** để phát hiện vật cản và đề xuất hướng đi bằng giọng nói, kèm một trợ lý AI giao tiếp bằng giọng nói tiếng Việt (mô tả xung quanh, thời tiết, định vị, chỉ đường đi bộ).

## Các file

| File | Mô tả |
|---|---|
| `test.py` | Đọc depth map 100x100 từ camera ToF qua UART, chia 3 vùng trái/giữa/phải để đề xuất hướng đi, phân loại vật cản (tường, cột, hộp) và đọc cảnh báo bằng giọng nói (pyttsx3, offline). |
| `tro_ly_giong_noi.py` | Trợ lý giọng nói bản đầu dùng Claude: trò chuyện, mô tả ảnh webcam, thời tiết (Open-Meteo), chỉ đường từng bước (Google Directions). |
| `gps.py` | Bản Claude có thêm định vị (GPS NMEA ngoài → Windows Location → IP) và dẫn đường trực tiếp theo GPS, cảnh báo đi lệch và tính lại lộ trình. |
| `test_1.py` | Bản Gemini: như `gps.py` nhưng dùng Gemini, chỉ đường mặc định bằng OpenStreetMap (miễn phí, không cần key), có xác nhận địa điểm bằng giọng nói. |

## Cài đặt

```bash
pip install -r requirements.txt
```

`pyaudio` trên Windows đôi khi cần cài riêng (`pip install pipwin && pipwin install pyaudio`).

## Biến môi trường

| Biến | Dùng cho |
|---|---|
| `ANTHROPIC_API_KEY` | `tro_ly_giong_noi.py`, `gps.py` |
| `GEMINI_API_KEY` | `test_1.py` |
| `GOOGLE_MAPS_API_KEY` | Chỉ đường bằng Google (tùy chọn với `test_1.py`) |
| `GPS_SERIAL_PORT`, `GPS_BAUD` | Module GPS ngoài (vd. `COM5`, `9600`) |
| `ROUTING_PROVIDER` | `osm` (mặc định) hoặc `google` — chỉ `test_1.py` |

Không ghi API key trực tiếp vào code.

## Chạy

```bash
# Camera ToF (sửa PORT trong file cho đúng cổng COM)
python test.py

# Trợ lý giọng nói
python test_1.py
```
