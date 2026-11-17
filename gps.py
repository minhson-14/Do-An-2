"""
Tro ly AI giao tiep bang giong noi danh cho nguoi khiem thi.

Tinh nang:
  1. Tro chuyen bang giong noi voi Claude.
  2. Mo ta anh chup tu webcam (xung quanh, doc chu...).
  3. Kiem tra thoi tiet (Open-Meteo, khong can API key).
  4. Dinh vi GPS: "toi dang o dau", tu dong lay vi tri hien tai.
  5. Chi duong di bo bang Google Maps:
       - Co GPS tot  -> che do DAN DUONG TRUC TIEP: tu dong bao buoc ke tiep khi
                        ban toi gan diem re, canh bao va tinh lai duong khi di lech.
       - Khong co GPS -> hoi ban dang o dau, doc tung buoc, ban noi "tiep" de nghe tiep.

Nguon dinh vi (thu theo thu tu, dung cai dau tien co ket qua):
  a) Module GPS gan ngoai (USB/Bluetooth NMEA)  -> chinh xac nhat (~3-10 m)
       dat bien moi truong GPS_SERIAL_PORT=COM5 (va GPS_BAUD=9600 neu can)
  b) Windows Location Service (dinh vi theo WiFi) -> ~20-100 m
       can bat: Settings > Privacy & security > Location
  c) Dinh vi theo dia chi IP -> chi ~ muc thanh pho (KHONG dung de dan duong)

Cai dat:
    pip install anthropic SpeechRecognition pyaudio gTTS pygame opencv-python requests
    pip install pyserial pynmea2        (chi can neu dung module GPS ngoai)

Bien moi truong:
    ANTHROPIC_API_KEY, GOOGLE_MAPS_API_KEY (bat Directions API + Geocoding API)

Chay:
    python tro_ly_giong_noi.py
"""

import base64
import html
import io
import math
import os
import re
import subprocess
import sys
import threading
import time

import cv2
import pygame
import requests
import speech_recognition as sr
from anthropic import Anthropic
from gtts import gTTS

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
MAX_HISTORY = 10
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Nguong dan duong truc tiep (met)
GOOD_ACCURACY_M = 100   # do chinh xac toi da de bat che do dan duong truc tiep
ARRIVE_RADIUS_M = 20    # coi nhu da toi diem re neu cach <= 20 m
NEAR_WARNING_M = 50     # bao truoc khi con <= 50 m
OFF_ROUTE_MARGIN_M = 60 # lech khoi buoc hien tai them bao nhieu met thi canh bao
POLL_SECONDS = 3

SYSTEM_PROMPT = """Ban la tro ly giong noi than thien cho nguoi khiem thi, noi tieng Viet.
Cau tra loi cua ban se duoc doc to bang may, nen:
- Tra loi ngan gon, tu nhien, 1 den 3 cau. Khong dung markdown, danh sach, bieu tuong cam xuc hay ky hieu dac biet.
- Khi mo ta hinh anh: noi nguy hiem hoac vat can truoc (bac thang, xe co, vat sac nhon...), sau do moi den bo cuc chung.
- Mo ta vi tri bang huong de hinh dung: "ben trai", "phia truoc khoang hai buoc chan", "o huong 2 gio".
- Khi doc chu trong anh, doc nguyen van phan quan trong va noi ro neu chu mo hoac khong chac chan.
- Neu khong chac dieu gi, hay noi thang. Voi viec lien quan den an toan (qua duong, thuoc men),
  nhac nguoi dung khong nen chi dua vao mo ta tu anh va nen nho nguoi xung quanh xac nhan.
- Xung "toi", goi nguoi dung la "ban"."""

EXIT_WORDS = ["thoát", "tạm biệt", "dừng lại", "kết thúc"]
CAMERA_WORDS = [
    "mô tả", "xung quanh", "trước mặt", "chụp ảnh", "nhìn thấy",
    "cái gì đây", "đây là gì", "đọc chữ", "đọc giúp", "đọc cho tôi",
]
WEATHER_WORDS = ["thời tiết", "trời hôm nay", "trời có mưa", "nắng hay mưa"]
WHERE_WORDS = ["tôi đang ở đâu", "vị trí của tôi", "vị trí hiện tại", "đây là đâu", "tôi ở đâu"]
DIRECTIONS_WORDS = [
    "chỉ đường", "dẫn đường", "đường đi đến", "đường đến", "đi đến",
    "làm sao để đến", "làm sao đi đến", "chỉ giúp tôi đường",
]
NAV_NEXT_WORDS = ["tiếp", "tiếp theo", "rồi sao", "sau đó"]
NAV_REPEAT_WORDS = ["nhắc lại", "lặp lại", "nói lại"]
NAV_STOP_WORDS = ["dừng chỉ đường", "hủy chỉ đường", "thôi khỏi chỉ đường", "dừng dẫn đường"]

client = Anthropic()
recognizer = sr.Recognizer()
pygame.mixer.init()

WEATHER_CODE_VI = {
    0: "trời quang, không mây", 1: "trời quang, ít mây", 2: "trời có mây rải rác",
    3: "trời nhiều mây, âm u", 45: "có sương mù", 48: "có sương mù đóng băng",
    51: "mưa phùn nhẹ", 53: "mưa phùn vừa", 55: "mưa phùn dày",
    61: "mưa nhỏ", 63: "mưa vừa", 65: "mưa to",
    66: "mưa lạnh, có thể đóng băng", 67: "mưa lạnh to",
    71: "tuyết rơi nhẹ", 73: "tuyết rơi vừa", 75: "tuyết rơi dày",
    80: "mưa rào nhẹ", 81: "mưa rào vừa", 82: "mưa rào rất to",
    95: "có dông", 96: "có dông kèm mưa đá nhẹ", 99: "có dông kèm mưa đá to",
}


# ---------- Dau ra: giong noi ----------
def speak(text: str) -> None:
    text = re.sub(r"[*_#`>~\[\]]", "", text).strip()
    if not text:
        return
    print(f"🔊 Trợ lý: {text}")
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang="vi").write_to_fp(buf)
        buf.seek(0)
        pygame.mixer.music.load(buf, "mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception as e:
        print(f"[Lỗi TTS] {e}")


def beep() -> None:
    import array

    rate, freq, ms = 22050, 880, 150
    n = int(rate * ms / 1000)
    samples = array.array(
        "h", [int(12000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]
    )
    pygame.mixer.quit()
    pygame.mixer.init(frequency=rate, size=-16, channels=1)
    pygame.mixer.Sound(buffer=samples.tobytes()).play()
    pygame.time.wait(ms + 50)
    pygame.mixer.quit()
    pygame.mixer.init()


# ---------- Dau vao: giong noi ----------
def listen(timeout: int = 8, phrase_time_limit: int = 15) -> str | None:
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        beep()
        print("🎤 Đang nghe...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return None
    try:
        text = recognizer.recognize_google(audio, language="vi-VN")
        print(f"👤 Bạn: {text}")
        return text
    except sr.UnknownValueError:
        speak("Xin lỗi, tôi chưa nghe rõ. Bạn nói lại được không?")
    except sr.RequestError:
        speak("Không kết nối được dịch vụ nhận dạng giọng nói. Vui lòng kiểm tra mạng.")
    return None


def ask_and_listen(question: str) -> str | None:
    speak(question)
    return listen(timeout=10, phrase_time_limit=12)


def contains_any(text: str, words: list[str]) -> bool:
    text = text.lower()
    return any(w in text for w in words)


def extract_after_keyword(text: str, keywords: list[str]) -> str | None:
    for kw in keywords:
        m = re.search(rf"{kw}\s+(.+)", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,?!")
    return None


# ---------- Camera ----------
def capture_image_b64(max_side: int = 1024) -> str | None:
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return None
    try:
        for _ in range(5):
            cam.read()
        ok, frame = cam.read()
    finally:
        cam.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(jpg.tobytes()).decode() if ok else None


# ======================================================================
#                           DINH VI GPS
# ======================================================================
def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Khoang cach giua 2 toa do, don vi met."""
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _location_from_serial(timeout: int = 10) -> dict | None:
    """Doc toa do tu module GPS gan ngoai (chuan NMEA) qua cong COM/USB."""
    port = os.getenv("GPS_SERIAL_PORT")
    if not port:
        return None
    try:
        import pynmea2
        import serial
    except ImportError:
        print("[GPS] Thiếu thư viện: pip install pyserial pynmea2")
        return None

    end = time.time() + timeout
    try:
        with serial.Serial(port, int(os.getenv("GPS_BAUD", "9600")), timeout=1) as ser:
            while time.time() < end:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if not line.startswith(("$GPGGA", "$GNGGA")):
                    continue
                try:
                    msg = pynmea2.parse(line)
                except pynmea2.ParseError:
                    continue
                if msg.gps_qual and int(msg.gps_qual) > 0 and msg.latitude:
                    hdop = float(msg.horizontal_dil or 5)
                    return {
                        "lat": msg.latitude,
                        "lng": msg.longitude,
                        "accuracy": round(hdop * 5),  # uoc luong gan dung
                        "source": "gps",
                    }
    except Exception as e:
        print(f"[GPS] Lỗi đọc cổng {port}: {e}")
    return None


_WIN_LOCATION_PS = r"""
Add-Type -AssemblyName System.Device
$w = New-Object System.Device.Location.GeoCoordinateWatcher([System.Device.Location.GeoPositionAccuracy]::High)
$w.Start()
$t = 0
while (($w.Status -ne 'Ready') -and ($t -lt 80)) { Start-Sleep -Milliseconds 100; $t++ }
$l = $w.Position.Location
if ($l.IsUnknown) { $w.Stop(); exit 1 }
$c = [cultureinfo]::InvariantCulture
"{0},{1},{2}" -f $l.Latitude.ToString($c), $l.Longitude.ToString($c), $l.HorizontalAccuracy.ToString($c)
$w.Stop()
"""


def _location_from_windows() -> dict | None:
    """Dung Windows Location Service (WiFi/GPS co san cua may)."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _WIN_LOCATION_PS],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        lat, lng, acc = out.stdout.strip().splitlines()[-1].split(",")
        acc = float(acc)
        return {
            "lat": float(lat),
            "lng": float(lng),
            "accuracy": round(acc) if acc == acc else 500,  # NaN -> 500
            "source": "windows",
        }
    except Exception:
        return None


def _location_from_ip() -> dict | None:
    """Dinh vi theo IP: chi o muc thanh pho, KHONG dung de dan duong."""
    try:
        d = requests.get("https://ipapi.co/json/", timeout=6).json()
        if "latitude" in d:
            return {"lat": d["latitude"], "lng": d["longitude"], "accuracy": 5000, "source": "ip"}
    except requests.RequestException:
        pass
    return None


def get_location(allow_ip: bool = True) -> dict | None:
    """Lay vi tri hien tai: GPS ngoai > Windows Location > IP."""
    return (
        _location_from_serial()
        or _location_from_windows()
        or (_location_from_ip() if allow_ip else None)
    )


def location_is_good(loc: dict | None) -> bool:
    return bool(loc) and loc["source"] != "ip" and loc["accuracy"] <= GOOD_ACCURACY_M


def reverse_geocode(lat: float, lng: float) -> str | None:
    """Doi toa do thanh dia chi de doc cho nguoi dung."""
    try:
        if GOOGLE_MAPS_API_KEY:
            res = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"latlng": f"{lat},{lng}", "language": "vi", "key": GOOGLE_MAPS_API_KEY},
                timeout=8,
            ).json()
            if res.get("status") == "OK" and res["results"]:
                return res["results"][0]["formatted_address"]
        res = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "json", "accept-language": "vi"},
            headers={"User-Agent": "tro-ly-giong-noi/1.0"},
            timeout=8,
        ).json()
        return res.get("display_name")
    except requests.RequestException:
        return None


def handle_where_am_i() -> None:
    speak("Đợi tôi một chút, tôi đang xác định vị trí.")
    loc = get_location()
    if not loc:
        speak("Tôi chưa xác định được vị trí. Bạn hãy kiểm tra dịch vụ vị trí của máy hoặc kết nối thiết bị GPS.")
        return
    addr = reverse_geocode(loc["lat"], loc["lng"]) or "không rõ địa chỉ cụ thể"
    msg = f"Bạn đang ở gần {addr}."
    if loc["source"] == "ip":
        msg += " Lưu ý đây chỉ là vị trí ước chừng ở mức thành phố, chưa chính xác."
    elif loc["accuracy"] > GOOD_ACCURACY_M:
        msg += f" Độ sai lệch có thể lên tới khoảng {loc['accuracy']} mét."
    else:
        msg += f" Sai lệch khoảng {loc['accuracy']} mét."
    speak(msg)


# ======================================================================
#                           THOI TIET
# ======================================================================
def get_weather(place: str) -> str:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 1, "language": "vi"},
            timeout=8,
        ).json()
        results = geo.get("results")
        if not results:
            return f"Tôi không tìm thấy địa điểm '{place}'. Bạn thử nói tên thành phố khác được không?"
        loc = results[0]
        ten = loc.get("name", place)

        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,apparent_temperature",
                "timezone": "auto",
            },
            timeout=8,
        ).json()
        cur = wx.get("current")
        if not cur:
            return "Xin lỗi, tôi không lấy được dữ liệu thời tiết lúc này."

        mo_ta = WEATHER_CODE_VI.get(cur.get("weather_code"), "chưa rõ tình trạng trời")
        nhiet_do = round(cur["temperature_2m"])
        cam_giac = round(cur["apparent_temperature"])
        do_am = cur["relative_humidity_2m"]
        gio = round(cur["wind_speed_10m"])

        loi_khuyen = ""
        if cur.get("weather_code", 0) >= 61:
            loi_khuyen = " Bạn nên mang theo áo mưa hoặc ô nếu ra ngoài."
        elif nhiet_do >= 35:
            loi_khuyen = " Trời khá nóng, bạn nhớ uống đủ nước."

        return (
            f"Thời tiết ở {ten} hiện tại là {mo_ta}, nhiệt độ {nhiet_do} độ C, "
            f"cảm giác như {cam_giac} độ. Độ ẩm {do_am} phần trăm, gió khoảng {gio} ki lô mét một giờ."
            f"{loi_khuyen}"
        )
    except requests.RequestException:
        return "Xin lỗi, tôi không kết nối được để kiểm tra thời tiết. Bạn kiểm tra mạng giúp tôi nhé."


def handle_weather(text: str) -> None:
    place = extract_after_keyword(text, ["ở", "tại", "cho"])
    if not place:
        place = ask_and_listen("Bạn muốn xem thời tiết ở đâu?")
    if not place:
        speak("Tôi chưa nghe rõ địa điểm, bạn thử lại sau nhé.")
        return
    speak(get_weather(place))


# ======================================================================
#                 CHI DUONG (Google Maps Directions API)
# ======================================================================
def strip_html(raw: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", raw)).strip()


def get_walking_directions(origin: str, destination: str) -> dict:
    if not GOOGLE_MAPS_API_KEY:
        return {"error": "Chưa thiết lập khóa Google Maps nên tôi không chỉ đường được. Bạn cần thêm biến môi trường GOOGLE_MAPS_API_KEY."}
    try:
        res = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params={
                "origin": origin,
                "destination": destination,
                "mode": "walking",
                "language": "vi",
                "key": GOOGLE_MAPS_API_KEY,
            },
            timeout=10,
        ).json()
    except requests.RequestException:
        return {"error": "Tôi không kết nối được tới Google Maps lúc này. Bạn kiểm tra mạng giúp tôi nhé."}

    status = res.get("status")
    if status == "ZERO_RESULTS":
        return {"error": "Tôi không tìm được đường đi bộ giữa hai địa điểm này."}
    if status == "NOT_FOUND":
        return {"error": "Tôi không xác định được một trong hai địa điểm bạn vừa nói. Bạn nói rõ hơn được không?"}
    if status != "OK":
        return {"error": f"Google Maps trả về lỗi: {status}. Bạn thử lại sau nhé."}

    leg = res["routes"][0]["legs"][0]
    steps = [
        {
            "instruction": strip_html(s["html_instructions"]),
            "distance": s["distance"]["text"],
            "meters": s["distance"]["value"],
            "end_lat": s["end_location"]["lat"],
            "end_lng": s["end_location"]["lng"],
        }
        for s in leg["steps"]
    ]
    return {
        "steps": steps,
        "total_distance": leg["distance"]["text"],
        "total_duration": leg["duration"]["text"],
        "end_address": leg.get("end_address", destination),
    }


def announce_route(info: dict) -> None:
    speak(
        f"Đã tìm thấy đường đi bộ đến {info['end_address']}. "
        f"Tổng quãng đường khoảng {info['total_distance']}, mất khoảng {info['total_duration']}."
    )


def run_manual_navigation(info: dict) -> None:
    """Che do doc tung buoc, nguoi dung noi 'tiep' de nghe buoc ke."""
    announce_route(info)
    speak("Tôi sẽ đọc từng bước một. Bạn nói 'tiếp' để nghe bước kế tiếp, 'nhắc lại' để nghe lại, hoặc 'dừng chỉ đường' để thoát.")

    steps = info["steps"]
    i = 0
    while i < len(steps):
        step = steps[i]
        speak(f"Bước {i + 1} trên {len(steps)}: {step['instruction']}, khoảng {step['distance']}.")
        reply = listen(timeout=15, phrase_time_limit=8)
        if not reply:
            continue
        if contains_any(reply, NAV_STOP_WORDS) or contains_any(reply, EXIT_WORDS):
            speak("Đã dừng chỉ đường.")
            return
        if contains_any(reply, NAV_REPEAT_WORDS):
            continue
        i += 1

    speak(f"Bạn đã đến nơi: {info['end_address']}. Chúc bạn một ngày tốt lành.")


def run_live_navigation(info: dict, destination: str) -> None:
    """Dan duong truc tiep bang GPS: tu dong bao buoc ke tiep, canh bao di lech, tinh lai duong."""
    announce_route(info)
    speak("Tôi sẽ theo dõi vị trí của bạn và báo khi tới chỗ rẽ. Bạn có thể nói 'nhắc lại' hoặc 'dừng chỉ đường' bất cứ lúc nào.")

    stop_event, repeat_event = threading.Event(), threading.Event()

    def on_audio(rec, audio):
        try:
            t = rec.recognize_google(audio, language="vi-VN")
        except Exception:
            return
        print(f"👤 (nền) Bạn: {t}")
        if contains_any(t, NAV_STOP_WORDS + EXIT_WORDS):
            stop_event.set()
        elif contains_any(t, NAV_REPEAT_WORDS):
            repeat_event.set()

    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
    stop_listening = recognizer.listen_in_background(mic, on_audio, phrase_time_limit=6)

    try:
        steps = info["steps"]
        i = 0
        while i < len(steps):
            step = steps[i]
            text = f"Bước {i + 1} trên {len(steps)}: {step['instruction']}, khoảng {step['distance']}."
            speak(text)
            warned_near = False
            recalculated = False

            while True:
                if stop_event.is_set():
                    speak("Đã dừng chỉ đường.")
                    return
                if repeat_event.is_set():
                    repeat_event.clear()
                    speak(text)

                loc = get_location(allow_ip=False)
                if not loc:
                    speak("Tôi tạm thời mất tín hiệu vị trí. Bạn hãy đứng yên một chút.")
                    time.sleep(POLL_SECONDS)
                    continue

                d = haversine_m(loc["lat"], loc["lng"], step["end_lat"], step["end_lng"])
                print(f"[GPS] cách điểm rẽ {d:.0f} m (sai số {loc['accuracy']} m)")

                if d <= max(ARRIVE_RADIUS_M, loc["accuracy"] * 0.5):
                    i += 1
                    break

                if not warned_near and d <= NEAR_WARNING_M and i + 1 < len(steps):
                    speak(f"Còn khoảng {int(d)} mét nữa. Sắp tới: {steps[i + 1]['instruction']}.")
                    warned_near = True

                if d > step["meters"] + max(OFF_ROUTE_MARGIN_M, loc["accuracy"] * 2):
                    speak("Có vẻ bạn đã đi lệch đường. Tôi đang tính lại lộ trình, bạn hãy dừng lại một chút.")
                    new_info = get_walking_directions(f"{loc['lat']},{loc['lng']}", destination)
                    if "error" in new_info:
                        speak(new_info["error"])
                    else:
                        info, steps, i = new_info, new_info["steps"], 0
                        speak(f"Đã tính lại. Còn khoảng {info['total_distance']}.")
                        recalculated = True
                        break

                time.sleep(POLL_SECONDS)

            if recalculated:
                continue

        speak(f"Bạn đã đến nơi: {info['end_address']}. Chúc bạn một ngày tốt lành.")
    finally:
        stop_listening(wait_for_stop=False)


def handle_directions(text: str) -> None:
    destination = extract_after_keyword(text, ["đến", "tới", "chỉ đường đi"])
    if not destination:
        destination = ask_and_listen("Bạn muốn đi đến địa chỉ hoặc địa điểm nào?")
    if not destination:
        speak("Tôi chưa nghe rõ địa điểm bạn muốn đến, bạn thử lại nhé.")
        return

    speak("Đợi tôi một chút, tôi đang xác định vị trí của bạn.")
    loc = get_location()

    if location_is_good(loc):
        info = get_walking_directions(f"{loc['lat']},{loc['lng']}", destination)
        if "error" in info:
            speak(info["error"])
            return
        run_live_navigation(info, destination)
        return

    # GPS khong du chinh xac -> hoi vi tri bang giong noi, doc tung buoc
    speak("Tôi chưa có tín hiệu vị trí đủ chính xác để dẫn đường trực tiếp.")
    origin = ask_and_listen("Bạn đang ở đâu? Bạn có thể nói địa chỉ hoặc mô tả gần đúng vị trí.")
    if not origin:
        speak("Tôi chưa nghe rõ vị trí hiện tại của bạn, bạn thử lại nhé.")
        return
    info = get_walking_directions(origin, destination)
    if "error" in info:
        speak(info["error"])
        return
    run_manual_navigation(info)


# ======================================================================
#                      Claude (tro chuyen chung + mo ta anh)
# ======================================================================
def ask_claude(history: list, user_text: str, image_b64: str | None = None) -> str:
    content = []
    if image_b64:
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}}
        )
    content.append({"type": "text", "text": user_text})

    messages = history[-MAX_HISTORY:] + [{"role": "user", "content": content}]
    response = client.messages.create(model=MODEL, max_tokens=400, system=SYSTEM_PROMPT, messages=messages)
    reply = "".join(b.text for b in response.content if b.type == "text").strip()

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    return reply


# ---------- Vong lap chinh ----------
def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Chưa đặt biến môi trường ANTHROPIC_API_KEY.")
    if not GOOGLE_MAPS_API_KEY:
        print("[Cảnh báo] Chưa đặt GOOGLE_MAPS_API_KEY - tính năng chỉ đường sẽ không hoạt động.")

    history: list = []
    speak(
        "Xin chào, tôi là trợ lý giọng nói của bạn. Sau tiếng bíp, hãy nói điều bạn cần. "
        "Bạn có thể hỏi thời tiết, hỏi tôi đang ở đâu, nhờ tôi chỉ đường, hoặc nhờ tôi mô tả xung quanh. "
        "Nói thoát để kết thúc."
    )

    while True:
        text = listen()
        if not text:
            continue

        if contains_any(text, EXIT_WORDS):
            speak("Tạm biệt bạn. Hẹn gặp lại.")
            break

        try:
            if contains_any(text, WEATHER_WORDS):
                handle_weather(text)
                continue

            if contains_any(text, WHERE_WORDS):
                handle_where_am_i()
                continue

            if contains_any(text, DIRECTIONS_WORDS):
                handle_directions(text)
                continue

            image = None
            if contains_any(text, CAMERA_WORDS):
                speak("Đợi tôi một chút, tôi đang chụp ảnh.")
                image = capture_image_b64()
                if image is None:
                    speak("Tôi không mở được camera. Bạn kiểm tra giúp nhé.")
                    continue
            speak(ask_claude(history, text, image))
        except Exception as e:
            print(f"[Lỗi] {e}")
            speak("Xin lỗi, đã có lỗi xảy ra. Bạn thử lại nhé.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng.")
