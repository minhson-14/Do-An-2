"""
Tro ly AI giao tiep bang giong noi danh cho nguoi khiem thi.

Tinh nang:
  1. Tro chuyen bang giong noi voi Gemini.
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
    pip install google-genai SpeechRecognition pyaudio gTTS pygame opencv-python requests
    pip install pyserial pynmea2        (chi can neu dung module GPS ngoai)

Bien moi truong:
    GEMINI_API_KEY (mien phi, lay tai https://aistudio.google.com/apikey)
    GEMINI_MODEL   (tuy chon, mac dinh gemini-2.5-flash; doi neu bi loi 404)
    GOOGLE_MAPS_API_KEY (TUY CHON, chi can neu dat ROUTING_PROVIDER=google; mac dinh chi duong dung OpenStreetMap mien phi, khong can the)

Chay:
    python tro_ly_giong_noi.py
"""

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
from google import genai
from google.genai import types
from gtts import gTTS

# [SỬA ĐỔI 1] Đổi tên model mặc định sang bản ổn định (tránh lỗi 404)
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_HISTORY = 10
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Nguong dan duong truc tiep (met)
GOOD_ACCURACY_M = int(os.getenv("LIVE_NAV_MAX_ACCURACY_M", "50"))   # <= muc nay: dan duong truc tiep
START_ACCURACY_M = int(os.getenv("START_MAX_ACCURACY_M", "200"))    # <= muc nay: dung lam diem xuat phat
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
WHERE_WORDS = ["đang ở đâu", "tôi ở đâu", "mình ở đâu", "vị trí của tôi", "vị trí hiện tại", "đây là đâu"]
DIRECTIONS_WORDS = [
    "chỉ đường", "dẫn đường", "đường đi đến", "đường đến", "đi đến",
    "làm sao để đến", "làm sao đi đến", "chỉ giúp tôi đường",
]
NAV_NEXT_WORDS = ["tiếp", "tiếp theo", "rồi sao", "sau đó"]
NAV_REPEAT_WORDS = ["nhắc lại", "lặp lại", "nói lại"]
NAV_STOP_WORDS = ["dừng chỉ đường", "hủy chỉ đường", "thôi khỏi chỉ đường", "dừng dẫn đường"]

# [SỬA ĐỔI 2] Tăng thời gian timeout từ 30s lên 120s để tránh lỗi Read timed out
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=120_000),
)
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


def clean_place(text: str) -> str:
    """Bo cac cum thua nhu 'toi dang o', 'toi o gan' de Google de tim dia diem hon."""
    text = re.sub(
        r"^\s*(tôi|mình|em|tớ)?\s*(đang)?\s*(đứng|ngồi)?\s*(ở|tại|gần)\s+(gần\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(" .,?!") or text


# ---------- Camera ----------
# [SỬA ĐỔI 3] Giảm kích thước ảnh và chất lượng nén để upload nhanh hơn, tránh timeout
def capture_image_bytes(max_side: int = 800) -> bytes | None:
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
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpg.tobytes() if ok else None


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
while (($w.Status -ne 'Ready') -and ($t -lt 200)) { Start-Sleep -Milliseconds 100; $t++ }
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
            capture_output=True, text=True, timeout=30,
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
    """Du chinh xac de dan duong truc tiep (tu dong bao khi toi cho re)."""
    return bool(loc) and loc["source"] != "ip" and loc["accuracy"] <= GOOD_ACCURACY_M


def location_ok_for_start(loc: dict | None) -> bool:
    """Du de lam diem xuat phat (sau khi nguoi dung xac nhan), chua du de dan duong truc tiep."""
    return bool(loc) and loc["source"] != "ip" and loc["accuracy"] <= START_ACCURACY_M


YES_WORDS = ["đúng", "phải", "ừ", "ok", "chính xác", "đồng ý", "có"]
NO_WORDS = ["không", "sai", "chưa", "nhầm"]


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
#                 CHI DUONG (Google Routes API)
# ======================================================================
def strip_html(raw: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", raw)).strip()


def _route_point(text: str) -> dict:
    """Chuyen 'lat,lng' thanh toa do, con lai coi la dia chi."""
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", text)
    if m:
        return {"location": {"latLng": {"latitude": float(m.group(1)), "longitude": float(m.group(2))}}}
    return {"address": text}


def get_walking_directions_google(origin: str, destination: str) -> dict:
    """Chi duong di bo bang Google Routes API (can tai khoan thanh toan)."""
    if not GOOGLE_MAPS_API_KEY:
        return {"error": "Chưa thiết lập khóa Google Maps nên tôi không chỉ đường được. Bạn cần thêm biến môi trường GOOGLE_MAPS_API_KEY."}

    field_mask = ",".join([
        "routes.legs.distanceMeters",
        "routes.legs.duration",
        "routes.legs.localizedValues",
        "routes.legs.steps.navigationInstruction",
        "routes.legs.steps.distanceMeters",
        "routes.legs.steps.endLocation",
        "routes.legs.steps.localizedValues",
    ])
    try:
        resp = requests.post(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
                "X-Goog-FieldMask": field_mask,
            },
            json={
                "origin": _route_point(origin),
                "destination": _route_point(destination),
                "travelMode": "WALK",
                "languageCode": "vi",
                "units": "METRIC",
            },
            timeout=10,
        )
    except requests.RequestException:
        return {"error": "Tôi không kết nối được tới Google Maps lúc này. Bạn kiểm tra mạng giúp tôi nhé."}

    if resp.status_code in (401, 403):
        print(f"[Routes API] {resp.status_code}: {resp.text}")
        return {"error": "Khóa Google Maps bị từ chối. Bạn kiểm tra đã bật Routes API, gắn thanh toán và không giới hạn sai dịch vụ chưa."}
    if resp.status_code == 404:
        return {"error": "Tôi không xác định được một trong hai địa điểm bạn vừa nói. Bạn nói rõ hơn được không?"}
    if resp.status_code != 200:
        print(f"[Routes API] {resp.status_code}: {resp.text}")
        return {"error": "Google Maps trả về lỗi khi tìm đường. Bạn thử lại sau nhé."}

    routes = resp.json().get("routes")
    if not routes:
        return {"error": "Tôi không tìm được đường đi bộ giữa hai địa điểm này."}

    leg = routes[0]["legs"][0]
    steps = []
    for st in leg.get("steps", []):
        instr = (st.get("navigationInstruction") or {}).get("instructions") or "Đi tiếp"
        loc = (st.get("endLocation") or {}).get("latLng") or {}
        meters = st.get("distanceMeters", 0)
        dist_text = (st.get("localizedValues") or {}).get("distance", {}).get("text") or f"{meters} mét"
        steps.append({
            "instruction": strip_html(instr),
            "distance": dist_text,
            "meters": meters,
            "end_lat": loc.get("latitude"),
            "end_lng": loc.get("longitude"),
        })
    if not steps:
        return {"error": "Tôi không lấy được các bước chỉ đường chi tiết cho tuyến này."}

    lv = leg.get("localizedValues") or {}
    total_distance = lv.get("distance", {}).get("text") or f"{leg.get('distanceMeters', 0)} mét"
    total_duration = lv.get("duration", {}).get("text")
    if not total_duration:
        secs = int(str(leg.get("duration", "0s")).rstrip("s") or 0)
        total_duration = f"{max(1, secs // 60)} phút"

    return {
        "steps": steps,
        "total_distance": total_distance,
        "total_duration": total_duration,
        "end_address": destination,
    }


# ---------- Chi duong MIEN PHI (OpenStreetMap, khong can the, khong can API key) ----------
ROUTING_PROVIDER = os.getenv("ROUTING_PROVIDER", "osm").lower()  # "osm" (mac dinh) hoac "google"
OSM_USER_AGENT = "tro-ly-giong-noi/1.0 (do an ca nhan)"
OSRM_FOOT_URL = "https://routing.openstreetmap.de/routed-foot/route/v1/foot"
_last_osm_call = 0.0


def _osm_wait() -> None:
    """Cac dich vu OSM mien phi yeu cau toi da ~1 yeu cau/giay."""
    global _last_osm_call
    gap = time.time() - _last_osm_call
    if gap < 1.1:
        time.sleep(1.1 - gap)
    _last_osm_call = time.time()


def parse_latlng(text: str) -> tuple[float, float] | None:
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def geocode_place(text: str) -> tuple[float, float, str] | None:
    """Ten dia diem -> (lat, lng, ten). Thu Nominatim roi Photon (deu mien phi)."""
    pt = parse_latlng(text)
    if pt:
        return pt[0], pt[1], text
    try:
        _osm_wait()
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": text, "format": "jsonv2", "limit": 1, "countrycodes": "vn", "accept-language": "vi"},
            headers={"User-Agent": OSM_USER_AGENT},
            timeout=10,
        ).json()
        if r:
            return float(r[0]["lat"]), float(r[0]["lon"]), r[0].get("display_name", text)
    except (requests.RequestException, ValueError, KeyError):
        pass
    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": text, "limit": 1, "bbox": "102.1,8.4,109.5,23.4"},  # khung quanh Viet Nam
            headers={"User-Agent": OSM_USER_AGENT},
            timeout=10,
        ).json()
        feats = r.get("features") or []
        if feats:
            lng, lat = feats[0]["geometry"]["coordinates"]
            pr = feats[0].get("properties", {})
            name = ", ".join(x for x in [pr.get("name"), pr.get("street"), pr.get("city") or pr.get("state")] if x)
            return lat, lng, name or text
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass
    return None


_MOD_VI = {
    "uturn": "quay đầu", "sharp right": "rẽ phải gấp", "right": "rẽ phải",
    "slight right": "chếch sang phải", "straight": "đi thẳng", "slight left": "chếch sang trái",
    "left": "rẽ trái", "sharp left": "rẽ trái gấp",
}
_COMPASS_VI = ["bắc", "đông bắc", "đông", "đông nam", "nam", "tây nam", "tây", "tây bắc"]


def _compass(deg: float) -> str:
    return _COMPASS_VI[int(((deg % 360) + 22.5) // 45) % 8]


def _fmt_dist(meters: float) -> str:
    m = int(round(meters))
    if m >= 1000:
        return f"{m / 1000:.1f}".replace(".", " phẩy ") + " ki lô mét"
    return f"{max(m, 1)} mét"


def _osrm_instruction(step: dict) -> str:
    """Tao cau huong dan tieng Viet tu du lieu buoc cua OSRM."""
    man = step.get("maneuver", {})
    typ, mod = man.get("type", ""), man.get("modifier", "")
    name = step.get("name", "")
    on_road = f" vào đường {name}" if name else ""
    along = f" trên đường {name}" if name else ""

    if typ == "depart":
        d = man.get("bearing_after")
        return f"Xuất phát, đi về hướng {_compass(d)}{along}" if d is not None else f"Xuất phát{along}"
    if typ == "arrive":
        return "Bạn đã đến nơi"
    if typ in ("roundabout", "rotary", "roundabout turn"):
        ex = man.get("exit")
        return f"Vào vòng xoay, ra ở lối thứ {ex}{on_road}" if ex else f"Đi qua vòng xoay{on_road}"
    if typ in ("exit roundabout", "exit rotary"):
        return f"Ra khỏi vòng xoay{on_road}"
    if typ in ("continue", "new name") and mod in ("", "straight"):
        return f"Đi thẳng{along}"

    action = _MOD_VI.get(mod, "đi tiếp")
    if typ == "fork":
        return f"Tại ngã ba, {action}{on_road}"
    if typ == "end of road":
        return f"Cuối đường, {action}{on_road}"
    return action[0].upper() + action[1:] + on_road


def get_walking_directions_osm(origin: str, destination: str) -> dict:
    """Chi duong di bo bang OSRM (FOSSGIS) + Nominatim: mien phi, khong can key."""
    o = geocode_place(origin)
    if not o:
        return {"error": f"Tôi không tìm thấy địa điểm {origin} trên bản đồ. Bạn nói rõ hơn tên đường và thành phố được không?"}
    d = geocode_place(destination)
    if not d:
        return {"error": f"Tôi không tìm thấy địa điểm {destination} trên bản đồ. Bạn nói rõ hơn tên đường và thành phố được không?"}
    olat, olng, _ = o
    dlat, dlng, _ = d

    try:
        _osm_wait()
        resp = requests.get(
            f"{OSRM_FOOT_URL}/{olng},{olat};{dlng},{dlat}",
            params={"steps": "true", "overview": "false", "geometries": "geojson"},
            headers={"User-Agent": OSM_USER_AGENT},
            timeout=20,
        )
    except requests.RequestException:
        return {"error": "Tôi không kết nối được dịch vụ bản đồ lúc này. Bạn kiểm tra mạng và thử lại nhé."}

    if resp.status_code != 200:
        print(f"[OSRM] {resp.status_code}: {resp.text[:300]}")
        return {"error": "Dịch vụ bản đồ miễn phí đang gặp lỗi hoặc quá tải. Bạn thử lại sau ít phút nhé."}

    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        return {"error": "Tôi không tìm được đường đi bộ giữa hai địa điểm này."}

    route = data["routes"][0]
    raw = route["legs"][0]["steps"]
    steps = []
    for i in range(len(raw) - 1):  # buoc cuoi la 'arrive', da co loi thong bao den noi
        end = raw[i + 1]["maneuver"]["location"]  # [lng, lat]: noi bat dau buoc ke tiep
        steps.append({
            "instruction": _osrm_instruction(raw[i]),
            "distance": _fmt_dist(raw[i].get("distance", 0)),
            "meters": raw[i].get("distance", 0),
            "end_lat": end[1],
            "end_lng": end[0],
        })
    if not steps:
        return {"error": "Hai địa điểm quá gần nhau nên không cần chỉ đường."}

    return {
        "steps": steps,
        "total_distance": _fmt_dist(route.get("distance", 0)),
        "total_duration": f"{max(1, round(route.get('duration', 0) / 60))} phút",
        "end_address": destination,
    }


def get_walking_directions(origin: str, destination: str) -> dict:
    """Mac dinh dung OpenStreetMap (mien phi). Dat ROUTING_PROVIDER=google de dung Google."""
    if ROUTING_PROVIDER == "google":
        return get_walking_directions_google(origin, destination)
    return get_walking_directions_osm(origin, destination)


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
                        new_info["end_address"] = info["end_address"]
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


# ---------- Tim dia diem chinh xac hon (nhieu cach tim + xac nhan bang giong noi) ----------
def _sort_by_bias(cands: list, bias: tuple | None) -> list:
    if bias and len(cands) > 1:
        return sorted(cands, key=lambda c: haversine_m(bias[0], bias[1], c[0], c[1]))
    return cands


def _nominatim_candidates(q: str, bias: tuple | None) -> list:
    params = {"q": q, "format": "jsonv2", "limit": 5, "countrycodes": "vn", "accept-language": "vi"}
    if bias:  # uu tien ket qua gan vi tri nguoi dung (khong gioi han cung)
        lat, lng = bias
        params["viewbox"] = f"{lng - 0.3},{lat + 0.3},{lng + 0.3},{lat - 0.3}"
    try:
        _osm_wait()
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params, headers={"User-Agent": OSM_USER_AGENT}, timeout=10,
        ).json()
        return [(float(x["lat"]), float(x["lon"]), x.get("display_name", q)) for x in r]
    except (requests.RequestException, ValueError, KeyError):
        return []


def _photon_candidates(q: str, bias: tuple | None) -> list:
    params = {"q": q, "limit": 5, "bbox": "102.1,8.4,109.5,23.4"}
    if bias:
        params["lat"], params["lon"] = bias
    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params=params, headers={"User-Agent": OSM_USER_AGENT}, timeout=10,
        ).json()
        out = []
        for f in r.get("features") or []:
            lng, lat = f["geometry"]["coordinates"]
            pr = f.get("properties", {})
            name = ", ".join(x for x in [pr.get("name"), pr.get("street"), pr.get("district"),
                                         pr.get("city") or pr.get("state")] if x)
            out.append((lat, lng, name or q))
        return out
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return []


def _query_variants(text: str) -> list[str]:
    variants = [text]
    t2 = re.sub(r"\b(thành phố|tp\.?)\s+", "", text, flags=re.IGNORECASE).strip()
    if t2 and t2 != text:
        variants.append(t2)
    return variants


def _gemini_query_variants(text: str) -> list[str]:
    """Nho Gemini goi y cach goi khac/ten chinh thuc de TIM KIEM (khong dung de lay toa do)."""
    prompt = (
        f"Người dùng nói bằng giọng nói (có thể bị nhận dạng sai chính tả) một địa điểm ở Việt Nam: \"{text}\". "
        "Hãy đưa ra tối đa 3 cụm từ tìm kiếm khác nhau để tìm địa điểm này trên OpenStreetMap "
        "(tên chính thức hoặc tên thường gọi, kèm quận hoặc thành phố nếu biết). "
        "Chỉ trả về mỗi cụm trên một dòng, không giải thích, không đánh số."
    )
    try:
        resp = client.models.generate_content(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=200),
        )
        lines = [l.strip(" -•*").strip() for l in (resp.text or "").splitlines()]
        return [l for l in lines if l][:3]
    except Exception as e:
        print(f"[Gemini] không gợi ý được cách tìm: {e}")
        return []


def find_place_candidates(text: str, bias: tuple | None) -> list:
    """Thu nhieu cach tim; tra ve danh sach (lat, lng, ten) khong trung lap."""
    pt = parse_latlng(text)
    if pt:
        return [(pt[0], pt[1], text)]

    def search(queries):
        found = []
        for q in queries:
            for c in _nominatim_candidates(q, bias) or _photon_candidates(q, bias):
                if all(haversine_m(c[0], c[1], f[0], f[1]) > 30 for f in found):
                    found.append(c)
            if found:
                break
        return _sort_by_bias(found, bias)

    cands = search(_query_variants(text))
    if not cands:
        cands = search(_gemini_query_variants(text))
    return cands


def short_name(name: str) -> str:
    return ", ".join(name.split(", ")[:3])


def resolve_place(text: str, bias: tuple | None = None) -> tuple[float, float, str] | None:
    """Tim dia diem va XAC NHAN bang giong noi truoc khi dan duong (an toan)."""
    for attempt in range(2):
        for lat, lng, name in find_place_candidates(text, bias)[:3]:
            label = short_name(name)
            ans = ask_and_listen(f"Tôi tìm thấy {label}. Đúng không?")
            if ans and contains_any(ans, YES_WORDS) and not contains_any(ans, NO_WORDS):
                return lat, lng, label
        if attempt == 0:
            text = ask_and_listen(
                "Tôi chưa tìm đúng địa điểm. Bạn nói lại tên đường, số nhà và thành phố giúp tôi nhé."
            )
            if not text:
                break
            text = text.strip(" .,?!")
    speak("Xin lỗi, tôi vẫn chưa tìm được địa điểm đó.")
    return None


def handle_directions(text: str) -> None:
    destination_text = extract_after_keyword(text, ["đến", "tới", "chỉ đường đi"])
    if not destination_text:
        destination_text = ask_and_listen("Bạn muốn đi đến địa chỉ hoặc địa điểm nào?")
    if not destination_text:
        speak("Tôi chưa nghe rõ địa điểm bạn muốn đến, bạn thử lại nhé.")
        return

    speak("Đợi tôi một chút, tôi đang xác định vị trí của bạn.")
    loc = get_location()
    print(f"[GPS] Kết quả định vị: {loc}")  # xem nguon (gps/windows/ip) va sai so (met)
    bias = (loc["lat"], loc["lng"]) if loc else None

    dest = resolve_place(destination_text, bias)
    if not dest:
        return
    dlat, dlng, dlabel = dest
    destination = f"{dlat},{dlng}"

    # 1) GPS tot: dan duong truc tiep
    if location_is_good(loc):
        info = get_walking_directions(f"{loc['lat']},{loc['lng']}", destination)
        if "error" in info:
            speak(info["error"])
            return
        info["end_address"] = dlabel
        run_live_navigation(info, destination)
        return

    # 2) GPS kha (~100 m): lam diem xuat phat sau khi nguoi dung xac nhan, doc tung buoc
    if location_ok_for_start(loc):
        addr = reverse_geocode(loc["lat"], loc["lng"])
        if addr:
            ans = ask_and_listen(
                f"Tôi đoán bạn đang ở gần {addr}, sai số khoảng {loc['accuracy']} mét. Đúng không?"
            )
            if ans and contains_any(ans, YES_WORDS) and not contains_any(ans, NO_WORDS):
                info = get_walking_directions(f"{loc['lat']},{loc['lng']}", destination)
                if "error" in info:
                    speak(info["error"])
                    return
                info["end_address"] = dlabel
                speak("Vì sai số vị trí còn lớn nên tôi sẽ đọc từng bước, bạn nói 'tiếp' để nghe bước kế tiếp.")
                run_manual_navigation(info)
                return

    # 3) Khong co vi tri dung duoc: hoi vi tri bang giong noi
    speak("Tôi chưa có tín hiệu vị trí đủ chính xác để dẫn đường trực tiếp.")
    origin_text = ask_and_listen("Bạn đang ở đâu? Bạn có thể nói địa chỉ hoặc mô tả gần đúng vị trí.")
    if not origin_text:
        speak("Tôi chưa nghe rõ vị trí hiện tại của bạn, bạn thử lại nhé.")
        return
    origin = resolve_place(clean_place(origin_text), bias)
    if not origin:
        return
    info = get_walking_directions(f"{origin[0]},{origin[1]}", destination)
    if "error" in info:
        speak(info["error"])
        return
    info["end_address"] = dlabel
    run_manual_navigation(info)


# ======================================================================
#                      Gemini (tro chuyen chung + mo ta anh)
# ======================================================================
# [SỬA ĐỔI 4] Sửa lại tên model Fallback về bản ổn định (gemini-1.5-flash)
FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-1.5-flash").split(",") if m.strip()]


# [SỬA ĐỔI 5] Thêm bắt lỗi "504", "deadline", "timeout", "timed out" vào điều kiện thử lại
def _is_overloaded(msg: str) -> bool:
    msg_lower = msg.lower()
    keywords = (
        "503", "unavailable",
        "429", "resource_exhausted",
        "504", "deadline_exceeded",
        "overloaded", "high demand",
        "timed out", "timeout", "deadline"
    )
    return any(k in msg_lower for k in keywords)


def _generate_with_retry(contents, config):
    """Thu lai khi may chu Google qua tai (503/429/504); that bai thi chuyen sang model nhe hon."""
    last_err = None
    told = False
    for model in [MODEL] + FALLBACK_MODELS:
        for attempt in range(2):
            try:
                return client.models.generate_content(model=model, contents=contents, config=config)
            except Exception as e:
                last_err = e
                msg = str(e)
                print(f"[Gemini] {model} lần {attempt + 1}: {msg[:200]}")
                if "404" in msg or "NOT_FOUND" in msg:
                    break  # model khong ton tai -> thu model khac
                if not _is_overloaded(msg):
                    raise
                if not told:
                    speak("Hệ thống đang quá tải, tôi thử lại ngay.")
                    told = True
                time.sleep(1.5 * (attempt + 1))
    raise last_err


def ask_gemini(history: list, user_text: str, image: bytes | None = None) -> str:
    parts = []
    if image:
        parts.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=user_text))

    contents = history[-MAX_HISTORY:] + [types.Content(role="user", parts=parts)]
    speak("Đợi tôi chút nhé.")  # tranh de nguoi dung nghe im lang qua lau
    t0 = time.time()
    try:
        response = _generate_with_retry(
            contents,
            types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=800,
            ),
        )
    except Exception as e:
        msg = str(e)
        print(f"[Gemini] lỗi sau {time.time() - t0:.1f}s: {msg}")
        if "timed out" in msg.lower() or "timeout" in msg.lower() or "deadline" in msg.lower():
            return "Hệ thống trả lời quá chậm. Bạn thử hỏi lại nhé."
        if _is_overloaded(msg):
            return "Máy chủ trí tuệ nhân tạo đang quá tải hoặc bạn đã hỏi quá nhiều. Bạn đợi một lát rồi hỏi lại nhé."
        if "404" in msg or "NOT_FOUND" in msg:
            return "Tên mô hình Gemini không còn hợp lệ. Bạn cần đổi biến môi trường GEMINI_MODEL."
        return "Xin lỗi, tôi không kết nối được với trí tuệ nhân tạo lúc này."

    print(f"[Gemini] trả lời sau {time.time() - t0:.1f}s")
    reply = (response.text or "").strip() or "Xin lỗi, tôi chưa có câu trả lời cho việc này."

    # Chi luu van ban vao lich su (khong luu anh) de tiet kiem
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
    return reply


# ---------- Vong lap chinh ----------
def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("Chưa đặt biến môi trường GEMINI_API_KEY.")
    if ROUTING_PROVIDER == "google" and not GOOGLE_MAPS_API_KEY:
        print("[Cảnh báo] ROUTING_PROVIDER=google nhưng chưa đặt GOOGLE_MAPS_API_KEY.")

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
                image = capture_image_bytes()
                if image is None:
                    speak("Tôi không mở được camera. Bạn kiểm tra giúp nhé.")
                    continue
            speak(ask_gemini(history, text, image))
        except Exception as e:
            print(f"[Lỗi] {e}")
            speak("Xin lỗi, đã có lỗi xảy ra. Bạn thử lại nhé.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng.")
