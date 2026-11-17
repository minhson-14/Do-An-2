
import base64
import html
import io
import os
import re
import sys

import cv2
import pygame
import requests
import speech_recognition as sr
from anthropic import Anthropic
from gtts import gTTS

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
MAX_HISTORY = 10  # so luot hoi thoai gan nhat duoc nho
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

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
DIRECTIONS_WORDS = [
    "chỉ đường", "dẫn đường", "đường đi đến", "đường đến", "đi đến",
    "làm sao để đến", "làm sao đi đến", "chỉ giúp tôi đường",
]
NAV_NEXT_WORDS = ["tiếp", "tiếp theo", "rồi sao", "sau đó"]
NAV_REPEAT_WORDS = ["nhắc lại", "lặp lại", "nói lại"]
NAV_STOP_WORDS = ["dừng chỉ đường", "hủy chỉ đường", "thôi khỏi chỉ đường", "dừng dẫn đường"]

client = Anthropic()  # tu doc ANTHROPIC_API_KEY tu bien moi truong
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
    """Doc to van ban bang giong tieng Viet."""
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
    """Am 'bip' ngan bao hieu bat dau nghe (khong can nhin man hinh)."""
    import array
    import math

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
    """Nghe tu micro va tra ve van ban tieng Viet (hoac None neu khong nghe duoc)."""
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
    """Hoi mot cau va cho nguoi dung tra loi bang giong noi."""
    speak(question)
    return listen(timeout=10, phrase_time_limit=12)


def contains_any(text: str, words: list[str]) -> bool:
    text = text.lower()
    return any(w in text for w in words)


def extract_after_keyword(text: str, keywords: list[str]) -> str | None:
    """Tim phan dia danh/dia chi sau cac tu khoa nhu 'o', 'tai', 'den', 'toi'."""
    for kw in keywords:
        m = re.search(rf"{kw}\s+(.+)", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,?!")
    return None


# ---------- Camera ----------
def capture_image_b64(max_side: int = 1024) -> str | None:
    """Chup mot anh tu webcam, tra ve chuoi base64 (JPEG)."""
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


# ---------- Thoi tiet (Open-Meteo, khong can API key) ----------
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
        lat, lon = loc["latitude"], loc["longitude"]
        ten = loc.get("name", place)

        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
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


# ---------- Chi duong (Google Maps Directions API) ----------
def strip_html(raw: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", raw)).strip()


def get_walking_directions(origin: str, destination: str) -> dict:
    """Goi Google Directions API, tra ve dict co 'error' hoac 'steps' + tong quan."""
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
        {"instruction": strip_html(s["html_instructions"]), "distance": s["distance"]["text"]}
        for s in leg["steps"]
    ]
    return {
        "steps": steps,
        "total_distance": leg["distance"]["text"],
        "total_duration": leg["duration"]["text"],
        "start_address": leg.get("start_address", origin),
        "end_address": leg.get("end_address", destination),
    }


def run_navigation(origin: str, destination: str) -> None:
    info = get_walking_directions(origin, destination)
    if "error" in info:
        speak(info["error"])
        return

    speak(
        f"Đã tìm thấy đường đi bộ đến {info['end_address']}. "
        f"Tổng quãng đường khoảng {info['total_distance']}, mất khoảng {info['total_duration']}. "
        f"Tôi sẽ đọc từng bước một, bạn nói 'tiếp' để nghe bước kế tiếp, "
        f"'nhắc lại' để nghe lại, hoặc 'dừng chỉ đường' để thoát."
    )

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
            continue  # doc lai buoc hien tai (khong tang i)
        if contains_any(reply, NAV_NEXT_WORDS):
            i += 1
            continue
        # Neu nguoi dung noi gi khac, mac dinh coi la muon sang buoc tiep theo
        i += 1

    speak(f"Bạn đã đến nơi: {info['end_address']}. Chúc bạn một ngày tốt lành.")


def handle_directions(text: str) -> None:
    destination = extract_after_keyword(text, ["đến", "tới", "chỉ đường đi"])
    if not destination:
        destination = ask_and_listen("Bạn muốn đi đến địa chỉ hoặc địa điểm nào?")
    if not destination:
        speak("Tôi chưa nghe rõ địa điểm bạn muốn đến, bạn thử lại nhé.")
        return

    origin = ask_and_listen("Bạn đang ở đâu? Bạn có thể nói địa chỉ hoặc mô tả gần đúng vị trí.")
    if not origin:
        speak("Tôi chưa nghe rõ vị trí hiện tại của bạn, bạn thử lại nhé.")
        return

    run_navigation(origin, destination)


# ---------- Claude (tro chuyen chung + mo ta anh) ----------
def ask_claude(history: list, user_text: str, image_b64: str | None = None) -> str:
    content = []
    if image_b64:
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
            }
        )
    content.append({"type": "text", "text": user_text})

    messages = history[-MAX_HISTORY:] + [{"role": "user", "content": content}]
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
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
        "Bạn có thể hỏi thời tiết, nhờ tôi chỉ đường, hoặc nhờ tôi mô tả xung quanh. "
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
