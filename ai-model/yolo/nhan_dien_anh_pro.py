# -*- coding: utf-8 -*-
"""
Nhận diện vật thể trong ảnh bằng YOLO và đọc kết quả bằng tiếng Việt.

Cách chạy (trong thư mục ai-model/yolo):
    python nhan_dien_anh_pro.py

Yêu cầu thư viện:
    pip install ultralytics opencv-python edge-tts pygame pyttsx3
"""

import asyncio
import os
import time

import cv2
import edge_tts
import pygame
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 0. Cấu hình (có thể chỉnh)
# ----------------------------------------------------------------------
THU_MUC_GOC = os.path.dirname(os.path.abspath(__file__))
THU_MUC_ANH = os.path.join(THU_MUC_GOC, "images")
THU_MUC_KET_QUA = os.path.join(THU_MUC_GOC, "outputs")
FILE_AM_THANH = os.path.join(THU_MUC_KET_QUA, "giong_viet.mp3")

TEN_MO_HINH = "yolo11n.pt"      # tự tải về nếu chưa có
NGUONG_TIN_CAY = 0.5            # chỉ nhận vật thể có độ tin cậy >= 50%
GIONG_DOC = "vi-VN-HoaiMyNeural"

os.makedirs(THU_MUC_KET_QUA, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Bảng dịch tên vật thể sang tiếng Việt (80 lớp COCO)
# ----------------------------------------------------------------------
TEN_TIENG_VIET = {
    "person": "người", "bicycle": "xe đạp", "car": "ô tô",
    "motorcycle": "xe máy", "airplane": "máy bay", "bus": "xe buýt",
    "train": "tàu hỏa", "truck": "xe tải", "boat": "thuyền",
    "traffic light": "đèn giao thông", "fire hydrant": "trụ cứu hỏa",
    "stop sign": "biển báo dừng", "parking meter": "đồng hồ tính phí đỗ xe",
    "bench": "ghế băng", "bird": "chim", "cat": "mèo", "dog": "chó",
    "horse": "ngựa", "sheep": "cừu", "cow": "bò", "elephant": "voi",
    "bear": "gấu", "zebra": "ngựa vằn", "giraffe": "hươu cao cổ",
    "backpack": "ba lô", "umbrella": "ô", "handbag": "túi xách",
    "tie": "cà vạt", "suitcase": "va li", "frisbee": "đĩa ném",
    "skis": "ván trượt tuyết", "snowboard": "ván trượt tuyết đứng",
    "sports ball": "quả bóng", "kite": "diều", "baseball bat": "gậy bóng chày",
    "baseball glove": "găng bóng chày", "skateboard": "ván trượt",
    "surfboard": "ván lướt sóng", "tennis racket": "vợt tennis",
    "bottle": "chai", "wine glass": "ly rượu", "cup": "cốc",
    "fork": "dĩa", "knife": "dao", "spoon": "thìa", "bowl": "bát",
    "banana": "quả chuối", "apple": "quả táo", "sandwich": "bánh mì kẹp",
    "orange": "quả cam", "broccoli": "súp lơ xanh", "carrot": "cà rốt",
    "hot dog": "xúc xích kẹp bánh mì", "pizza": "bánh pizza",
    "donut": "bánh vòng", "cake": "bánh ngọt", "chair": "ghế",
    "couch": "ghế sofa", "potted plant": "chậu cây", "bed": "giường",
    "dining table": "bàn ăn", "toilet": "bồn cầu", "tv": "ti vi",
    "laptop": "máy tính xách tay", "mouse": "chuột máy tính",
    "remote": "điều khiển từ xa", "keyboard": "bàn phím",
    "cell phone": "điện thoại", "microwave": "lò vi sóng", "oven": "lò nướng",
    "toaster": "máy nướng bánh mì", "sink": "bồn rửa", "refrigerator": "tủ lạnh",
    "book": "quyển sách", "clock": "đồng hồ", "vase": "lọ hoa",
    "scissors": "cái kéo", "teddy bear": "gấu bông",
    "hair drier": "máy sấy tóc", "toothbrush": "bàn chải đánh răng",
}


# ----------------------------------------------------------------------
# 2. Đọc văn bản bằng giọng tiếng Việt
# ----------------------------------------------------------------------
async def _tao_file_giong(cau_noi):
    giong = edge_tts.Communicate(text=cau_noi, voice=GIONG_DOC, rate="+0%")
    await giong.save(FILE_AM_THANH)


def _doc_online(cau_noi):
    """Giọng neural của edge-tts (cần mạng)."""
    asyncio.run(_tao_file_giong(cau_noi))
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(FILE_AM_THANH)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    finally:
        pygame.mixer.quit()


def _doc_offline(cau_noi):
    """Giọng có sẵn trong máy (dự phòng khi mất mạng)."""
    import pyttsx3
    may_doc = pyttsx3.init()
    for giong in may_doc.getProperty("voices"):
        ten = (giong.name or "").lower()
        if "viet" in ten or "vi-vn" in ten:
            may_doc.setProperty("voice", giong.id)
            break
    may_doc.say(cau_noi)
    may_doc.runAndWait()


def doc_tieng_viet(cau_noi):
    try:
        _doc_online(cau_noi)
    except Exception as loi:
        print("Không dùng được giọng online:", loi)
        print("Chuyển sang giọng offline (có thể không phải tiếng Việt).")
        try:
            _doc_offline(cau_noi)
        except Exception as loi2:
            print("Không đọc được giọng nói:", loi2)


# ----------------------------------------------------------------------
# 3. Chương trình chính
# ----------------------------------------------------------------------
def main():
    print("Đang tải mô hình nhận diện...")
    model = YOLO(TEN_MO_HINH)
    print("Đã tải mô hình thành công!")

    if not os.path.isdir(THU_MUC_ANH):
        print("Không tìm thấy thư mục ảnh:", THU_MUC_ANH)
        return

    danh_sach_anh = sorted(
        os.path.join(THU_MUC_ANH, ten)
        for ten in os.listdir(THU_MUC_ANH)
        if ten.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not danh_sach_anh:
        print("Không tìm thấy ảnh trong thư mục images.")
        return
    print("Tìm thấy", len(danh_sach_anh), "ảnh.")

    for duong_dan in danh_sach_anh:
        ten_file = os.path.basename(duong_dan)
        print("\nĐang nhận diện ảnh:", ten_file)

        ket_qua = model.predict(source=duong_dan, conf=NGUONG_TIN_CAY, verbose=False)[0]

        ten_da_thay = []
        for vat_the in ket_qua.boxes:
            ten_goc = model.names[int(vat_the.cls[0])]
            ten_viet = TEN_TIENG_VIET.get(ten_goc, ten_goc)
            do_tin_cay = float(vat_the.conf[0]) * 100
            ten_da_thay.append(ten_viet)
            print(f"  Vật thể: {ten_viet} - Độ tin cậy: {do_tin_cay:.1f}%")

        # Lưu ảnh đã vẽ khung
        anh_ve = ket_qua.plot()
        ten_ket_qua = os.path.splitext(ten_file)[0] + "_ket_qua.jpg"
        cv2.imwrite(os.path.join(THU_MUC_KET_QUA, ten_ket_qua), anh_ve)

        # Tạo câu nói: đếm số lượng từng loại, ví dụ "3 người, 2 xe máy"
        if ten_da_thay:
            dem = {}
            for ten in ten_da_thay:
                dem[ten] = dem.get(ten, 0) + 1
            cau_noi = "Phát hiện " + ", ".join(f"{so} {ten}" for ten, so in dem.items())
        else:
            cau_noi = "Không phát hiện vật thể nào trong ảnh."
        print("Thông báo:", cau_noi)

        # Hiện ảnh trước, rồi mới đọc
        cv2.imshow("Ket qua nhan dien YOLO", anh_ve)
        cv2.waitKey(1)
        doc_tieng_viet(cau_noi)

        print("Nhấn một phím bất kỳ trên cửa sổ ảnh để sang ảnh tiếp theo.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nĐã nhận diện xong tất cả ảnh.")


if __name__ == "__main__":
    main()
