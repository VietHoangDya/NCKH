# -*- coding: utf-8 -*-
"""
Nhận diện vật thể THỜI GIAN THỰC từ video / webcam / camera điện thoại,
báo HƯỚNG kiểu đồng hồ, KHOẢNG CÁCH và CHỈ ĐƯỜNG tránh vật cản bằng giọng tiếng Việt.

Điểm mới (v2):
  - Giọng nói tạo sẵn thành các đoạn ngắn lưu trong thư mục giong_cache/, khi đọc chỉ
    ghép lại => gần như không trễ, và đọc nhanh hơn (tham số --toc-do).
  - Video file được đồng bộ theo thời gian thực: máy chậm sẽ tự bỏ bớt khung hình.
  - Chỉ đường: tìm khe hở an toàn rộng nhất gần hướng thẳng; nếu bị chặn hẳn thì
    chỉ nên xoay người sang bên nào. Lời chỉ hướng cũng có trong cảnh báo "Dừng lại".
  - Mũi tên chỉ hướng đi hiện trên màn hình.

Cách chạy (trong ai-model/yolo):
    python camera_pro.py --nguon video\\cua_nha.mp4
    python camera_pro.py --nguon 0
    python camera_pro.py --ip 192.168.1.10          (camera điện thoại, app IP Webcam)

Lần chạy ĐẦU TIÊN sẽ mất 1-3 phút để tạo thư viện giọng nói (cần Internet), các lần sau
chạy ngay.  Phím: q/ESC thoát, m bật/tắt giọng, c hiệu chuẩn khoảng cách.
"""

import argparse
import asyncio
import hashlib
import math
import os
import queue
import subprocess
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tu_dien_viet import TEN_TIENG_VIET

# ----------------------------------------------------------------------
# 0. Cấu hình
# ----------------------------------------------------------------------
THU_MUC_GOC = os.path.dirname(os.path.abspath(__file__))
THU_MUC_GIONG = os.path.join(THU_MUC_GOC, "giong_cache")

NGUONG_TIN_CAY = 0.45
GIONG_DOC = "vi-VN-HoaiMyNeural"
TOC_DO_GIONG = "-40%"            # tốc độ đọc (đổi bằng --toc-do)

FOV_NGANG_NGANG = 66.0           # độ, camera điện thoại cầm ngang
FOV_NGANG_DOC = 52.0             # độ, cầm dọc

TAM_XA_BAO = 6.0                 # chỉ chủ động đọc vật trong vòng 6 m
NGUONG_CAN = 2.5                 # vật gần hơn mức này (m) được coi là chắn đường
NGUONG_KHAN_CAP = 1.5            # gần hơn mức này và ở phía trước => "Dừng lại"
GOC_PHIA_TRUOC = 17.0
TOC_DO_LAI_GAN = 0.8             # m/s

SO_LAN_TOI_DA = 2                # mỗi cảnh báo / mỗi vật chỉ đọc tối đa số lần này
GIAN_CACH_NOI = 2.5              # giây, tối thiểu giữa 2 lần đọc thường
NHAC_LAI_SAU = 10.0              # giây, nhắc lại cùng một vật
NHAC_LAI_LOAI = 6.0              # giây, nhắc lại cùng (loại, giờ)
NHAC_HUONG_SAU = 6.0             # giây, nhắc lại chỉ đường khi vẫn còn vật cản

CHIEU_CAO_THAT = {
    "person": 1.70, "bicycle": 1.00, "motorcycle": 1.10, "car": 1.50,
    "bus": 3.00, "truck": 3.00, "traffic light": 0.90, "stop sign": 0.75,
    "fire hydrant": 0.70, "bench": 0.85, "dog": 0.50, "cat": 0.25,
    "backpack": 0.50, "suitcase": 0.60, "umbrella": 0.90, "bottle": 0.25,
    "chair": 0.90, "couch": 0.85, "potted plant": 0.50, "bed": 0.55,
    "dining table": 0.75, "tv": 0.50, "laptop": 0.25, "refrigerator": 1.70,
    "oven": 0.60, "microwave": 0.30, "sink": 0.20, "toilet": 0.40,
    "cell phone": 0.15, "cup": 0.10, "book": 0.22,
}

VAT_NHO = {
    "bottle", "cup", "fork", "knife", "spoon", "bowl", "cell phone", "remote",
    "book", "mouse", "keyboard", "toothbrush", "scissors", "wine glass",
    "tie", "sports ball", "frisbee", "banana", "apple", "orange", "clock",
    "vase", "hair drier",
}


TIMEOUT_MOT_DOAN = 10.0          # giây chờ tối đa cho 1 đoạn giọng (tránh treo)
TONG_THOI_GIAN_TAO = 240.0       # giây, tổng thời gian tối đa tạo thư viện giọng
KHOANG_LANG_NGAN = 0.15          # giây nghỉ giữa các cụm trong cùng một ý
KHOANG_LANG_DAI = 0.50           # giây nghỉ giữa hai ý (chỉ đường / từng vật thể)
SO_LUONG_SONG_SONG = 3           # số yêu cầu tạo giọng chạy cùng lúc
THU_LAI_SAU = 60.0               # giây, thử lại đoạn giọng đã thất bại

# ----------------------------------------------------------------------
# 1. Hình học
# ----------------------------------------------------------------------
def tieu_cu_pixel(rong_anh, fov_ngang_do):
    return (rong_anh / 2.0) / math.tan(math.radians(fov_ngang_do) / 2.0)


def goc_ngang(x, rong_anh, f):
    return math.degrees(math.atan((x - rong_anh / 2.0) / f))


def goc_sang_gio(goc_do):
    gio = int(round(goc_do / 30.0)) % 12
    return 12 if gio == 0 else gio


def uoc_luong_khoang_cach(ten_goc, chieu_cao_px, f):
    h_that = CHIEU_CAO_THAT.get(ten_goc)
    if h_that is None or chieu_cao_px < 5:
        return None
    return f * h_that / chieu_cao_px


def noi_khoang_cach(d):
    if d < 1.0:
        return "dưới 1 mét"
    if d > 8.0:
        return "hơn 8 mét"
    m = round(d * 2) / 2
    nguyen = int(m)
    return f"khoảng {nguyen} mét rưỡi" if m - nguyen else f"khoảng {nguyen} mét"


def le_an_toan(d):
    """Góc chừa thêm hai bên vật cản: vật càng gần thì càng phải tránh rộng."""
    if d is None:
        return 12.0
    return min(35.0, math.degrees(math.atan(0.40 / max(d, 0.5))) + 3.0)


# ----------------------------------------------------------------------
# 2. Các cụm từ giọng nói (mỗi cụm = 1 đoạn âm thanh tạo sẵn)
# ----------------------------------------------------------------------
def cum_gio(gio):
    return f"hướng {gio} giờ"


def cum_cach(d):
    return "cách " + noi_khoang_cach(d)


def cum_re(goc):
    ben = "trái" if goc < 0 else "phải"
    return f"Hãy đi chếch nhẹ sang {ben}" if abs(goc) < 15 else f"Hãy đi sang {ben}"


def cum_xoay(ben):
    return f"Hãy xoay người sang {ben}"


CAC_CUM_CO_DINH = [
    "Dừng lại", "ngay phía trước", "Cẩn thận", "đang tiến nhanh lại gần",
    "đang tiến lại gần", "Phía trước có vật cản",
    "Hãy đi chếch nhẹ sang trái", "Hãy đi chếch nhẹ sang phải",
    "Hãy đi sang trái", "Hãy đi sang phải",
    "Phía trước bị chặn, hãy dừng lại",
    "Hãy xoay người sang trái", "Hãy xoay người sang phải",
    "Phía trước đã thông thoáng, có thể đi thẳng",
    "Hệ thống đã sẵn sàng",
]


def tat_ca_cum_tu():
    cum = set(TEN_TIENG_VIET.values())
    cum |= {cum_gio(g) for g in range(1, 13)}
    cum.add("cách dưới 1 mét")
    cum.add("cách hơn 8 mét")
    for n in range(1, 9):
        cum.add(f"cách khoảng {n} mét")
        if n < 8:
            cum.add(f"cách khoảng {n} mét rưỡi")
    cum |= set(CAC_CUM_CO_DINH)
    return sorted(cum)


# ----------------------------------------------------------------------
# 3. Bộ phân tích
# ----------------------------------------------------------------------
@dataclass
class PhatHien:
    khoa: object
    ten_goc: str
    ten: str
    do_tin_cay: float
    x1: float
    y1: float
    x2: float
    y2: float
    goc: float
    gio: int
    khoang_cach: "float | None"
    bi_cat_khung: bool
    toc_do_lai_gan: float
    dien_tich: float
    vat_nho: bool


@dataclass
class TheoDoi:
    d: "float | None" = None
    lich_su: deque = field(default_factory=lambda: deque(maxlen=40))
    thay_lan_cuoi: float = 0.0
    lan_noi: "float | None" = None
    gio_da_noi: "int | None" = None
    d_da_noi: "float | None" = None
    lan_khan_cap: float = -1e9
    lan_dung: float = -1e9
    so_lan_dung: int = 0
    so_lan_nhanh: int = 0
    so_lan_noi: int = 0


class BoPhanTich:
    def __init__(self, rong, cao, fov_ngang, tieu_cu=None):
        self.rong, self.cao, self.fov = rong, cao, fov_ngang
        self.f = tieu_cu or tieu_cu_pixel(rong, fov_ngang)
        self.theo_doi = {}
        self.da_noi_loai = {}
        self.trang_thai_cu = None
        self.lan_noi_huong = -1e9
        self.lan_noi_cuoi = -1e9
        self.so_lan_huong = {}           # (loại, bên) -> số lần đã đọc lời chỉ đường
        self.lan_dung_loai = {}          # loại vật -> lần cuối đọc "Dừng lại"
        self.t_thang_tu = None           # từ lúc nào đường phía trước thông liên tục
        self.cho_bao_thong = False       # đã báo vật cản, còn nợ lời "đã thông thoáng"

    # ---- 3.1 phát hiện -> PhatHien
    def cap_nhat(self, ds_tho, t):
        ds = []
        for (x1, y1, x2, y2, ten_goc, conf, id_td) in ds_tho:
            h_px = y2 - y1
            goc = goc_ngang((x1 + x2) / 2.0, self.rong, self.f)
            gio = goc_sang_gio(goc)
            d = uoc_luong_khoang_cach(ten_goc, h_px, self.f)

            cat = (y2 >= self.cao - 3) or (y1 <= 2)
            if d is not None and cat and h_px >= 0.85 * self.cao:
                d = min(d, NGUONG_KHAN_CAP - 0.1)

            khoa = id_td if id_td is not None else ("n", ten_goc, gio)
            ts = self.theo_doi.setdefault(khoa, TheoDoi())
            ts.thay_lan_cuoi = t
            if d is not None:
                ts.d = d if ts.d is None else 0.6 * ts.d + 0.4 * d
                ts.lich_su.append((t, ts.d))

            ds.append(PhatHien(
                khoa=khoa, ten_goc=ten_goc,
                ten=TEN_TIENG_VIET.get(ten_goc, ten_goc), do_tin_cay=conf,
                x1=x1, y1=y1, x2=x2, y2=y2, goc=goc, gio=gio,
                khoang_cach=ts.d if d is not None else None,
                bi_cat_khung=cat, toc_do_lai_gan=self._toc_do(ts),
                dien_tich=((x2 - x1) * h_px) / (self.rong * self.cao),
                vat_nho=ten_goc in VAT_NHO,
            ))
        for k in [k for k, v in self.theo_doi.items() if t - v.thay_lan_cuoi > 10]:
            del self.theo_doi[k]
        return ds

    @staticmethod
    def _toc_do(ts):
        if len(ts.lich_su) < 2:
            return 0.0
        t_moi, d_moi = ts.lich_su[-1]
        for t_cu, d_cu in ts.lich_su:
            if t_moi - t_cu <= 1.5:
                dt = t_moi - t_cu
                return (d_cu - d_moi) / dt if dt >= 0.5 else 0.0
        return 0.0

    # ---- 3.2 chỉ đường
    def goi_y_huong_di(self, ds):
        """Trả về (loại, góc_độ, bên):
             ("THANG", 0, None)       đi thẳng được
             ("RE", góc, None)        đi theo góc này (âm = trái, dương = phải)
             ("CHAN", None, "trái"/"phải")  bị chặn hết, nên xoay người sang bên đó."""
        vung = []
        for p in ds:
            if p.vat_nho:
                continue
            gan = (p.khoang_cach is not None and p.khoang_cach <= NGUONG_CAN) \
                or (p.khoang_cach is None and p.dien_tich > 0.15)
            if gan:
                le = le_an_toan(p.khoang_cach)
                vung.append((goc_ngang(p.x1, self.rong, self.f) - le,
                             goc_ngang(p.x2, self.rong, self.f) + le, p))

        def thong(g):
            return all(not (a <= g <= b) for a, b, _ in vung)

        if thong(0.0):
            return ("THANG", 0.0, None)

        gioi_han = int(self.fov / 2.0) - 2
        trong = [g for g in range(-gioi_han, gioi_han + 1, 2) if thong(float(g))]
        cac_khe = []
        for g in trong:
            if cac_khe and g - cac_khe[-1][-1] <= 2:
                cac_khe[-1].append(g)
            else:
                cac_khe.append([g])
        tot_nhat = None
        for khe in cac_khe:
            if khe[-1] - khe[0] < 4:           # khe quá hẹp
                continue
            muc_tieu = (khe[0] + khe[-1]) / 2.0     # đi vào GIỮA khe cho an toàn
            if tot_nhat is None or abs(muc_tieu) < abs(tot_nhat):
                tot_nhat = muc_tieu
        if tot_nhat is not None:
            return ("RE", float(tot_nhat), None)

        # bị chặn hết: xoay về phía có vật cản ở xa hơn
        def gan_nhat(dau):
            ds_d = [(p.khoang_cach if p.khoang_cach is not None else 1.0)
                    for _, _, p in vung if (p.goc < 0) == (dau < 0)]
            return min(ds_d) if ds_d else 99.0
        ben = "trái" if gan_nhat(-1) > gan_nhat(1) else "phải"
        return ("CHAN", None, ben)

    @staticmethod
    def _cum_chi_huong(goi_y):
        loai, goc, ben = goi_y
        if loai == "RE":
            return [cum_re(goc), cum_gio(goc_sang_gio(goc))]
        if loai == "CHAN":
            return [cum_xoay(ben)]
        return []

    # ---- 3.3 tạo thông báo (danh sách cụm từ)
    def tao_thong_bao(self, ds, goi_y, t, dang_ban):
        """Trả về (list_cụm_từ, khẩn_cấp) hoặc None.
        Mỗi cảnh báo chỉ đọc tối đa SO_LAN_TOI_DA lần để không gây nhiễu."""
        loai, goc, ben = goi_y
        bên_re = ("trái" if (goc or 0) < 0 else "phải") if loai == "RE" else ben
        khoa_huong = (loai, bên_re)

        # đường thông liên tục >= 3 giây => coi như vật cản đã qua, lần sau báo lại từ đầu
        if loai == "THANG":
            if self.t_thang_tu is None:
                self.t_thang_tu = t
        else:
            self.t_thang_tu = None
        thong_lau = self.t_thang_tu is not None and t - self.t_thang_tu >= 3.0
        if thong_lau:
            self.so_lan_huong.clear()

        # (a) khẩn cấp: vật cản gần ngay phía trước, KÈM lời chỉ hướng (tối đa 2 lần/vật)
        khan = []
        for p in ds:
            if p.vat_nho or p.khoang_cach is None or abs(p.goc) > GOC_PHIA_TRUOC:
                continue
            ts = self.theo_doi.get(p.khoa)
            if p.khoang_cach <= NGUONG_KHAN_CAP:
                if (ts.so_lan_dung < SO_LAN_TOI_DA and t - ts.lan_dung >= 3.0
                        and t - self.lan_dung_loai.get(p.ten_goc, -1e9) >= 4.0):
                    khan.append((p.khoang_cach, p, "dung",
                                 ["Dừng lại", p.ten, "ngay phía trước", cum_cach(p.khoang_cach)]))
            elif (p.toc_do_lai_gan >= 1.2 and p.khoang_cach <= 3.0
                  and ts.so_lan_nhanh < SO_LAN_TOI_DA and t - ts.lan_khan_cap >= 3.0):
                khan.append((p.khoang_cach, p, "nhanh",
                             ["Cẩn thận", p.ten, "đang tiến nhanh lại gần"]))
        if khan:
            khan.sort(key=lambda x: x[0])
            _, p, kieu, cum = khan[0]
            ts = self.theo_doi[p.khoa]
            if kieu == "dung":
                ts.lan_dung, ts.so_lan_dung = t, ts.so_lan_dung + 1
                self.lan_dung_loai[p.ten_goc] = t
            else:
                ts.lan_khan_cap, ts.so_lan_nhanh = t, ts.so_lan_nhanh + 1
            ts.lan_noi, ts.gio_da_noi, ts.d_da_noi = t, p.gio, p.khoang_cach
            ts.so_lan_noi = max(ts.so_lan_noi, 1)
            self.da_noi_loai[(p.ten_goc, p.gio)] = t
            self.lan_noi_cuoi = t
            self.lan_noi_huong = t
            self.trang_thai_cu = khoa_huong
            self.so_lan_huong[khoa_huong] = self.so_lan_huong.get(khoa_huong, 0) + 1
            self.cho_bao_thong = loai != "THANG" or self.cho_bao_thong
            huong = self._cum_chi_huong(goi_y)
            return (cum + ["|"] + huong if huong else cum), True

        # (b) thông báo thường
        if dang_ban or t - self.lan_noi_cuoi < GIAN_CACH_NOI:
            return None

        cac_cum = []
        if loai == "THANG":
            if thong_lau and self.cho_bao_thong:
                cac_cum = ["Phía trước đã thông thoáng, có thể đi thẳng"]
                self.cho_bao_thong = False
                self.lan_noi_huong = t
                self.trang_thai_cu = khoa_huong
        else:
            da_doc = self.so_lan_huong.get(khoa_huong, 0)
            doi = khoa_huong != self.trang_thai_cu
            den_luot = (t - self.lan_noi_huong >= 3.0) if doi else (t - self.lan_noi_huong >= NHAC_HUONG_SAU)
            if da_doc < SO_LAN_TOI_DA and den_luot:
                dau = ("Phía trước có vật cản" if loai == "RE"
                       else "Phía trước bị chặn, hãy dừng lại")
                cac_cum = [dau, "|"] + self._cum_chi_huong(goi_y)
                self.so_lan_huong[khoa_huong] = da_doc + 1
                self.cho_bao_thong = True
                self.lan_noi_huong = t
                self.trang_thai_cu = khoa_huong

        gioi_han_vat = 1 if cac_cum else 2
        ung_vien = sorted(ds, key=lambda p: (p.khoang_cach is None, p.khoang_cach or 0.0))
        so_vat = 0
        for p in ung_vien:
            if so_vat >= gioi_han_vat:
                break
            if p.khoang_cach is not None and p.khoang_cach > TAM_XA_BAO:
                continue
            if p.khoang_cach is None and p.dien_tich < 0.05:
                continue
            ts = self.theo_doi[p.khoa]
            if ts.so_lan_noi >= SO_LAN_TOI_DA:
                continue
            moi = ts.so_lan_noi == 0
            doi_nhieu = (ts.lan_noi is not None and t - ts.lan_noi >= NHAC_LAI_SAU and (
                ts.gio_da_noi != p.gio or
                (p.khoang_cach is not None and ts.d_da_noi is not None
                 and abs(p.khoang_cach - ts.d_da_noi) >= 1.0)))
            if not (moi or doi_nhieu):
                continue
            if t - self.da_noi_loai.get((p.ten_goc, p.gio), -1e9) < NHAC_LAI_LOAI:
                ts.lan_noi, ts.gio_da_noi, ts.d_da_noi = t, p.gio, p.khoang_cach
                ts.so_lan_noi += 1
                continue
            cum = [p.ten, cum_gio(p.gio)]
            if p.khoang_cach is not None:
                cum.append(cum_cach(p.khoang_cach))
            if p.toc_do_lai_gan >= TOC_DO_LAI_GAN:
                cum.append("đang tiến lại gần")
            if cac_cum:
                cac_cum.append("|")
            cac_cum += cum
            ts.lan_noi, ts.gio_da_noi, ts.d_da_noi = t, p.gio, p.khoang_cach
            ts.so_lan_noi += 1
            self.da_noi_loai[(p.ten_goc, p.gio)] = t
            so_vat += 1

        if not cac_cum:
            return None
        self.lan_noi_cuoi = t
        return cac_cum, False


# ----------------------------------------------------------------------
# 4. Giọng nói: tạo sẵn từng cụm từ, khi đọc chỉ ghép lại (rất nhanh)
# ----------------------------------------------------------------------
def van_ban_doc(cum):
    return cum.replace("!", ".").strip()


def duong_dan_clip(cum):
    ma = hashlib.md5(f"{GIONG_DOC}|{TOC_DO_GIONG}|{cum}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(THU_MUC_GIONG, ma + ".mp3")


def tao_thu_vien_giong(cac_cum):
    """Tạo các đoạn giọng còn thiếu, lặp tối đa 3 vòng cho đến khi đủ hoặc hết thời gian.
    Đoạn lỗi/treo bị bỏ qua, chương trình vẫn chạy được."""
    os.makedirs(THU_MUC_GIONG, exist_ok=True)

    def danh_sach_thieu():
        return [c for c in cac_cum if not os.path.exists(duong_dan_clip(c))]

    tong = len(cac_cum)
    thieu = danh_sach_thieu()
    print(f"Thư viện giọng: có sẵn {tong - len(thieu)}/{tong} đoạn.")
    if not thieu:
        return True
    print(f"Đang tạo {len(thieu)} đoạn còn thiếu "
          f"(tối đa ~{int(TONG_THOI_GIAN_TAO)} giây, cần Internet)...")
    import edge_tts
    han = time.monotonic() + TONG_THOI_GIAN_TAO

    for vong in range(1, 4):
        thieu = danh_sach_thieu()
        con_lai = han - time.monotonic()
        if not thieu or con_lai < 5:
            break
        dem = {"ok": 0, "xong": 0}

        async def mot(cum, sem, thieu=thieu, dem=dem):
            dd = duong_dan_clip(cum)
            tam = dd + ".tmp"
            try:
                async with sem:
                    for _ in range(2):
                        try:
                            await asyncio.wait_for(
                                edge_tts.Communicate(text=van_ban_doc(cum), voice=GIONG_DOC,
                                                     rate=TOC_DO_GIONG).save(tam),
                                timeout=TIMEOUT_MOT_DOAN)
                            if os.path.getsize(tam) > 0:
                                os.replace(tam, dd)
                                dem["ok"] += 1
                                return
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            await asyncio.sleep(0.3)
            finally:
                dem["xong"] += 1
                if dem["xong"] % 10 == 0 or dem["xong"] == len(thieu):
                    print(f"  vòng {vong}: {dem['xong']}/{len(thieu)} "
                          f"(thành công {dem['ok']})")
                if os.path.exists(tam):
                    try:
                        os.remove(tam)
                    except OSError:
                        pass

        async def tat_ca(thieu=thieu, con_lai=con_lai):
            sem = asyncio.Semaphore(SO_LUONG_SONG_SONG)
            await asyncio.wait_for(
                asyncio.gather(*(mot(c, sem) for c in thieu), return_exceptions=True),
                timeout=con_lai)

        try:
            asyncio.run(tat_ca())
        except (asyncio.TimeoutError, TimeoutError):
            print("[Giọng] Hết thời gian chờ ở vòng này.")
        except Exception as loi:
            print("[Giọng] Lỗi khi tạo thư viện giọng:", repr(loi))

    con_thieu = danh_sach_thieu()
    print(f"Thư viện giọng: {tong - len(con_thieu)}/{tong} đoạn.")
    if con_thieu:
        print(f"[Giọng] Còn thiếu {len(con_thieu)} đoạn: {', '.join(con_thieu[:6])}"
              f"{'...' if len(con_thieu) > 6 else ''}.")
        print("[Giọng] Những câu có đoạn thiếu sẽ bị bỏ qua. Hãy kiểm tra Internet rồi "
              "chạy lại để tạo nốt (các đoạn đã có được giữ nguyên).")
    return not con_thieu


def tao_clip_tien_trinh(cum):
    """Dự phòng: tạo 1 đoạn còn thiếu lúc đang chạy."""
    dd = duong_dan_clip(cum)
    tam = dd + ".tmp"
    try:
        kq = subprocess.run(
            [sys.executable, "-m", "edge_tts", "--voice", GIONG_DOC,
             f"--rate={TOC_DO_GIONG}", "--text", van_ban_doc(cum), "--write-media", tam],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if kq.returncode == 0 and os.path.exists(tam) and os.path.getsize(tam) > 0:
            os.replace(tam, dd)
            return True
    except Exception:
        pass
    return False


def cat_lang(mang, tan_so, nguong=300):
    """Cắt khoảng lặng đầu/cuối đoạn âm để ghép liền mạch."""
    m = np.abs(mang.astype(np.int32)).max(axis=1)
    idx = np.flatnonzero(m > nguong)
    if idx.size == 0:
        return mang
    giu = int(0.015 * tan_so)
    return mang[max(0, idx[0] - giu): idx[-1] + 1 + giu]


def ghep_am_thanh(ds, tan_so, ngan=KHOANG_LANG_NGAN, dai=KHOANG_LANG_DAI):
    """ds gồm các mảng âm thanh và chuỗi "|" (= nghỉ dài giữa hai ý)."""
    mang_dau = next((m for m in ds if not isinstance(m, str)), None)
    if mang_dau is None:
        return np.zeros((1, 1), dtype=np.int16)
    kenh = mang_dau.shape[1]
    phan, nghi_dai = [], False
    for m in ds:
        if isinstance(m, str):
            nghi_dai = True
            continue
        if phan:
            giay = dai if nghi_dai else ngan
            phan.append(np.zeros((int(giay * tan_so), kenh), dtype=np.int16))
        nghi_dai = False
        phan.append(cat_lang(m, tan_so))
    return np.concatenate(phan, axis=0)


class NguoiDoc:
    """Đọc ở luồng riêng. Câu mới thay câu đang chờ; câu khẩn cắt câu đang đọc."""

    def __init__(self, bat=True):
        self.bat = bat
        self.hang_doi = queue.Queue(maxsize=1)
        self.dang_noi = threading.Event()
        self._am = {}
        self._pygame = None
        self._tan_so, self._kenh = 24000, 1
        self._loi_lien_tiep = 0
        self._da_bao_thieu = set()
        self._that_bai = {}              # cụm -> thời điểm thất bại gần nhất
        self._dang_tao = set()
        self._tao_nen = queue.Queue()
        if bat:
            try:
                import pygame
                pygame.mixer.init(frequency=24000, size=-16, channels=1, allowedchanges=0)
                tan_so, kich_thuoc, kenh = pygame.mixer.get_init()
                if kich_thuoc != -16:
                    raise RuntimeError("thiết bị âm thanh không hỗ trợ 16-bit")
                self._pygame, self._tan_so, self._kenh = pygame, tan_so, kenh
                threading.Thread(target=self._chay, daemon=True).start()
                threading.Thread(target=self._chay_tao_nen, daemon=True).start()
                print(f"[Giọng] Âm thanh sẵn sàng ({tan_so} Hz, {kenh} kênh).")
            except Exception as loi:
                print("[Giọng] Không khởi tạo được âm thanh:", repr(loi))
                self.bat = False

    def noi(self, cac_cum, khan_cap=False):
        if not self.bat:
            return
        if khan_cap:
            try:
                self._pygame.mixer.stop()
            except Exception:
                pass
        try:
            while True:
                self.hang_doi.get_nowait()
        except queue.Empty:
            pass
        self.hang_doi.put_nowait(cac_cum)

    def ban(self):
        return self.dang_noi.is_set() or not self.hang_doi.empty()

    def _chay_tao_nen(self):
        """Tạo dần các đoạn giọng còn thiếu ở nền để không làm khựng việc đọc."""
        while True:
            cum = self._tao_nen.get()
            if not os.path.exists(duong_dan_clip(cum)):
                if not tao_clip_tien_trinh(cum):
                    self._that_bai[cum] = time.monotonic()
            self._dang_tao.discard(cum)

    def _lay_clip(self, cum):
        if cum in self._am:
            return self._am[cum]
        dd = duong_dan_clip(cum)
        if not os.path.exists(dd):
            if (cum not in self._dang_tao
                    and time.monotonic() - self._that_bai.get(cum, -1e9) >= THU_LAI_SAU):
                self._dang_tao.add(cum)
                self._tao_nen.put(cum)
            if cum not in self._da_bao_thieu:
                self._da_bao_thieu.add(cum)
                print(f"[Giọng] Thiếu đoạn giọng '{cum}', bỏ qua câu này (đang tạo ở nền).")
            return None
        snd = self._pygame.mixer.Sound(dd)
        mang = np.frombuffer(snd.get_raw(), dtype=np.int16).reshape(-1, self._kenh)
        self._am[cum] = mang
        return mang

    def _chay(self):
        while True:
            cac_cum = self.hang_doi.get()
            self.dang_noi.set()
            try:
                cac_mang = [c if c == "|" else self._lay_clip(c) for c in cac_cum]
                if any(m is None for m in cac_mang):
                    continue                     # bỏ câu này, câu sau vẫn đọc bình thường
                mang = ghep_am_thanh(cac_mang, self._tan_so)
                snd = self._pygame.mixer.Sound(buffer=np.ascontiguousarray(mang).tobytes())
                kenh = snd.play()
                while kenh is not None and kenh.get_busy():
                    time.sleep(0.02)
            except Exception as loi:
                print("[Giọng] Lỗi khi đọc:", repr(loi))
            finally:
                self.dang_noi.clear()


# ----------------------------------------------------------------------
# 5. Vẽ kết quả
# ----------------------------------------------------------------------
_FONT = {}
_DUONG_DAN_FONT = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def bo_dau(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def lay_font(co):
    if co not in _FONT:
        _FONT[co] = None
        for dd in _DUONG_DAN_FONT:
            if os.path.exists(dd):
                _FONT[co] = ImageFont.truetype(dd, co)
                break
    return _FONT[co]


def ve_chu(khung, ds_chu):
    if not ds_chu:
        return khung
    anh = Image.fromarray(cv2.cvtColor(khung, cv2.COLOR_BGR2RGB))
    ve = ImageDraw.Draw(anh)
    for van_ban, (x, y), mau, co, nen in ds_chu:
        font = lay_font(co)
        if font is None:
            font, van_ban = ImageFont.load_default(), bo_dau(van_ban)
        if nen is not None:
            l, t, r, b = ve.textbbox((x, y), van_ban, font=font)
            ve.rectangle((l - 3, t - 2, r + 3, b + 2), fill=nen[::-1])
        ve.text((x, y), van_ban, font=font, fill=mau[::-1])
    return cv2.cvtColor(np.array(anh), cv2.COLOR_RGB2BGR)


def mau_theo_khoang_cach(d):
    if d is None:
        return (200, 200, 200)
    if d <= NGUONG_KHAN_CAP:
        return (0, 0, 255)
    if d <= NGUONG_CAN + 0.5:
        return (0, 165, 255)
    return (0, 200, 0)


def ve_ket_qua(khung, ds, goi_y, ph):
    cao, rong = khung.shape[:2]
    chu = []

    for goc_moc, nhan in ((-30, "11h"), (0, "12h"), (30, "1h")):
        if abs(goc_moc) <= ph.fov / 2:
            x = int(rong / 2 + ph.f * math.tan(math.radians(goc_moc)))
            cv2.line(khung, (x, cao - 38), (x, cao), (255, 255, 255), 2)
            chu.append((nhan, (x - 14, cao - 62), (255, 255, 255), 20, None))

    for p in sorted(ds, key=lambda q: -(q.khoang_cach or 0)):
        mau = mau_theo_khoang_cach(p.khoang_cach)
        cv2.rectangle(khung, (int(p.x1), int(p.y1)), (int(p.x2), int(p.y2)), mau, 2)
        nhan = f"{p.ten} · {p.gio}h"
        if p.khoang_cach is not None:
            nhan += f" · {p.khoang_cach:.1f}m"
        chu.append((nhan, (int(p.x1) + 2, max(2, int(p.y1) - 26)), (255, 255, 255), 18, mau))

    loai, goc, ben = goi_y
    goc_goc = (rong // 2, cao - 80)
    if loai == "THANG":
        dong, mau = "Đi thẳng được (12 giờ)", (0, 160, 0)
        cv2.arrowedLine(khung, goc_goc, (rong // 2, cao - 200), (0, 220, 0), 8, tipLength=0.35)
    elif loai == "RE":
        dong, mau = (f"Vật cản! {cum_re(goc).replace('Hãy đi', 'Đi')}, "
                     f"{goc_sang_gio(goc)} giờ"), (0, 140, 255)
        x_dich = int(rong / 2 + ph.f * math.tan(math.radians(goc)))
        cv2.arrowedLine(khung, goc_goc, (x_dich, cao - 200), (0, 255, 255), 8, tipLength=0.35)
    else:
        dong, mau = f"DỪNG LẠI - xoay người sang {ben}", (0, 0, 220)
    chu.append((dong, (10, 8), (255, 255, 255), max(16, min(28, rong // 22)), mau))
    return ve_chu(khung, chu)


# ----------------------------------------------------------------------
# 6. Chương trình chính
# ----------------------------------------------------------------------
def mo_nguon(nguon):
    cap = cv2.VideoCapture(int(nguon) if nguon.isdigit() else nguon)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, os.path.isfile(nguon)


class LuongCamera:
    """Đọc camera trực tiếp ở luồng riêng và chỉ giữ KHUNG MỚI NHẤT, để khi YOLO xử
    chậm hơn camera thì hình không bị dồn lại (gây trễ)."""

    def __init__(self, cap):
        self.cap = cap
        self.khung = None
        self.moi = False
        self.dung = False
        self.khoa = threading.Lock()
        threading.Thread(target=self._doc, daemon=True).start()

    def _doc(self):
        while not self.dung:
            ok, k = self.cap.read()
            if ok:
                with self.khoa:
                    self.khung, self.moi = k, True
            else:
                time.sleep(0.05)

    def doc(self, cho_toi_da=0.2):
        han = time.monotonic() + cho_toi_da
        while time.monotonic() < han:
            with self.khoa:
                if self.moi:
                    self.moi = False
                    return True, self.khung
            time.sleep(0.003)
        return False, None

    def dong(self):
        self.dung = True


def main():
    global TOC_DO_GIONG
    ap = argparse.ArgumentParser(description="YOLO: hướng đồng hồ + khoảng cách + chỉ đường")
    ap.add_argument("--nguon", default="0")
    ap.add_argument("--ip", default=None,
                    help="địa chỉ IP điện thoại (app IP Webcam), ví dụ 192.168.1.10 hoặc 192.168.1.10:8080")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=NGUONG_TIN_CAY)
    ap.add_argument("--fov", type=float, default=None)
    ap.add_argument("--tieu-cu", type=float, default=None)
    ap.add_argument("--rong", type=int, default=960)
    ap.add_argument("--xoay", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--imgsz", type=int, default=640, help="nhỏ hơn (480, 320) = nhanh hơn")
    ap.add_argument("--luu", default=None)
    ap.add_argument("--toc-do", type=int, default=10,
                    help="tốc độ đọc (%%): âm = chậm hơn, 0 = bình thường, dương = nhanh hơn")
    ap.add_argument("--khong-noi", action="store_true")
    ap.add_argument("--thu-giong", action="store_true",
                    help="chỉ kiểm tra giọng nói (không mở camera) rồi thoát")
    ap.add_argument("--khong-hien", action="store_true")
    ap.add_argument("--khoang-cach-chuan", type=float, default=2.0)
    args = ap.parse_args()
    TOC_DO_GIONG = f"{args.toc_do:+d}%"
    if args.ip:
        ip = args.ip.replace("http://", "").strip("/ ")
        args.nguon = f"http://{ip if ':' in ip else ip + ':8080'}/video"
        print("Nguồn camera điện thoại:", args.nguon)

    if args.thu_giong:
        tao_thu_vien_giong(tat_ca_cum_tu())
        nd = NguoiDoc(bat=True)
        if nd.bat:
            nd.noi(["Hệ thống đã sẵn sàng", "|", "ghế", "hướng 12 giờ", "cách khoảng 2 mét",
                    "|", "Hãy đi sang trái", "hướng 11 giờ"])
            time.sleep(0.5)
            while nd.ban():
                time.sleep(0.1)
            time.sleep(0.5)
            print("Nếu bạn nghe được câu thử thì giọng nói hoạt động bình thường.")
        return

    from ultralytics import YOLO
    duong_dan_model = os.path.join(THU_MUC_GOC, args.model)
    print("Đang tải mô hình...")
    model = YOLO(duong_dan_model if os.path.exists(duong_dan_model) else args.model)

    if not args.khong_noi:
        tao_thu_vien_giong(tat_ca_cum_tu())

    cap, la_file = mo_nguon(args.nguon)
    if not cap.isOpened():
        print("Không mở được nguồn video:", args.nguon)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 30.0

    nguoi_doc = NguoiDoc(bat=not args.khong_noi)
    if nguoi_doc.bat:
        nguoi_doc.noi(["Hệ thống đã sẵn sàng"])     # nghe được câu này = giọng đang hoạt động
    luong = None if la_file else LuongCamera(cap)
    ph, ghi = None, None
    chi_so, loi_lien_tiep = 0, 0
    dong_bo = la_file and not args.luu          # bỏ bớt khung hình để theo kịp video
    bat_dau = time.monotonic()

    print("Bắt đầu. Nhấn q hoặc ESC để thoát.")
    while True:
        if dong_bo:
            bo = int((time.monotonic() - bat_dau) * fps) - chi_so
            for _ in range(min(max(bo, 0), 15)):
                if not cap.grab():
                    break
                chi_so += 1

        ok, khung = cap.read() if luong is None else luong.doc()
        if not ok:
            if la_file:
                break
            loi_lien_tiep += 1
            if loi_lien_tiep > 30:
                print("Mất kết nối với camera (kiểm tra Wi-Fi và app trên điện thoại).")
                break
            if not args.khong_hien:
                cv2.waitKey(1)
            continue
        loi_lien_tiep = 0

        if args.xoay:
            khung = cv2.rotate(khung, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}[args.xoay])
        c, r = khung.shape[:2]
        if r > args.rong:
            khung = cv2.resize(khung, (args.rong, int(c * args.rong / r)))
        cao, rong = khung.shape[:2]

        if ph is None:
            fov = args.fov or (FOV_NGANG_NGANG if rong >= cao else FOV_NGANG_DOC)
            ph = BoPhanTich(rong, cao, fov, args.tieu_cu)
            print(f"Khung {rong}x{cao}, FOV ngang {fov:.0f}°, tiêu cự ≈ {ph.f:.0f}px")
            if args.luu:
                os.makedirs(os.path.dirname(os.path.abspath(args.luu)), exist_ok=True)
                ghi = cv2.VideoWriter(args.luu, cv2.VideoWriter_fourcc(*"mp4v"), fps, (rong, cao))

        t = chi_so / fps if la_file else time.monotonic() - bat_dau

        kq = model.track(khung, persist=True, conf=args.conf, imgsz=args.imgsz,
                         verbose=False, tracker="bytetrack.yaml")[0]
        ds_tho = []
        if kq.boxes is not None:
            for b in kq.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                id_td = int(b.id[0]) if b.id is not None else None
                ds_tho.append((x1, y1, x2, y2, model.names[int(b.cls[0])],
                               float(b.conf[0]), id_td))

        ds = ph.cap_nhat(ds_tho, t)
        goi_y = ph.goi_y_huong_di(ds)
        tb = ph.tao_thong_bao(ds, goi_y, t, nguoi_doc.ban())
        if tb:
            cac_cum, khan = tb
            print(f"[{t:6.1f}s] {'!! ' if khan else ''}"
                  f"{', '.join(c for c in cac_cum if c != '|')}.")
            nguoi_doc.noi(cac_cum, khan_cap=khan)

        khung_ve = ve_ket_qua(khung, ds, goi_y, ph)
        if ghi is not None:
            ghi.write(khung_ve)
        chi_so += 1

        if not args.khong_hien:
            cv2.imshow("Nhan dien - huong dan duong di", khung_ve)
            cho = 1
            if la_file and not dong_bo:
                cho = max(1, int((chi_so / fps - (time.monotonic() - bat_dau)) * 1000))
            phim = cv2.waitKey(cho) & 0xFF
            if phim in (ord("q"), 27):
                break
            if phim == ord("m"):
                nguoi_doc.bat = not nguoi_doc.bat
                print("Giọng nói:", "BẬT" if nguoi_doc.bat else "TẮT")
            if phim == ord("c"):
                nguoi = [p for p in ds if p.ten_goc == "person"]
                if nguoi:
                    lon = max(nguoi, key=lambda p: p.y2 - p.y1)
                    ph.f = (lon.y2 - lon.y1) * args.khoang_cach_chuan / CHIEU_CAO_THAT["person"]
                    print(f"Đã hiệu chuẩn: tiêu cự = {ph.f:.0f}px "
                          f"(dùng --tieu-cu {ph.f:.0f} cho lần sau)")
                else:
                    print(f"Hiệu chuẩn: cần thấy 1 người đứng cách camera {args.khoang_cach_chuan} m.")

    if luong is not None:
        luong.dong()
        time.sleep(0.1)
    cap.release()
    if ghi is not None:
        ghi.release()
        print("Đã lưu video:", args.luu)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()