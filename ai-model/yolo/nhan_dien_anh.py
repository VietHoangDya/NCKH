
from ultralytics import YOLO

# 1. Tai mo hinh YOLO da duoc huan luyen
model = YOLO("yolo11n.pt")

# 2. Duong dan anh can nhan dien
duong_dan_anh = "images/anh_thu2.jpg"

# 3. Nhan dien vat the
ket_qua = model(duong_dan_anh, conf=0.4)

# 4. In thong tin vat the
for ket_qua_anh in ket_qua:
    for vat_the in ket_qua_anh.boxes:
        ma_lop = int(vat_the.cls[0])
        ten_vat_the = model.names[ma_lop]
        do_tin_cay = float(vat_the.conf[0])

        print("Vat the:", ten_vat_the)
        print("Do tin cay:", round(do_tin_cay * 100, 2), "%")
        print("-------------------")

# 5. Hien thi anh da ve khung nhan dien
ket_qua[0].show()

# 6. Luu anh ket qua
ket_qua[0].save(filename="ket_qua.jpg")
