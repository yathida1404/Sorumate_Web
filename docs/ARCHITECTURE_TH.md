# SoruMate — โครงสร้างระบบ (หน้าบ้าน / หลังบ้าน)

## ภาพรวม

```
เบราว์เซอร์ (static/index.html — หน้าตาและ JS ทั้งหมดอยู่ในไฟล์นี้)
   │  GET /                    → หน้าเว็บ
   │  GET /api/catalog         → รายการ 25 ชิ้น + กฎ
   │  POST /api/ask            → ถามวิธีทิ้ง (ค้นตรงตัวจาก JSON เท่านั้น)
   │  POST /predict            → ส่งภาพ ได้ผลตรวจจับกลับมา
   ▼
FastAPI (main.py → แพ็กเกจ sorumate/)
   ├─ routes/        รับ-ส่ง HTTP เท่านั้น ไม่มีตรรกะ
   ├─ catalog.py     ค้นกฎจาก data/*.json
   ├─ inference/     ตัวตรวจจับภาพ (เลือก ONNX หรือ Roboflow)
   ├─ imaging.py     ตรวจไฟล์ภาพที่อัปโหลด
   ├─ security.py    จำกัดจำนวนคำขอ / รหัสทดสอบ
   └─ config.py      ค่าตั้งทั้งหมดจาก Environment Variables
```

## ข้อมูลเก็บที่ไหน (ไม่มีฐานข้อมูล)

| ข้อมูล | ที่เก็บ | แก้ยังไง |
|---|---|---|
| กฎของ 25 ชิ้น | `data/waste_catalog.json` | แก้ไฟล์ JSON แล้ว commit |
| ของนอก 25 ชิ้นที่ค้นหาได้ | `data/other_waste_items.json` | เพิ่มรายการใน `items` (ต้องตรวจจากคู่มือเมือง) |
| โมเดลตรวจจับ | `models/best.onnx` | อัปโหลดไฟล์ที่ export จาก Colab |
| ภาพที่ผู้ใช้อัปโหลด | ไม่เก็บ (ประมวลผลในหน่วยความจำแล้วทิ้ง) | — |
| สถิติ/คำตอบทดสอบ | ไม่เก็บบนเซิร์ฟเวอร์ (`/event` รับแล้วทิ้ง) | ใช้ Google Form |

ของที่ไม่อยู่ใน JSON ทั้งสองไฟล์ ระบบจะตอบว่า "ยังไม่มีข้อมูลที่ยืนยัน" พร้อมเบอร์เมือง ไม่ให้ AI เดากฎเอง

## เลือกตัวตรวจจับ

ตั้ง `SORUMATE_ENGINE` บน Render:

- `auto` (ค่าเริ่มต้น): ถ้ามี `models/best.onnx` ใช้ ONNX, ไม่งั้นถ้าตั้ง Roboflow ไว้ใช้ Roboflow
- `onnx`: ใช้ไฟล์ ONNX เท่านั้น
- `roboflow`: ใช้ Roboflow เท่านั้น (ต้องเพิ่ม `inference-sdk` ใน requirements.txt และตั้ง `ROBOFLOW_API_KEY`, `ROBOFLOW_MODEL_ID`)

ดูสถานะจริงได้ที่ `/health` (`engine`, `ai_ready`, `engine_error`, `unmapped_classes`)

## Environment Variables

| ชื่อ | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `SORUMATE_ENGINE` | `auto` | ตัวตรวจจับ |
| `ONNX_MODEL_PATH` | `models/best.onnx` | พาธไฟล์โมเดล |
| `CONFIDENCE_THRESHOLD` | `0.30` | ต่ำกว่านี้ไม่แสดง |
| `RATE_LIMIT_PER_10MIN` | `60` | จำนวนคำขอ/IP/10 นาที (เดิมคือ 20) |
| `DEMO_ACCESS_CODE` | ว่าง | ถ้าตั้ง ต้องส่ง header `X-Demo-Code` |
| `ORT_THREADS` | `1` | จำนวน thread ของ ONNX Runtime |
| `AGNOSTIC_NMS_IOU` | `0.6` | กรอบต่าง class ที่ซ้อนกันเกินค่านี้ เก็บแค่อันที่มั่นใจสุด (ตั้ง 1 = ปิด) |

## ตรวจชื่อ class หลังเทรน

ระบบจับคู่ด้วย **ชื่อ class** (ไม่สนตัวพิมพ์/ขีดล่าง) ไม่ใช่เลข ID:

```bash
python scripts/check_classes.py data.yaml        # หรือ models/best.onnx
python scripts/try_onnx.py ภาพตัวอย่าง.jpg       # ลองรันโมเดลกับภาพจริง 1 ภาพ
python tests/test_core.py                         # ทดสอบตรรกะหลัก
```

## ส่วนที่ไม่ได้แตะ

โฟลเดอร์ `static/` (ดีไซน์หน้าบ้าน) ไม่ถูกแก้แม้แต่ไบต์เดียว
