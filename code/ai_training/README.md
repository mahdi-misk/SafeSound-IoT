# 🎙️ نظام مراقبة المحركات الذكي باستخدام الذكاء الاصطناعي والصوت
## Smart Motor Monitoring System Using Sound and AI

هذا المستند يلخص الخطوات والملفات التي تم إعدادها لتشغيل وتدريب نموذج الذكاء الاصطناعي وتجهيز نظام المراقبة الذكي.

---

## 🛠️ ملخص ما تم عمله (What We Did)

### 1️⃣ موازنة وتجهيز البيانات (`prepare_dataset.py`)
- تم إنشاء كود بايثون لمعالجة البيانات الأولية للأجهزة الأربعة (`id_fan_00`, `id_pump_00`, `id_slider_00`, `id_valve_00`).
- يقوم الكود بموازنة فئتي الأصوات (`normal` و `abnormal`) لتصل إلى **500 عينة لكل فئة** لكل جهاز:
  - **Under-sampling (تقليل العينات):** اختيار عينات عشوائية إذا كان العدد أكبر من 500.
  - **Over-sampling (زيادة العينات):** تكرار العينات عشوائياً إذا كان العدد أقل من 500.
- في النهاية، يقوم بضغط المجلد الناتج باسم **`balanced_dataset.zip`** لتسهيل رفعه إلى السحابة.

### 2️⃣ دليل التدريب على Google Colab (`Google_Colab_Guide.pdf` / `.py`)
- تم كتابة كود بايثون **`generate_pdf_guide.py`** لتوليد دليل تدريب منسق بصيغة **PDF** باللغة العربية مع دعم الخطوط وعرض الأكواد بشكل جمالي.
- يحتوي الدليل على تعليمات مفصلة لخطوات:
  1. رفع البيانات إلى Google Drive.
  2. إعداد الـ GPU (T4) في Google Colab.
  3. تشغيل خلايا الكود الستة لبناء نموذج شبكة عصبية التفافية (CNN) وتدريبها وحفظ الموديل بصيغة `machine_sounds_model.h5`.

### 3️⃣ تحديث خادم الاستقبال المباشر (`server.py`)
- تم دمج وتحديث خادم الويب المبني على **FastAPI** و **WebSockets**.
- نظرًا لأن بيئة العمل المحلية تستخدم **Python 3.14** (الذي لا يدعم TensorFlow حالياً)، قمنا بإضافة دعم كامل لتشغيل الموديل بصيغة **ONNX** الأكثر سرعة وخفة ومتوافقة بالكامل مع Python 3.14 عبر مكتبة `onnxruntime`.
- يقوم الخادم الآن بـ:
  - استقبال البث الصوتي المباشر من جهاز الـ **ESP32** عبر الـ WebSocket.
  - التحقق من وجود نموذج الـ CNN بصيغة ONNX (`machine_sounds_model.onnx`) وتفضيله في العمل.
  - في حال عدم توفره، يتراجع السيرفر تلقائياً لتشغيل النموذج القديم (`q_table_rf_model.pkl`) لتفادي توقف الخدمة.

---

## 📂 هيكلية المجلد والملفات الرئيسية

- 📁 **`ai_training/`** : يحتوي على ملفات التدريب.
  - 📄 `prepare_dataset.py` : موازنة البيانات وضغطها.
  - 📄 `Google_Colab_Guide.pdf` : دليل التدريب خطوة بخطوة.
  - 📄 `machine_sounds_model.onnx` : الموديل الجديد بعد التحويل (يتم وضعه هنا).
- 📄 **`server.py`** : الخادم الرئيسي للاستقبال المباشر والتحليل اللحظي للمحركات.
- 📄 **`generate_pdf_guide.py`** : كود توليد دليل الـ PDF.
- 📄 **`poster.html`** : واجهة بوستر المشروع الفنية.

---

## 🚀 كيفية التشغيل والاستخدام الآن

### الخطوة 1: تحويل الموديل إلى ONNX وتنزيله
قم بتشغيل الكود التالي في Google Colab بعد انتهاء التدريب لتحويل الموديل وتنزيله:
```python
!pip install tf2onnx onnx
import tensorflow as tf
import tf2onnx

# تحميل الموديل
model = tf.keras.models.load_model('/content/machine_sounds_model.h5')

# التحويل وتصدير الملف
spec = (tf.TensorSpec((None, 128, 128, 1), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open("/content/machine_sounds_model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

# تنزيل الملف للجهاز
from google.colab import files
files.download("/content/machine_sounds_model.onnx")
```

### الخطوة 2: وضع الملف في المسار المخصص
انقل ملف `machine_sounds_model.onnx` الذي تم تنزيله إلى المجلد التالي:
`c:\Users\mahdi\OneDrive\Desktop\0096\ai_training\`

### الخطوة 3: تشغيل خادم FastAPI
افتح التيرمنال في مجلد المشروع الرئيسي وشغّل الأمر التالي:
```bash
python server.py
```
سيبدأ الخادم بالعمل وسيعرض لك عنوان الاستقبال اللحظي للبيانات من الـ ESP32.
