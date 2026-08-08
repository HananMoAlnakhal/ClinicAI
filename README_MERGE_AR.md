# ملفات دمج ClinicAI V4

هذه الملفات مبنية على `ClinicAI_fixed_2026-07-14.zip` كأساس، ثم دُمجت فيها الإصلاحات التالية دون حذف إصلاحات 14 يوليو:

- الحفاظ على تعديل الساعة قبل التأكيد.
- الحفاظ على معالجة التخصص غير المتوفر وعرض طبيب عام بموافقة صريحة.
- الحفاظ على إزالة أزرار التأكيد بعد انتهاء الطلب.
- الحفاظ على عدم تكرار سؤال الاسم.
- الحفاظ على حل تعارض الـSlot في اللحظة الأخيرة.
- عدم تعديل ملفات `data/levantine/symptoms.json` و`time_phrases.json` الموجودة أصلًا.

الإضافات:

- `find_slot_offer()` ورسالة واضحة عند الانتقال من اليوم المطلوب إلى موعد بديل.
- البحث داخل Slots الطبيب/العيادة المختارة فقط.
- منع التحويل الصامت إلى طب عام.
- حجز ذري للـSlot بواسطة تحديث شرطي `available -> booked`.
- حفظ نص شكوى المريض الأصلي وعدم السماح للـLLM باستبداله.
- فهم النفي في علامات الخطورة مثل `بدون نزيف`.
- حماية Gemini بمهلة زمنية و`try/except` كامل.
- fallback اختياري إلى OpenAI API عند ضبط `OPENAI_API_KEY`.
- fallback محلي حتمي إلى الـFSM والقواعد عند فشل المزودين، بحيث لا يتوقف البوت.
- Telegram error handler عام.

## التطبيق

استبدلوا الملفات داخل المشروع مع الحفاظ على نفس المسارات، ثم نفذوا:

```powershell
pip install -r requirements.txt
python -m py_compile database\crud.py fsm\patient_fsm.py scheduler\priority.py nlp\gemini_client.py main.py config.py
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
python main.py
```

## إعداد OpenAI الاختياري

داخل `.env` يمكن إضافة:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.6
LLM_TIMEOUT_SECONDS=20
```

عدم ضبط المفتاح لا يمنع تشغيل المشروع؛ عند فشل Gemini أو عدم توفر أي API يكمل النظام بالقواعد المحلية والـFSM.
