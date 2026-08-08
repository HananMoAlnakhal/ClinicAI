# ClinicAI 🏥

مساعد ذكي عربي لحجز مواعيد العيادات عبر تيليجرام: محادثة نصية وصوتية → تصنيف التخصص → حساب الأولوية (P1/P2/P3) → حجز الموعد → لوحة تحكم حية للعيادة.

Arabic AI clinic assistant on Telegram: text/voice conversation → specialty classification → priority scoring (P1/P2/P3) → appointment booking → live clinic dashboard.

> 📋 آخر تعديلات وإصلاحات النسخة الحالية موثقة بالتفصيل في **CHANGES_AR.md**

---

## المتطلبات | Requirements

- Python 3.11+
- توكن بوت تيليجرام (مجاني من [@BotFather](https://t.me/BotFather)) | Telegram bot token
- اختياري: مفتاح Gemini API (للردود الذكية عند الغموض) | Optional: Gemini API key
- اختياري: ffmpeg (فقط إذا أردتم الميزات الصوتية) | Optional: ffmpeg (voice features only)

## التثبيت | Install

```bash
# 1) الحزم الأساسية (الحجز + الأولوية + لوحة التحكم — بدون صوت)
#    Core packages (booking + priority + dashboard — no voice)
pip install python-telegram-bot sqlalchemy fastapi uvicorn jinja2 python-dotenv google-genai

# 2) حزم الصوت — اختيارية وثقيلة (عدة جيجابايت)! تخطوها إذا الإنترنت محدود
#    Voice packages — OPTIONAL and heavy (several GB). Skip on limited internet.
pip install openai-whisper edge-tts

# أو كل شيء دفعة واحدة | or everything at once:
pip install -r requirements.txt
```

## الإعداد | Setup

```bash
# انسخوا القالب وضعوا التوكنات الحقيقية | copy the template, add real tokens
cp .env.example .env
```

عدّلوا `.env` | Edit `.env`:

| المتغير | القيمة |
|---|---|
| `TELEGRAM_BOT_TOKEN` | التوكن من BotFather — **مطلوب** |
| `GEMINI_API_KEY` | اختياري — بدونه يعمل كل شيء بالقواعد فقط |
| `WHISPER_MODEL` | `small` جيد للبداية (اختياري، للصوت فقط) |
| `DASHBOARD_PORT` | مثلاً `8000` |

⚠️ لا ترفعوا ملف `.env` إلى GitHub أبداً (موجود في `.gitignore`).

## التشغيل | Run

```bash
python main.py
```

عند أول تشغيل تُبنى قاعدة البيانات تلقائياً وتُعبّأ بـ 8 أطباء و~960 موعداً لأسبوعين.
First run auto-creates the database and seeds 8 doctors + ~960 slots.

- **البوت**: راسلوه على تيليجرام (الرابط من BotFather)
- **لوحة التحكم**: http://localhost:8000

## ملاحظات تشغيل مهمة | Operational notes

1. **لا حاجة لأي استضافة**: البوت يستخدم long-polling — أي لابتوب متصل بالإنترنت يكفي، بدون IP عام أو خادم.
2. **توكن واحد = جهاز واحد**: لا تشغّلوا `main.py` بنفس التوكن على جهازين معاً (تيليجرام يرمي أخطاء تعارض). أنشئوا توكناً ثانياً للتطوير إذا لزم.
3. **البوت يعيش ما دام الجهاز شغالاً**: إذا أُغلق اللابتوب يتوقف الرابط. حدّدوا «جهاز العرض» وجهّزوا جهازاً بديلاً بنفس الخطوات أعلاه.
4. **بدون حزم الصوت**: كل شيء يعمل (حجز، أولوية، لوحة تحكم) — الرسائل الصوتية فقط تُرد برسالة اعتذار.

## بنية المشروع | Structure

```
bot/        معالجات تيليجرام (نص، صوت، أزرار)
fsm/        آلة الحالات: جمع البيانات → تحقق → تصنيف → حجز
scheduler/  مصنّف التخصصات + محرك الأولوية (5 عوامل موزونة)
nlp/        التطبيع العربي + الاستخراج + عميل Gemini
voice/      Whisper (تفريغ) + edge-tts (رد صوتي)
database/   SQLAlchemy: نماذج، عمليات، ترحيلات، تعبئة
dashboard/  FastAPI + Jinja2: مواعيد، أطباء، جلسات
data/       مفردات اللهجة الشامية (أعراض + عبارات وقت) — وسّعوها!
```
