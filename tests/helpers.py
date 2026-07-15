def infer_urgency_score(text: str) -> float:
    t = text
    high = ["الآن", "فجأة", "حاد", "شديد", "صعوبة نطق", "ضعف مفاجئ", "وقع", "أزمة", "هبوط سكر"]
    mid = ["دوخة", "تعب", "تنميل", "صفير", "تورم", "رجفة"]
    low = ["متابعة", "دوري", "روتيني", "مراجعة", "بدون أعراض جديدة"]

    if any(k in t for k in high):
        return 0.9
    if any(k in t for k in mid):
        return 0.55
    if any(k in t for k in low):
        return 0.25
    return 0.4
