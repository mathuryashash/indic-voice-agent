"""Bigger eval set (v2, 23 Nov 2026). Not part of the original 42-case eval.

IN-SCOPE: the same 42 scripted texts from eval/cases.py (so the answer checks still apply), rendered with
NEW speakers that were never used before: other edge-tts voices (other accents / multilingual voices),
a different TTS engine (Google gTTS), and speed / pitch shifted versions of the original voices.
Each clip also gets one acoustic condition, cycled: clean, white noise 15 dB, white noise 5 dB, phone band (300-3400 Hz, 8 kHz).

OUT-OF-SCOPE: 7 Indian languages the agent does NOT support (mr, bn, gu, ta, te, ml, ur), 4 questions each,
2 voices each. Translations written by the assistant (not checked by a native speaker).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.cases import CASES  # noqa: E402

CONDS = ["clean", "white15", "white5", "phone"]

# variant: (key, engine, voice, rate, pitch)
VARIANTS = {
    "en": [
        ("en_neerjaX", "edge", "en-IN-NeerjaExpressiveNeural", "+0%", "+0Hz"),
        ("en_usGuy", "edge", "en-US-GuyNeural", "+0%", "+0Hz"),
        ("en_gbSonia", "edge", "en-GB-SoniaNeural", "+0%", "+0Hz"),
        ("en_ngAbeo", "edge", "en-NG-AbeoNeural", "+0%", "+0Hz"),
        ("en_auNatasha", "edge", "en-AU-NatashaNeural", "+0%", "+0Hz"),
        ("en_gtts_in", "gtts", "co.in", "", ""),
        ("en_prabhat_fast", "edge", "en-IN-PrabhatNeural", "+35%", "+0Hz"),
        ("en_neerja_low", "edge", "en-IN-NeerjaNeural", "-15%", "-40Hz"),
    ],
    "hi": [
        ("hi_avaML", "edge", "en-US-AvaMultilingualNeural", "+0%", "+0Hz"),
        ("hi_andrewML", "edge", "en-US-AndrewMultilingualNeural", "+0%", "+0Hz"),
        ("hi_remyML", "edge", "fr-FR-RemyMultilingualNeural", "+0%", "+0Hz"),
        ("hi_gtts", "gtts", "co.in", "", ""),
        ("hi_madhur_fast", "edge", "hi-IN-MadhurNeural", "+35%", "+0Hz"),
        ("hi_swara_slow", "edge", "hi-IN-SwaraNeural", "-20%", "+0Hz"),
        ("hi_swara_high", "edge", "hi-IN-SwaraNeural", "+0%", "+50Hz"),
        ("hi_madhur_low", "edge", "hi-IN-MadhurNeural", "+0%", "-50Hz"),
    ],
    "kn": [
        ("kn_gtts", "gtts", "co.in", "", ""),
        ("kn_gtts_slow", "gtts_slow", "co.in", "", ""),
        ("kn_gagan_fast", "edge", "kn-IN-GaganNeural", "+35%", "+0Hz"),
        ("kn_sapna_fast", "edge", "kn-IN-SapnaNeural", "+35%", "+0Hz"),
        ("kn_sapna_slow", "edge", "kn-IN-SapnaNeural", "-20%", "+0Hz"),
        ("kn_sapna_high", "edge", "kn-IN-SapnaNeural", "+0%", "+50Hz"),
        ("kn_gagan_low", "edge", "kn-IN-GaganNeural", "+0%", "-50Hz"),
        ("kn_gagan_slow", "edge", "kn-IN-GaganNeural", "-20%", "+0Hz"),
    ],
}

IN_SCOPE = []
for ci, c in enumerate(CASES):
    for vi, (key, engine, voice, rate, pitch) in enumerate(VARIANTS[c["lang"]]):
        IN_SCOPE.append(dict(c, clip=f"{c['id']}__{key}", base_id=c["id"], variant=key, engine=engine,
                             voice=voice, rate=rate, pitch=pitch, cond=CONDS[(ci + vi) % 4], scope="in"))

DEC = ["December", "Dec", "12-", "/12"]
OOS_Q = {  # question key -> (expected tool, check)
    "exam": ("get_exam_schedule", {"exam_code": "BCS502", "facts": [["14"]]}),
    "sgpa": ("calculate_sgpa", {"facts": [["9.09"]]}),
    "dns": ("search_syllabus", {"syllabus_code": "BCS502", "facts": []}),
    "hello": (None, {"facts": []}),
}
OOS_TEXT = {
    "mr": {"exam": "माझी Computer Networks ची परीक्षा कधी आहे?",
           "sgpa": "माझा SGPA काढ: 4 क्रेडिटच्या विषयात 92 गुण, दुसऱ्या 4 क्रेडिटच्या विषयात 85 गुण आणि 3 क्रेडिटच्या विषयात 72 गुण.",
           "dns": "कॉम्प्युटर नेटवर्क्सच्या अभ्यासक्रमात DNS कुठे येतो?",
           "hello": "नमस्कार, तू कोण आहेस आणि काय करू शकतोस?"},
    "bn": {"exam": "আমার Computer Networks পরীক্ষা কবে?",
           "sgpa": "আমার SGPA হিসাব করো: 4 ক্রেডিটের বিষয়ে 92 নম্বর, আরেকটি 4 ক্রেডিটের বিষয়ে 85 আর 3 ক্রেডিটের বিষয়ে 72।",
           "dns": "কম্পিউটার নেটওয়ার্কস সিলেবাসে DNS কোথায় আছে?",
           "hello": "নমস্কার, তুমি কে আর কী করতে পারো?"},
    "gu": {"exam": "મારી Computer Networks ની પરીક્ષા ક્યારે છે?",
           "sgpa": "મારો SGPA ગણો: 4 ક્રેડિટના વિષયમાં 92 માર્ક્સ, બીજા 4 ક્રેડિટના વિષયમાં 85 અને 3 ક્રેડિટના વિષયમાં 72.",
           "dns": "કમ્પ્યુટર નેટવર્ક્સના સિલેબસમાં DNS ક્યાં આવે છે?",
           "hello": "નમસ્તે, તમે કોણ છો અને શું કરી શકો છો?"},
    "ta": {"exam": "என்னுடைய Computer Networks தேர்வு எப்போது?",
           "sgpa": "என் SGPA கணக்கிடு: 4 கிரெடிட் பாடத்தில் 92 மதிப்பெண், இன்னொரு 4 கிரெடிட் பாடத்தில் 85, 3 கிரெடிட் பாடத்தில் 72.",
           "dns": "கம்ப்யூட்டர் நெட்வொர்க்ஸ் பாடத்திட்டத்தில் DNS எங்கே வருகிறது?",
           "hello": "வணக்கம், நீங்கள் யார், என்ன செய்ய முடியும்?"},
    "te": {"exam": "నా Computer Networks పరీక్ష ఎప్పుడు?",
           "sgpa": "నా SGPA లెక్కించు: 4 క్రెడిట్ సబ్జెక్ట్‌లో 92 మార్కులు, ఇంకో 4 క్రెడిట్ సబ్జెక్ట్‌లో 85, 3 క్రెడిట్ సబ్జెక్ట్‌లో 72.",
           "dns": "కంప్యూటర్ నెట్‌వర్క్స్ సిలబస్‌లో DNS ఎక్కడ వస్తుంది?",
           "hello": "నమస్కారం, మీరు ఎవరు, ఏమి చేయగలరు?"},
    "ml": {"exam": "എന്റെ Computer Networks പരീക്ഷ എപ്പോഴാണ്?",
           "sgpa": "എന്റെ SGPA കണക്കാക്കൂ: 4 ക്രെഡിറ്റ് വിഷയത്തിൽ 92 മാർക്ക്, മറ്റൊരു 4 ക്രെഡിറ്റ് വിഷയത്തിൽ 85, 3 ക്രെഡിറ്റ് വിഷയത്തിൽ 72.",
           "dns": "കമ്പ്യൂട്ടർ നെറ്റ്‌വർക്ക്സ് സിലബസിൽ DNS എവിടെയാണ് വരുന്നത്?",
           "hello": "നമസ്കാരം, നിങ്ങൾ ആരാണ്, എന്തൊക്കെ ചെയ്യാൻ കഴിയും?"},
    "ur": {"exam": "میرا Computer Networks کا امتحان کب ہے؟",
           "sgpa": "میرا SGPA نکالو: 4 کریڈٹ والے مضمون میں 92 نمبر، دوسرے 4 کریڈٹ والے مضمون میں 85 اور 3 کریڈٹ والے مضمون میں 72۔",
           "dns": "کمپیوٹر نیٹ ورکس کے سلیبس میں DNS کہاں آتا ہے؟",
           "hello": "السلام علیکم، آپ کون ہیں اور کیا کر سکتے ہیں؟"},
}
OOS_VOICES = {
    "mr": ["mr-IN-AarohiNeural", "mr-IN-ManoharNeural"], "bn": ["bn-IN-TanishaaNeural", "bn-IN-BashkarNeural"],
    "gu": ["gu-IN-DhwaniNeural", "gu-IN-NiranjanNeural"], "ta": ["ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"],
    "te": ["te-IN-ShrutiNeural", "te-IN-MohanNeural"], "ml": ["ml-IN-SobhanaNeural", "ml-IN-MidhunNeural"],
    "ur": ["ur-IN-GulNeural", "ur-IN-SalmanNeural"],
}
OUT_OF_SCOPE = []
for lang, qs in OOS_TEXT.items():
    for qk, text in qs.items():
        for vi, voice in enumerate(OOS_VOICES[lang]):
            tool, chk = OOS_Q[qk]
            OUT_OF_SCOPE.append(dict(clip=f"oos_{lang}_{qk}_{vi}", id=f"oos_{lang}_{qk}_{vi}", base_id=qk, lang=lang,
                                     text=text, expected_tool=tool, check=chk, variant=voice, engine="edge",
                                     voice=voice, rate="+0%", pitch="+0Hz", cond="clean", scope="oos"))

ALL = IN_SCOPE + OUT_OF_SCOPE
assert len(IN_SCOPE) == 42 * 8 and len(OUT_OF_SCOPE) == 7 * 4 * 2
