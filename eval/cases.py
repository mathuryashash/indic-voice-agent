"""Scripted eval utterances: 14 per language (hi / kn / en) = 42.

Each case:
  text           reference transcript (also what edge-tts speaks). Hindi/Kannada cases are
                 code-mixed the way VTU students actually talk (subject names often in English).
  expected_tool  tool that must be called (None = answer without tools)
  check          dict of machine checks:
                   exam_code      -> get_exam_schedule result must resolve to this course code
                   syllabus_code  -> top search_syllabus hit must be from this course
                   reminder_time  -> set_reminder time argument must equal this
                   facts          -> list of fact groups; each group is a list of acceptable
                                     strings, at least one of which must appear in the reply
Eval "today" is fixed to Sunday 2026-10-04 10:00 so relative reminder times are deterministic.
"""

DEC = ["December", "दिसंबर", "दिसम्बर", "ಡಿಸೆಂಬರ್", "Dec", "12-", "/12"]

CASES = [
    # ------------------------------------------------------------------ English
    dict(id="en01", lang="en", text="When is my Computer Networks exam?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS502", "facts": [["14"], DEC]}),
    dict(id="en02", lang="en", text="What time does the Theory of Computation exam start?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS503", "facts": [["2:00", "14:00", "2 PM", "2 p.m.", "2pm", "two"]]}),
    dict(id="en03", lang="en", text="Where is the software engineering exam hall?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS501", "facts": [["204"]]}),
    dict(id="en04", lang="en", text="Which day is the web technology lab exam?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCSL504", "facts": [["Thursday", "3 December", "December 3", "3rd December", "December 3rd"]]}),
    dict(id="en05", lang="en", text="Tell me the date of the TOC exam.", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS503", "facts": [["17"], DEC]}),
    dict(id="en06", lang="en", text="Which module of computer networks covers TCP congestion control?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS502", "facts": [["4", "four", "Transport"]]}),
    dict(id="en07", lang="en", text="In which module is the pumping lemma taught in Theory of Computation?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS503", "facts": [["2", "two", "Regular", "4", "four"]]}),
    dict(id="en08", lang="en", text="What is the textbook for Theory of Computation?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS503", "facts": [["Hopcroft"]]}),
    dict(id="en09", lang="en", text="Does the software engineering syllabus include Scrum?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS501", "facts": [["3", "three", "Agile"]]}),
    dict(id="en10", lang="en", text="Remind me to revise computer networks tomorrow at 6 pm.", expected_tool="set_reminder",
         check={"reminder_time": "2026-10-05 18:00", "facts": [["6", "18:00", "six"]]}),
    dict(id="en11", lang="en", text="Set a reminder to submit the web lab record on October 10 at 9 am.", expected_tool="set_reminder",
         check={"reminder_time": "2026-10-10 09:00", "facts": [["10"]]}),
    dict(id="en12", lang="en", text="Calculate my SGPA. I got 92 marks in a 4 credit subject, 85 in another 4 credit subject and 72 in a 3 credit subject.",
         expected_tool="calculate_sgpa", check={"facts": [["9.09"]]}),
    dict(id="en13", lang="en", text="My SGPA was 8 in a 20 credit semester and 9 in another 20 credit semester. What is my CGPA?",
         expected_tool="calculate_cgpa", check={"facts": [["8.5"]]}),
    dict(id="en14", lang="en", text="Hi, who are you and what can you do?", expected_tool=None,
         check={"facts": [["exam", "syllabus", "reminder", "SGPA", "CGPA"]]}),

    # ------------------------------------------------------------------ Hindi
    dict(id="hi01", lang="hi", text="मेरा Computer Networks का एग्जाम कब है?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS502", "facts": [["14", "चौदह"], DEC]}),
    dict(id="hi02", lang="hi", text="थ्योरी ऑफ कंप्यूटेशन की परीक्षा किस समय शुरू होगी?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS503", "facts": [["2", "14:00", "दो"]]}),
    dict(id="hi03", lang="hi", text="सॉफ्टवेयर इंजीनियरिंग की परीक्षा किस कमरे में है?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS501", "facts": [["204"]]}),
    dict(id="hi04", lang="hi", text="वेब टेक्नोलॉजी लैब की परीक्षा किस दिन है?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCSL504", "facts": [["गुरुवार", "Thursday", "3"]]}),
    dict(id="hi05", lang="hi", text="TOC की परीक्षा की तारीख बताओ।", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS503", "facts": [["17", "सत्रह"], DEC]}),
    dict(id="hi06", lang="hi", text="कंप्यूटर नेटवर्क्स में TCP congestion control कौन से मॉड्यूल में है?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS502", "facts": [["4", "चार", "Transport", "ट्रांसपोर्ट"]]}),
    dict(id="hi07", lang="hi", text="थ्योरी ऑफ कंप्यूटेशन में ट्यूरिंग मशीन कौन से मॉड्यूल में पढ़ाई जाती है?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS503", "facts": [["5", "पांच", "पाँच"]]}),
    dict(id="hi08", lang="hi", text="सॉफ्टवेयर इंजीनियरिंग में COCOMO मॉडल किस मॉड्यूल में है?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS501", "facts": [["4", "चार"]]}),
    dict(id="hi09", lang="hi", text="कंप्यूटर नेटवर्क्स के सिलेबस में DNS कहाँ आता है?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS502", "facts": [["5", "पांच", "पाँच", "Application", "एप्लिकेशन", "एप्लीकेशन"]]}),
    dict(id="hi10", lang="hi", text="मुझे कल शाम 7 बजे TOC रिवाइज करने की याद दिलाना।", expected_tool="set_reminder",
         check={"reminder_time": "2026-10-05 19:00", "facts": [["7", "सात", "19:00"]]}),
    dict(id="hi11", lang="hi", text="8 अक्टूबर सुबह 10 बजे assignment जमा करने का reminder लगा दो।", expected_tool="set_reminder",
         check={"reminder_time": "2026-10-08 10:00", "facts": [["8", "आठ"]]}),
    dict(id="hi12", lang="hi", text="मेरा SGPA निकालो: 4 क्रेडिट के विषय में 92 नंबर, दूसरे 4 क्रेडिट विषय में 85 और 3 क्रेडिट विषय में 72 नंबर।",
         expected_tool="calculate_sgpa", check={"facts": [["9.09"]]}),
    dict(id="hi13", lang="hi", text="पहले सेमेस्टर में SGPA 8 था और दूसरे में 9, दोनों 20 क्रेडिट के थे। मेरा CGPA क्या है?",
         expected_tool="calculate_cgpa", check={"facts": [["8.5"]]}),
    dict(id="hi14", lang="hi", text="नमस्ते, तुम कौन हो और क्या कर सकते हो?", expected_tool=None,
         check={"facts": [["परीक्षा", "सिलेबस", "एग्जाम", "रिमाइंडर", "SGPA", "CGPA", "exam", "syllabus"]]}),

    # ------------------------------------------------------------------ Kannada
    dict(id="kn01", lang="kn", text="ನನ್ನ ಕಂಪ್ಯೂಟರ್ ನೆಟ್‌ವರ್ಕ್ಸ್ ಪರೀಕ್ಷೆ ಯಾವಾಗ?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS502", "facts": [["14", "ಹದಿನಾಲ್ಕು"], DEC]}),
    dict(id="kn02", lang="kn", text="ಥಿಯರಿ ಆಫ್ ಕಂಪ್ಯೂಟೇಶನ್ ಪರೀಕ್ಷೆ ಎಷ್ಟು ಗಂಟೆಗೆ ಶುರು?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS503", "facts": [["2", "14:00", "ಎರಡು"]]}),
    dict(id="kn03", lang="kn", text="ಸಾಫ್ಟ್‌ವೇರ್ ಎಂಜಿನಿಯರಿಂಗ್ ಪರೀಕ್ಷೆ ಯಾವ ಕೊಠಡಿಯಲ್ಲಿ ಇದೆ?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS501", "facts": [["204"]]}),
    dict(id="kn04", lang="kn", text="ವೆಬ್ ಟೆಕ್ನಾಲಜಿ ಲ್ಯಾಬ್ ಪರೀಕ್ಷೆ ಯಾವ ದಿನ?", expected_tool="get_exam_schedule",
         check={"exam_code": "BCSL504", "facts": [["ಗುರುವಾರ", "Thursday", "3"]]}),
    dict(id="kn05", lang="kn", text="TOC ಪರೀಕ್ಷೆಯ ದಿನಾಂಕ ಹೇಳಿ.", expected_tool="get_exam_schedule",
         check={"exam_code": "BCS503", "facts": [["17", "ಹದಿನೇಳು"], DEC]}),
    dict(id="kn06", lang="kn", text="ಕಂಪ್ಯೂಟರ್ ನೆಟ್‌ವರ್ಕ್ಸ್‌ನಲ್ಲಿ TCP congestion control ಯಾವ ಮಾಡ್ಯೂಲ್‌ನಲ್ಲಿ ಇದೆ?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS502", "facts": [["4", "ನಾಲ್ಕು", "Transport", "ಟ್ರಾನ್ಸ್‌ಪೋರ್ಟ್"]]}),
    dict(id="kn07", lang="kn", text="ಥಿಯರಿ ಆಫ್ ಕಂಪ್ಯೂಟೇಶನ್‌ನಲ್ಲಿ ಟ್ಯೂರಿಂಗ್ ಮಷೀನ್ ಯಾವ ಮಾಡ್ಯೂಲ್?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS503", "facts": [["5", "ಐದು"]]}),
    dict(id="kn08", lang="kn", text="ಸಾಫ್ಟ್‌ವೇರ್ ಎಂಜಿನಿಯರಿಂಗ್‌ನಲ್ಲಿ COCOMO ಮಾದರಿ ಯಾವ ಮಾಡ್ಯೂಲ್‌ನಲ್ಲಿ ಇದೆ?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS501", "facts": [["4", "ನಾಲ್ಕು"]]}),
    dict(id="kn09", lang="kn", text="ಕಂಪ್ಯೂಟರ್ ನೆಟ್‌ವರ್ಕ್ಸ್ ಸಿಲಬಸ್‌ನಲ್ಲಿ DNS ಎಲ್ಲಿ ಬರುತ್ತದೆ?", expected_tool="search_syllabus",
         check={"syllabus_code": "BCS502", "facts": [["5", "ಐದು", "Application", "ಅಪ್ಲಿಕೇಶನ್"]]}),
    dict(id="kn10", lang="kn", text="ನಾಳೆ ಸಂಜೆ 6 ಗಂಟೆಗೆ ಕಂಪ್ಯೂಟರ್ ನೆಟ್‌ವರ್ಕ್ಸ್ ಓದಲು ನೆನಪಿಸು.", expected_tool="set_reminder",
         check={"reminder_time": "2026-10-05 18:00", "facts": [["6", "ಆರು", "18:00"]]}),
    dict(id="kn11", lang="kn", text="ಅಕ್ಟೋಬರ್ 9 ಬೆಳಿಗ್ಗೆ 8 ಗಂಟೆಗೆ ಲ್ಯಾಬ್ ರೆಕಾರ್ಡ್ ಸಲ್ಲಿಸಲು ರಿಮೈಂಡರ್ ಇಡು.", expected_tool="set_reminder",
         check={"reminder_time": "2026-10-09 08:00", "facts": [["9", "ಒಂಬತ್ತು"]]}),
    dict(id="kn12", lang="kn", text="ನನ್ನ SGPA ಲೆಕ್ಕ ಹಾಕಿ: 4 ಕ್ರೆಡಿಟ್ ವಿಷಯದಲ್ಲಿ 92 ಅಂಕ, ಇನ್ನೊಂದು 4 ಕ್ರೆಡಿಟ್ ವಿಷಯದಲ್ಲಿ 85 ಮತ್ತು 3 ಕ್ರೆಡಿಟ್ ವಿಷಯದಲ್ಲಿ 72 ಅಂಕ.",
         expected_tool="calculate_sgpa", check={"facts": [["9.09"]]}),
    dict(id="kn13", lang="kn", text="ಮೊದಲ ಸೆಮಿಸ್ಟರ್ SGPA 8, ಎರಡನೇ ಸೆಮಿಸ್ಟರ್ SGPA 9, ಎರಡೂ 20 ಕ್ರೆಡಿಟ್. ನನ್ನ CGPA ಎಷ್ಟು?",
         expected_tool="calculate_cgpa", check={"facts": [["8.5"]]}),
    dict(id="kn14", lang="kn", text="ನಮಸ್ಕಾರ, ನೀವು ಯಾರು ಮತ್ತು ಏನು ಮಾಡಬಹುದು?", expected_tool=None,
         check={"facts": [["ಪರೀಕ್ಷೆ", "ಸಿಲಬಸ್", "ಪಠ್ಯಕ್ರಮ", "ರಿಮೈಂಡರ್", "ನೆನಪಿ", "SGPA", "CGPA", "exam", "syllabus"]]}),
]

# Voice per case: alternate the two neural voices of each language; every 3rd case gets noise.
from app.lang import VOICES  # noqa: E402

for i, c in enumerate(CASES):
    c["voice"] = VOICES[c["lang"]][i % 2]
    c["noise_snr_db"] = 15 if i % 3 == 2 else None

assert len(CASES) == 42 and len({c["id"] for c in CASES}) == 42
