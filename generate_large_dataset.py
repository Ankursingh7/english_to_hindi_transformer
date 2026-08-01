import pandas as pd
import random

base_pairs = [
    ("Hello, how are you?", "नमस्ते, आप कैसे हैं?"),
    ("Welcome to our college.", "हमारे कॉलेज में आपका स्वागत है।"),
    ("Machine Learning is very interesting.", "मशीन लर्निंग बहुत दिलचस्प है।"),
    ("Artificial Intelligence is the future.", "आर्टिफिशियल इंटेलिजेंस भविष्य है।"),
    ("I love programming in Python.", "मुझे पाइथॉन में प्रोग्रामिंग करना पसंद है।"),
    ("The weather is very pleasant today.", "आज मौसम बहुत सुहावना है।"),
    ("Transformers are state-of-the-art NLP models.", "ट्रांसफॉर्मर अत्याधुनिक एनएलपी मॉडल हैं।"),
    ("Students are preparing for their exams.", "छात्र अपनी परीक्षाओं की तैयारी कर रहे हैं।"),
    ("Knowledge is power.", "ज्ञान ही शक्ति है।"),
    ("Hard work leads to success.", "कड़ी मेहनत सफलता की ओर ले जाती है।"),
    ("What is your name?", "आपका नाम क्या है?"),
    ("I live in India.", "मैं भारत में रहता हूँ।"),
    ("She reads books every day.", "वह हर दिन पुस्तकें पढ़ती है।"),
    ("Technology is changing rapidly.", "प्रौद्योगिकी तेजी से बदल रही है।"),
    ("We must protect the environment.", "हमें पर्यावरण की रक्षा करनी चाहिए।"),
    ("Education opens many doors.", "शिक्षा कई दरवाजे खोलती है।"),
    ("Practice makes a person perfect.", "अभ्यास इंसान को निपुण बनाता है।"),
    ("Computers process data quickly.", "कंप्यूटर डेटा को तेज़ी से प्रोसेस करते हैं।"),
    ("Nature is beautiful.", "प्रकृति सुंदर है।"),
    ("Never stop learning.", "सीखना कभी बंद न करें।"),
    ("Good morning, have a great day!", "सुप्रभात, आपका दिन शुभ हो!"),
    ("Python is a popular programming language.", "पाइथॉन एक लोकप्रिय प्रोग्रामिंग भाषा है।"),
    ("Artificial intelligence will help humanity.", "कृत्रिम बुद्धिमत्ता मानवता की मदद करेगी।"),
    ("Learning new skills is very important.", "नए कौशल सीखना बहुत महत्वपूर्ण है।"),
    ("This is a neural machine translation project.", "यह एक न्यूरल मशीन ट्रांसलेशन प्रोजेक्ट है।"),
    ("Deep learning models require training data.", "डीप लर्निंग मॉडल को ट्रेनिंग डेटा की आवश्यकता होती है।"),
    ("Data science is an essential field.", "डेटा साइंस एक आवश्यक क्षेत्र है।"),
    ("Computers use binary numbers.", "कंप्यूटर बाइनरी संख्याओं का उपयोग करते हैं।"),
    ("Algorithms solve complex problems.", "एल्गोरिदम जटिल समस्याओं को हल करते हैं।"),
    ("Neural networks mimic the human brain.", "न्यूरल नेटवर्क मानव मस्तिष्क की नकल करते हैं।")
]

en_templates = [
    "I am studying {topic}.",
    "The professor explained {topic} today.",
    "We are building a project on {topic}.",
    "Understanding {topic} is essential for students.",
    "Can you teach me about {topic}?",
    "Every student should learn {topic}.",
    "This book contains information about {topic}.",
    "The research paper focuses on {topic}.",
    "He scored high marks in {topic}.",
    "She is writing a paper on {topic}."
]

hi_templates = [
    "मैं {topic} का अध्ययन कर रहा हूँ।",
    "प्रोफेसर ने आज {topic} के बारे में समझाया।",
    "हम {topic} पर एक प्रोजेक्ट बना रहे हैं।",
    "छात्रों के लिए {topic} को समझना आवश्यक है।",
    "क्या आप मुझे {topic} के बारे में पढ़ा सकते हैं?",
    "प्रत्येक छात्र को {topic} सीखना चाहिए।",
    "इस पुस्तक में {topic} के बारे में जानकारी है।",
    "शोध पत्र {topic} पर केंद्रित है।",
    "उन्होंने {topic} में अच्छे अंक प्राप्त किए।",
    "वह {topic} पर एक पेपर लिख रही है।"
]

topics = [
    ("Machine Learning", "मशीन लर्निंग"),
    ("Artificial Intelligence", "आर्टिफिशियल इंटेलिजेंस"),
    ("Data Structures", "डेटा स्ट्रक्चर्स"),
    ("Python Programming", "पाइथॉन प्रोग्रामिंग"),
    ("Neural Networks", "न्यूरल नेटवर्क"),
    ("Computer Networks", "कंप्यूटर नेटवर्क"),
    ("Database Management", "डेटाबेस प्रबंधन"),
    ("Operating Systems", "ऑपरेटिंग सिस्टम"),
    ("Software Engineering", "सॉफ्टवेयर इंजीनियरिंग"),
    ("Natural Language Processing", "नेचुरल लैंग्वेज प्रोसेसिंग"),
    ("Computer Vision", "कंप्यूटर विज़न"),
    ("Cloud Computing", "क्लाउड कंप्यूटिंग"),
    ("Cyber Security", "साइबर सुरक्षा"),
    ("Web Development", "वेब डेवलपमेंट"),
    ("Robotics", "रोबोटिक्स"),
    ("Deep Learning", "डीप लर्निंग"),
    ("Algorithm Design", "एल्गोरिदम डिज़ाइन"),
    ("Big Data Analytics", "बिग डेटा एनालिटिक्स"),
    ("Internet of Things", "इंटरनेट ऑफ थिंग्स"),
    ("Blockchain Technology", "ब्लॉकचेन तकनीक")
]

pairs = list(base_pairs)

for en_tmpl, hi_tmpl in zip(en_templates, hi_templates):
    for en_top, hi_top in topics:
        pairs.append((en_tmpl.format(topic=en_top), hi_tmpl.format(topic=hi_top)))

random.seed(42)
while len(pairs) < 2000:
    selected_pair = random.choice(pairs)
    pairs.append(selected_pair)

df = pd.DataFrame(pairs, columns=["english", "hindi"])
df.to_csv("data/sample_en_hi_large.csv", index=False, encoding="utf-8")
print(f"Generated {len(df)} sentence pairs in 'data/sample_en_hi_large.csv'")
