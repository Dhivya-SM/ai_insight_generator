import pandas as pd
import os

# Ensure output folder exists
os.makedirs("data/training", exist_ok=True)

# -------------------------
# 1. Sentiment Training Data
# -------------------------
sentiment_data = [
    ["I love my new Dell laptop, it's super fast!", "positive"],
    ["Dell customer service was really helpful today.", "positive"],
    ["My Dell XPS is overheating again. Disappointed.", "negative"],
    ["Battery life on my Dell Inspiron is terrible.", "negative"],
    ["Dell monitors have great picture quality.", "positive"],
    ["The new Dell update broke my Wi-Fi connection.", "negative"],
    ["The Dell keyboard feels very comfortable to type on.", "positive"],
    ["Dell support took too long to respond.", "negative"],
    ["Dell Alienware is perfect for gaming!", "positive"],
    ["Dell's warranty service was frustrating and slow.", "negative"],
    ["I am very happy with Dell's new touchscreen laptop.", "positive"],
    ["Dell laptop screen cracked within a month.", "negative"],
    ["Dell XPS 15 is sleek, lightweight, and powerful.", "positive"],
    ["My Dell PC crashes every time I open a browser.", "negative"],
    ["Dell technicians fixed my laptop within a day.", "positive"],
    ["Dell charged me extra for a simple repair.", "negative"],
    ["Best customer support I’ve had in years from Dell.", "positive"],
    ["The Dell BIOS update made my laptop unusable.", "negative"],
    ["Dell’s build quality feels premium and durable.", "positive"],
    ["Dell shipped my laptop late without any notice.", "negative"]
]

sentiment_df = pd.DataFrame(sentiment_data, columns=["text", "label"])
sentiment_df.to_csv("data/training/sentiment_train.csv", index=False)
print("✅ sentiment_train.csv created with", len(sentiment_df), "records.")

# -------------------------
# 2. Theme/Topic Classification Data
# -------------------------
theme_data = [
    ["Dell launches new range of business laptops", "product_launch"],
    ["Dell's quarterly earnings exceed expectations", "financial_news"],
    ["How to troubleshoot a Dell BIOS error", "technical_support"],
    ["Dell partners with Microsoft on cloud solutions", "partnership"],
    ["Dell employee benefits ranked among the best", "hr_policy"],
    ["Dell expands manufacturing in India", "business_expansion"],
    ["Dell hosts annual tech innovation summit", "event"],
    ["Dell recalls defective power adapters", "product_issue"],
    ["Dell integrates AI into enterprise solutions", "product_update"],
    ["Dell donates laptops to underprivileged students", "csr_initiative"],
    ["Dell unveils eco-friendly packaging initiative", "csr_initiative"],
    ["Dell signs multi-year contract with NASA", "partnership"],
    ["Dell introduces liquid cooling for data centers", "product_update"],
    ["Dell to hire 5000 new employees in 2025", "hr_policy"],
    ["Dell experiences supply chain delays due to global crisis", "business_issue"]
]

theme_df = pd.DataFrame(theme_data, columns=["text", "label"])
theme_df.to_csv("data/training/theme_train.csv", index=False)
print("✅ theme_train.csv created with", len(theme_df), "records.")

# -------------------------
# 3. Summarization Training Data
# -------------------------
summarization_data = [
    [
        "Dell has announced a new range of XPS laptops featuring the latest Intel processors, improved battery life, and upgraded display panels for better color accuracy.",
        "Dell announces upgraded XPS laptops with faster processors and better displays."
    ],
    [
        "Dell is partnering with several universities to provide free laptops and technical training to students in rural areas as part of its CSR initiative.",
        "Dell to provide free laptops and training to rural students."
    ],
    [
        "Following a firmware bug in certain models, Dell has issued a recall for specific laptop batteries due to potential overheating risks.",
        "Dell recalls laptop batteries over overheating risks."
    ],
    [
        "Dell's revenue for Q4 exceeded analyst expectations, driven by strong demand for enterprise servers and cloud solutions.",
        "Dell beats Q4 revenue expectations on strong server sales."
    ],
    [
        "At the annual tech summit, Dell showcased its vision for AI-powered business tools and announced collaborations with key software vendors.",
        "Dell unveils AI business tools at annual tech summit."
    ],
    [
        "Dell has introduced a new liquid cooling solution for its data center customers, aiming to reduce energy consumption and improve performance.",
        "Dell launches liquid cooling for data centers to save energy."
    ],
    [
        "As part of its green initiative, Dell announced it will use 100% recycled materials in its packaging starting next year.",
        "Dell to switch to 100% recycled packaging."
    ],
    [
        "Dell's Alienware division released a new gaming laptop featuring NVIDIA's latest RTX graphics and high-refresh displays.",
        "Alienware launches new gaming laptop with latest RTX graphics."
    ],
    [
        "Dell signed a strategic agreement with NASA to provide high-performance computing solutions for upcoming space missions.",
        "Dell partners with NASA for high-performance computing."
    ],
    [
        "Due to global semiconductor shortages, Dell warns customers of potential delivery delays for certain models.",
        "Dell warns of delivery delays due to chip shortage."
    ]
]

summarization_df = pd.DataFrame(summarization_data, columns=["text", "summary"])
summarization_df.to_csv("data/training/summarization_train.csv", index=False)
print("✅ summarization_train.csv created with", len(summarization_df), "records.")
