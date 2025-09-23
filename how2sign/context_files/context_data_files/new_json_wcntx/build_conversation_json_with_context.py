
import pandas as pd
import json
from docx import Document

# === Load and process DOCX question sets ===
def load_docx_questions(path):
    doc = Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

# Load question sets
q_sent_translations = load_docx_questions("sent_Translations_Qset.docx")
q_transcript = load_docx_questions("questions_set_onlytranscript.docx")
q_keytopics = load_docx_questions("context_keytopics.docx")
q_sentiment = load_docx_questions("context_sentiment.docx")
q_summary = load_docx_questions("context_summary.docx")
q_insights = load_docx_questions("context_insights.docx")

# === Load CSV file ===
df = pd.read_csv("how2sign_realigned_train_sorted.csv")
df['SENTENCE'] = df['SENTENCE']  # Placeholder for actual sentence content

# Group sentences by VIDEO_NAME
grouped = df.groupby("VIDEO_NAME")["SENTENCE"].apply(list).to_dict()

# === Load context JSON ===
with open("context_train_all.json", "r") as f:
    context_data = json.load(f)

# Build a lookup for video_name -> assistant responses
context_lookup = {entry["VIDEO_NAME"]: entry["CONTEXTUAL_RESULT"] for entry in context_data}

# === rotate through questions ===
def cyclic_get(lst, idx):
    return lst[idx % len(lst)]

# Extract specific assistant field from context
def extract_assistant_response(convos, section_header):
    for turn in convos:
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            parts = content.split(f"**{section_header}**:")
            if len(parts) > 1:
                return parts[1].split("**")[0].strip()
    return f"(Missing {section_header})"


# === Build the JSON structure ===
final_json = []

for idx, (video_name, sentences) in enumerate(grouped.items()):
    context_convos = context_lookup.get(video_name, [])
    conversations = [
        {"from": "human", "value": "<image>\n" + cyclic_get(q_sent_translations, idx)},
        {"from": "gpt", "value": "\n".join([f"<sentence{i+1}>{s}<sentence{i+1}>" for i, s in enumerate(sentences)])},
        {"from": "human", "value": "<image>\n" + cyclic_get(q_transcript, idx)},
        {"from": "gpt", "value": " ".join(sentences)},
        {"from": "human", "value": "<image>\n" + cyclic_get(q_keytopics, idx)},
        {"from": "gpt", "value": extract_assistant_response(context_convos, "Key Topics Discussed")},
        {"from": "human", "value": "<image>\n" + cyclic_get(q_sentiment, idx)},
        {"from": "gpt", "value": extract_assistant_response(context_convos, "Sentiment")},
        {"from": "human", "value": "<image>\n" + cyclic_get(q_summary, idx)},
        {"from": "gpt", "value": extract_assistant_response(context_convos, "Short Summary")},
        {"from": "human", "value": "<image>\n" + cyclic_get(q_insights, idx)},
        {"from": "gpt", "value": extract_assistant_response(context_convos, "Notable Key Insights")}
    ]

    final_json.append({
        "id": video_name,
        "video": f"{video_name}.mp4",
        "conversations": conversations
    })

# === Save to JSON file ===
with open("constructed_video_conversations.json", "w") as f:
    json.dump(final_json, f, indent=2)
