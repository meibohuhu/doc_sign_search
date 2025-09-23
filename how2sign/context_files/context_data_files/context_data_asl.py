import torch
import json
from transformers import pipeline

# Define the model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load the JSON file with video IDs and GPT transcripts
input_json_path = "/home/vp1837/ASL-AI/ASL_research/datasets/How2Sign/TRAIN_DATA/rgb_videos_df_ssh.json"
with open(input_json_path, "r", encoding="utf-8") as f:
    video_data = json.load(f)

# Process each entry for contextual insights
contextual_results = []
for entry in video_data:
    video_id = entry["id"]
    video_transcript = ""
    
    # Get the transcript from GPT's response
    for conv in entry["conversations"]:
        if conv["from"] == "gpt":
            video_transcript = conv["value"]
            break  # Take the first GPT response

    if not video_transcript.strip():
        continue  # Skip if no transcript found

    messages = [
        {"role": "system", "content": "You are an AI assistant trained to analyze video transcripts and extract contextual insights."},
        {"role": "user", "content": (
            f"Analyze the following video transcript and extract contextual information:\n\n{video_transcript}\n\n"
            "Provide:\n- A list of key topics discussed.\n- The sentiment (positive, negative, or neutral).\n"
            "- A short summary.\n- Any notable key insights."
        )},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=500,
        do_sample=False,
    )

    assistant_response = outputs[0]["generated_text"]
    print(f"\n{video_id}:\n{assistant_response}\n")

    contextual_results.append({
        "VIDEO_ID": video_id,
        "VIDEO_TEXTS": video_transcript,
        "CONTEXTUAL_RESULT": assistant_response
    })

# Save the contextual results
output_json_path = "/home/vp1837/ASL-AI/ASL_research/datasets/How2Sign/TRAIN_DATA/context_train.json"
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(contextual_results, json_file, indent=4, ensure_ascii=False)

print(f"Contextual results saved to: {output_json_path}")
