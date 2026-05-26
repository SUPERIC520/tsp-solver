import json

def main():
    log_path = r"C:\Users\eric2\.gemini\antigravity-cli\brain\c7d1dbfb-5cee-41c9-a666-aa7a89842b52\.system_generated\logs\transcript_full.jsonl"
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            step = json.loads(line)
            if step.get('step_index') == 276:
                print(step.get('thinking'))
                break

if __name__ == '__main__':
    main()
