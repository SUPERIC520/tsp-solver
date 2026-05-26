import json

def main():
    log_path = r"C:\Users\eric2\.gemini\antigravity-cli\brain\c7d1dbfb-5cee-41c9-a666-aa7a89842b52\.system_generated\logs\transcript.jsonl"
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    
    indices = [275, 276, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 325, 341, 342, 343, 344, 345]
    for idx in indices:
        if idx < len(lines):
            step = lines[idx]
            print(f"=== STEP {step.get('step_index')} ({step.get('source')} / {step.get('type')}) ===")
            if 'thinking' in step:
                print('Thinking:', step['thinking'])
            if 'tool_calls' in step:
                print('Tool Calls:', step['tool_calls'])
            if 'content' in step:
                print('Content:', step['content'][:300] + '...')
            print()

if __name__ == '__main__':
    main()
