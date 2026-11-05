# R1-M-Lab C layout 2 ("images in two slots"): slot B request = 1 image first, then 12k tokens of real text, then the question
import base64, json, sys
img = "data:image/png;base64," + base64.b64encode(open(sys.argv[1], "rb").read()).decode()
text = " ".join(f"Line {i}: the quick brown fox jumps over the lazy dog number {i}." for i in range(40000, 40480))
content = [{"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": text + "\n\nWhat number is on the last line? Answer with the number only."}]
body = {"messages": [{"role": "user", "content": content}], "max_tokens": 32, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
json.dump(body, open("TI12k_cp.json", "w"))
