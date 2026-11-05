# R1 split-stream requests: mkreq.py IMG OUTDIR  (same text generator as ~/repro_img/mkreq.py)
import base64, json, sys
img = "data:image/png;base64," + base64.b64encode(open(sys.argv[1], "rb").read()).decode()
out = sys.argv[2]
def text(a, b):
    return " ".join(f"Line {i}: the quick brown fox jumps over the lazy dog number {i}." for i in range(a, b))
def req(parts, name):
    content = []
    for p in parts:
        content.append({"type": "image_url", "image_url": {"url": img}} if p == "IMG" else {"type": "text", "text": p})
    content.append({"type": "text", "text": "\n\nWhat number is on the last line? Answer with the number only."})
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 32, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    json.dump(body, open(f"{out}/{name}.json", "w"))
req(["IMG", text(10000, 12000)], "S0_img")     # slot 0: 1 image then ~50k text, answer 11999
req([text(10000, 12000)], "S0_txt")            # control: same text, no image
req([text(30000, 32000)], "S1_txt")            # slot 1: ~50k text, answer 31999
# deep variant: slot-0 text sized so its last batch before the 1028-token checkpoint break holds ~512 tokens
req([text(30000, 32518)], "T_deep")            # ~63k text, answer 32517
req([text(10000, 12000)], "S0_txt")
