import os
import re

CSS_VARS = {
    r"rgba\(\s*15,\s*23,\s*42,\s*0\.\d+\s*\)": "var(--panel-bg)",
    r"rgba\(\s*30,\s*41,\s*59,\s*0\.\d+\s*\)": "var(--card-bg)",
    r"rgba\(\s*255,\s*255,\s*255,\s*0\.0\d+\s*\)": "var(--border-color-light)",
    r"rgba\(\s*148,\s*163,\s*184,\s*0\.\d+\s*\)": "var(--border-color)",
    r"#0f172a": "var(--panel-bg-solid)",
    r"#1e293b": "var(--card-bg-solid)",
    r"#1e1e38": "var(--card-bg-solid)",
    r"#f8fafc": "var(--text-main)",
    r"#94a3b8": "var(--text-muted)",
    r"#6366f1": "var(--primary-color)"
}

directory = r"d:\三创\LaRE-main\web\vue\src"

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".vue") or file.endswith(".css"):
            path = os.path.join(root, file)
            # Skip main.css to not overwrite variable definitions
            if file == "main.css":
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                new_content = content
                for pattern, repl in CSS_VARS.items():
                    new_content = re.sub(pattern, repl, new_content, flags=re.IGNORECASE)
                    
                if new_content != content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated {path}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
