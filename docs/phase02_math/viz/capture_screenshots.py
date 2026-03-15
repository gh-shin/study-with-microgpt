from playwright.sync_api import sync_playwright
import os

urls = [
    ("viz_01_power_rule.html", "viz_01.png"),
    ("viz_02_log_and_loss.html", "viz_02.png"),
    ("viz_03_exp.html", "viz_03.png"),
    ("viz_04_partial_add.html", "viz_04.png"),
    ("viz_05_partial_mul.html", "viz_05.png"),
    ("viz_06_chain_rule.html", "viz_06.png"),
    ("viz_07_softmax_ce.html", "viz_07.png"),
]

viz_dir = r"e:\workspace\micro-gpt\docs\phase02_math\viz"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 800, "height": 800})
    for html_file, out_png in urls:
        abs_path = os.path.join(viz_dir, html_file)
        # Using file:// schema
        file_url = f"file:///{abs_path.replace(chr(92), '/')}"
        print(f"Navigating to {file_url}")
        page.goto(file_url, wait_until="networkidle")
        
        # Wait a bit for charts to draw
        page.wait_for_timeout(1000)
        
        out_path = os.path.join(viz_dir, out_png)
        # take screenshot of the card element
        try:
             card = page.locator(".card")
             card.screenshot(path=out_path)
             print(f"Saved {out_path}")
        except Exception as e:
             print(f"Error on {html_file}: {e}")
             page.screenshot(path=out_path)
             print(f"Saved full page {out_path}")
    browser.close()
