import re

with open("fortuna.py", "r") as f:
    content = f.read()

replacements = {
    'Official_Mountaineer': 'https://www.cnty.com/mountaineer/racing/entries-results/',
    'Official_MahoningValley': 'https://www.hollywoodmahoningvalley.com/racing/entries-results/',
    'Official_NorthfieldPark': 'https://mgmnorthfieldpark.mgmresorts.com/en/racing.html',
    'Official_Hastings': 'https://www.hastingsracecourse.com/racing/',
    'Official_SaratogaHarness': 'https://saratogacasino.com/racing/',
    'Official_TampaBayDowns': 'https://www.tampabaydowns.com/betting/entries/',
}

for name, url in replacements.items():
    pattern = rf'(class {name}Adapter\(OfficialTrackAdapter\):.*?super\(\)\.__init__\(".*?",\s*)"(.*?)"'
    def sub_func(m):
        return f'{m.group(1)}"{url}"'
    content = re.sub(pattern, sub_func, content, flags=re.DOTALL)

# Switch failing ones to PLAYWRIGHT or just ensure they use chrome124 correctly
# Actually, OfficialTrackAdapter now uses CURL_CFFI chrome124 by default in my last change.

with open("fortuna.py", "w") as f:
    f.write(content)
