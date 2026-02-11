with open('fortuna.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    line_num = i + 1
    # Revert if it's not where we wanted it
    if "for race in (all_races or races):" in line:
        if line_num not in [3492, 4013]:
             line = line.replace('(all_races or races)', 'races')
    new_lines.append(line)

with open('fortuna.py', 'w') as f:
    f.writelines(new_lines)
