import asyncio
import os
import sys
from pathlib import Path

# Ensure PYTHONPATH=.
sys.path.append(os.getcwd())
from fortuna import FortunaDB
from fortuna_utils import now_eastern

async def main():
    db_path = Path("fortuna.db")
    if not db_path.exists():
        return

    db = FortunaDB(str(db_path))
    await db.initialize()
    stats = await db.get_stats()

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "summary.md")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n## 📈 Overall ROI Performance\n\n")
        f.write(f"- **Total Tips:** {stats['total_tips']}\n")
        f.write(f"- **Cashed:** {stats['cashed']} | **Burned:** {stats['burned']} | **Voided:** {stats['voided']}\n")
        f.write(f"- **Net Profit:** ${stats['total_profit']:+.2f}\n")

        if stats.get('builder_analytics'):
            f.write("\n### 🛠️ Top Builders (Best Bets)\n\n")
            f.write("| Builder | BB Total | BB Cashed | BB Profit |\n")
            f.write("| --- | --- | --- | --- |\n")
            # Sort by BB Profit
            sorted_builders = sorted(stats['builder_analytics'].items(),
                                    key=lambda x: x[1].get('bb_profit', 0), reverse=True)
            for name, b in sorted_builders[:5]:
                if b['bb_total'] > 0:
                    f.write(f"| {name} | {b['bb_total']} | {b['bb_cashed']} | ${b['bb_profit']:+.2f} |\n")

    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
