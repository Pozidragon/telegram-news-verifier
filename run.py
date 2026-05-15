"""Лаунчер проекту Telegram News Verifier."""
from __future__ import annotations

import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

W = 60


def line(char="─"):
    print(char * W)


def header(text: str) -> None:
    print()
    line("═")
    print(f"  {text}")
    line("═")


def run(script: str, desc: str = "") -> None:
    if desc:
        print(f"\n  {desc}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        print(f"\n  [!] Завершився з кодом {result.returncode}")
    input("\n  [Enter — повернутись до меню]")


def pick(options: list[tuple[str, str]]) -> str:
    print()
    for i, (_, label) in enumerate(options, 1):
        print(f"  {i}. {label}")
    print("  0. Назад")
    print()
    while True:
        choice = input("  Вибір: ").strip()
        if choice == "0":
            return ""
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except ValueError:
            pass
        print("  Введіть число зі списку.")


MENUS: dict[str, list[tuple[str, str, str]]] = {
    "collection": [
        ("scripts/collect_telegram.py", "Зібрати пости з Telegram", "Підключаємось до Telegram API..."),
        ("scripts/collect_news.py",     "Зібрати новини з RSS",     "Завантажуємо новини..."),
    ],
    "verification": [
        ("scripts/run_verification.py",          "TF-IDF верифікація",                     ""),
        ("scripts/run_semantic_verification.py", "Семантична верифікація (~400 МБ модель)", ""),
    ],
    "experiments": [
        ("scripts/run_experiment.py",          "TF-IDF експеримент",      ""),
        ("scripts/run_semantic_experiment.py", "Семантичний експеримент", ""),
    ],
}

TITLES = {
    "collection":   "ЗБІР ДАНИХ",
    "verification": "ВЕРИФІКАЦІЯ",
    "experiments":  "ОЦІНКА",
}

MAIN_OPTS = [
    ("collection",   "Збір даних"),
    ("verification", "Верифікація"),
    ("experiments",  "Оцінка (експерименти)"),
]


def submenu(key: str) -> None:
    entries = MENUS[key]
    opts = [(e[0], e[1]) for e in entries]
    desc_map = {e[0]: e[2] for e in entries}
    while True:
        header(TITLES[key])
        script = pick(opts)
        if not script:
            break
        run(script, desc_map[script])


def main() -> None:
    while True:
        header("TELEGRAM NEWS VERIFIER")
        key = pick(MAIN_OPTS)
        if not key:
            print("\n  До побачення!\n")
            break
        submenu(key)


if __name__ == "__main__":
    main()
