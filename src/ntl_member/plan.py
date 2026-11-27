"""Ordered A1 stages, also available before the server environment is installed."""

NAMES = (
    "preflight",
    "smoke",
    "download",
    "data",
    "stage1",
    "eval_s1",
    "stage2",
    "control",
    "eval_s2",
    "package",
    "report",
)


def print_plan() -> None:
    for i, name in enumerate(NAMES):
        print(f"{i:02d} {name}")


if __name__ == "__main__":
    print_plan()
