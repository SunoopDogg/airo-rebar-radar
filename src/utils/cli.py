"""CLI utilities for user interaction."""

from pathlib import Path


def select_csv_file(input_dir: Path) -> Path | None:
    """Interactively select a single CSV file to process.

    Args:
        input_dir: Directory containing CSV files

    Returns:
        Selected CSV file path, or None if no files available
    """
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return None

    # Display file list
    print("\nAvailable CSV files:")
    print("-" * 40)
    for i, f in enumerate(csv_files, 1):
        print(f"  [{i}] {f.name}")
    print("-" * 40)

    # User input prompt
    print("\nSelect a file to process (enter number):")

    while True:
        selection = input("\nSelection: ").strip()

        # Default to first file if empty
        if selection == "":
            print(f"Selected: {csv_files[0].name}")
            return csv_files[0]

        try:
            idx = int(selection)
            if 1 <= idx <= len(csv_files):
                print(f"Selected: {csv_files[idx - 1].name}")
                return csv_files[idx - 1]
            print(f"Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
