"""CLI utilities for user interaction."""

from pathlib import Path


def select_csv_files(input_dir: Path) -> list[Path]:
    """Interactively select CSV files to process.

    Args:
        input_dir: Directory containing CSV files

    Returns:
        List of selected CSV file paths
    """
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return []

    # Display file list
    print("\nAvailable CSV files:")
    print("-" * 40)
    for i, f in enumerate(csv_files, 1):
        print(f"  [{i}] {f.name}")
    print("-" * 40)

    # User input prompt
    print("\nSelect files to process:")
    print("  - Single file: 1")
    print("  - Multiple files: 1,2,3")
    print("  - Range: 1-3")
    print("  - All files: all or a")

    while True:
        selection = input("\nSelection: ").strip().lower()

        # Parse input
        if selection in ('all', 'a', ''):
            return csv_files

        try:
            selected_indices = set()
            for part in selection.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = part.split('-')
                    selected_indices.update(range(int(start), int(end) + 1))
                else:
                    selected_indices.add(int(part))

            # Filter valid indices
            selected = [csv_files[i - 1] for i in sorted(selected_indices)
                        if 1 <= i <= len(csv_files)]

            if selected:
                return selected
            print("No valid files selected. Please try again.")
        except ValueError:
            print("Invalid input format. Please try again.")
