"""CLI utilities for user interaction."""

from pathlib import Path

from ..structure.config import (
    Orientation,
    StructureConfig,
    StructureType,
    create_ppvc_cluster_2_config,
    create_ppvc_cluster_4_config,
    create_ppvc_linear_config,
)


def _get_directory_contents(directory: Path) -> tuple[list[Path], list[Path]]:
    """Return subdirectories and CSV files in a directory."""
    subdirs = sorted([d for d in directory.iterdir() if d.is_dir()])
    csv_files = sorted(directory.glob("*.csv"))
    return subdirs, csv_files


def select_structure_type() -> StructureConfig:
    """Interactively select a structure type for rebar detection.

    Returns:
        StructureConfig for the selected structure type
    """
    print("\nAvailable structure types:")
    print("-" * 50)
    print("  [1] PPVC Type 1 - Linear (Linear, 4 rebars)")
    print("  [2] PPVC Type 2 - Cluster-2 (Cluster-2, 4 rebars)")
    print("  [3] PPVC Type 3 - Cluster-4 (Cluster-4, 8 rebars)")
    print("  [4] Column (2x2 grid)")
    print("-" * 50)
    print("\nSelect structure type (enter number):")

    while True:
        selection = input("\nSelection: ").strip()

        # Default to PPVC Linear if empty
        if selection == "":
            print("Selected: PPVC Type 1 - Linear (Linear)")
            return create_ppvc_linear_config(orientation=Orientation.HORIZONTAL)

        try:
            idx = int(selection)
            if idx == 1:
                print("Selected: PPVC Type 1 - Linear (Linear)")
                return create_ppvc_linear_config(orientation=Orientation.HORIZONTAL)
            elif idx == 2:
                print("Selected: PPVC Type 2 - Cluster-2 (Cluster-2)")
                return create_ppvc_cluster_2_config(orientation=Orientation.HORIZONTAL)
            elif idx == 3:
                print("Selected: PPVC Type 3 - Cluster-4 (Cluster-4)")
                return create_ppvc_cluster_4_config(orientation=Orientation.HORIZONTAL)
            elif idx == 4:
                print("Selected: Column (2x2 grid)")
                return StructureConfig()
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def select_csv_file(input_dir: Path, root_dir: Path | None = None) -> Path | None:
    """Interactively select a single CSV file to process with folder navigation.

    Args:
        input_dir: Directory containing CSV files
        root_dir: Root directory boundary for navigation (default: input_dir)

    Returns:
        Selected CSV file path, or None if cancelled or no files available
    """
    if root_dir is None:
        root_dir = input_dir

    current_dir = input_dir

    while True:
        subdirs, csv_files = _get_directory_contents(current_dir)

        if not subdirs and not csv_files:
            print(f"\nNo folders or CSV files found in {current_dir}")
            if current_dir != root_dir:
                current_dir = current_dir.parent
                continue
            return None

        print(f"\nCurrent directory: {current_dir}")
        print("-" * 50)

        items: list[tuple[str, Path | None]] = []
        if current_dir != root_dir:
            items.append((".. (parent)", None))
        for d in subdirs:
            items.append((f"{d.name}/", d))
        for f in csv_files:
            items.append((f.name, f))

        for i, (name, _) in enumerate(items):
            print(f"  [{i}] {name}")
        print("-" * 50)

        print("\nSelect a file or folder (enter number, 'q' to quit):")
        selection = input("\nSelection: ").strip().lower()

        if selection == "q":
            print("Cancelled.")
            return None

        # Default: select first CSV file
        if selection == "":
            if csv_files:
                print(f"Selected: {csv_files[0].name}")
                return csv_files[0]
            print("No CSV files in current directory. Please select a folder.")
            continue

        try:
            idx = int(selection)
            if 0 <= idx < len(items):
                name, path = items[idx]
                if path is None:
                    current_dir = current_dir.parent
                    continue
                if path.is_dir():
                    current_dir = path
                    continue
                print(f"Selected: {path.name}")
                return path
            print(f"Please enter a number between 0 and {len(items) - 1}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
