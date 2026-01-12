"""CLI utilities for user interaction."""

from pathlib import Path

from .structure import (
    Orientation,
    StructureConfig,
    StructureType,
    create_ppvc_cluster_2_config,
    create_ppvc_cluster_4_config,
    create_ppvc_linear_config,
)


def select_structure_type() -> StructureConfig:
    """Interactively select a structure type for rebar detection.

    Returns:
        StructureConfig for the selected structure type
    """
    print("\nAvailable structure types:")
    print("-" * 50)
    print("  [1] Column (2x2 grid)")
    print("  [2] PPVC Type 1 - Linear (Linear, 4 rebars)")
    print("  [3] PPVC Type 2 - Cluster-2 (Cluster-2, 4 rebars)")
    print("  [4] PPVC Type 3 - Cluster-4 (Cluster-4, 8 rebars)")
    print("-" * 50)
    print("\nSelect structure type (enter number):")

    while True:
        selection = input("\nSelection: ").strip()

        # Default to Column if empty
        if selection == "":
            print("Selected: Column (2x2 grid)")
            return StructureConfig()  # Default COLUMN type

        try:
            idx = int(selection)
            if idx == 1:
                print("Selected: Column (2x2 grid)")
                return StructureConfig()  # Default COLUMN type
            elif idx == 2:
                print("Selected: PPVC Type 1 - Linear (Linear)")
                return create_ppvc_linear_config(orientation=Orientation.HORIZONTAL)
            elif idx == 3:
                print("Selected: PPVC Type 2 - Cluster-2 (Cluster-2)")
                return create_ppvc_cluster_2_config(orientation=Orientation.HORIZONTAL)
            elif idx == 4:
                print("Selected: PPVC Type 3 - Cluster-4 (Cluster-4)")
                return create_ppvc_cluster_4_config(orientation=Orientation.HORIZONTAL)
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")


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
