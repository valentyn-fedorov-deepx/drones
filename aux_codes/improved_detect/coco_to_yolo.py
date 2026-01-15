import fiftyone as fo
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Optional
from fiftyone import ViewField as F
from tqdm import tqdm


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Convert COCO format dataset to Ultralytics (YOLO) format with class filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Convert only person class (default behavior)
    python script.py --input /path/to/coco --output /path/to/output
    
    # Convert specific classes
    python script.py --input /path/to/coco --output /path/to/output --classes person car dog
    
    # Convert all classes
    python script.py --input /path/to/coco --output /path/to/output --all-classes
        '''
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        type=str,
        help='Path to COCO dataset directory containing images and annotations'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        type=str,
        help='Path to output directory for YOLO format dataset'
    )
    
    # Create mutually exclusive group for class selection
    class_group = parser.add_mutually_exclusive_group()
    
    class_group.add_argument(
        '--classes',
        nargs='+',
        type=str,
        help='List of classes to keep (default: person)'
    )
    
    class_group.add_argument(
        '--all-classes',
        action='store_true',
        help='Convert all classes instead of filtering'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite output directory if it exists'
    )

    parser.add_argument(
        '--split-name',
        help='Name of the split. Either val or train.'
    )
    
    return parser.parse_args()

def convert_coco_to_yolo(
    coco_dataset_path: str, 
    output_dir: str, 
    classes_to_keep: Optional[List[str]] = None,
    force: bool = False,
    split_name: str = "train"
):
    """
    Convert COCO format dataset to Ultralytics (YOLO) format using FiftyOne.
    Filters for specific classes, defaulting to 'person' if no classes specified.
    
    Args:
        coco_dataset_path (str): Path to COCO dataset directory containing annotations and images
        output_dir (str): Path to output directory for YOLO format dataset
        classes_to_keep (List[str], optional): List of class names to keep. Defaults to ['person']
        force (bool): Whether to overwrite existing output directory
    """
    # Set default classes to keep
    if classes_to_keep is None:
        classes_to_keep = ['person']
        
    # Check if output directory exists
    output_dir = Path(output_dir)
    if output_dir.exists() and not force:
        raise ValueError(
            f"Output directory '{output_dir}' already exists. "
            "Use --force to overwrite."
        )
    
    # Create output directory structure
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Load COCO dataset using FiftyOne
    print(f"Loading COCO dataset from {coco_dataset_path}...")
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(coco_dataset_path, f"{split_name}/images"),
        labels_path=os.path.join(coco_dataset_path, f"{split_name}_annotations.json"),
        # max_samples=10,
    )
    
    # If all_classes is True, use all available classes
    if len(classes_to_keep) == 0:  # all classes
        classes_to_keep = sorted(list(dataset.default_classes))
        print(f"Converting all {len(classes_to_keep)} classes")
    else:
        print(f"Filtering for classes: {classes_to_keep}")
    
    # Filter dataset to keep only specified classes
    filtered_dataset = dataset.clone()
    
    # Create a filter expression for the specified classes
    class_filter = F("label").is_in(classes_to_keep)
    
    print("Filtering dataset...")
    filtered_dataset = filtered_dataset.filter_labels("detections", class_filter)
    # Apply the filter to each sample in the dataset
    # for sample in tqdm(dataset):
    #     import ipdb; ipdb.set_trace()
    #     # Get detections for specified classes only
    #     filtered_detections = sample.detections.filter(class_filter)
        
    #     if len(filtered_detections) > 0:
    #         # If sample has detections of interest, add it to filtered dataset with only those detections
    #         sample.detections = filtered_detections
    #         filtered_dataset.add_sample(sample)
    #     else:
    #         # If sample has no detections of interest, remove it from filtered dataset
            # filtered_dataset.delete_samples(sample.id)
    
    print(f"\nOriginal dataset size: {len(dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    # Export filtered dataset to YOLO format
    print("\nExporting to YOLO format...")
    filtered_dataset.export(
        export_dir=str(output_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        split_prefix="images",
        classes=classes_to_keep
    )
    
    # Create dataset.yaml file
    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val

nc: {len(classes_to_keep)}
names: {classes_to_keep}
    """
    
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content.strip())
    
    print(f"\nDataset converted successfully. Output directory: {output_dir}")
    print(f"Classes converted: {classes_to_keep}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    train_images = len(list((output_dir / 'images' / 'train').glob('*.jpg')))
    val_images = len(list((output_dir / 'images' / 'val').glob('*.jpg')))
    
    print(f"Training images: {train_images}")
    print(f"Validation images: {val_images}")
    print(f"Total images: {train_images + val_images}")
    
    # Print label statistics
    print("\nLabel Statistics:")
    label_counts = {}
    for sample in tqdm(filtered_dataset):
        for detection in sample.detections.detections:
            label = detection.label
            label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count} instances")

if __name__ == "__main__":
    args = parse_args()
    
    # Determine classes to keep
    if args.all_classes:
        classes_to_keep = []  # empty list signals to use all classes
    else:
        classes_to_keep = args.classes if args.classes else ['person']
    
    # Run conversion
    convert_coco_to_yolo(
        coco_dataset_path=args.input,
        output_dir=args.output,
        classes_to_keep=classes_to_keep,
        force=args.force,
        split_name=args.split_name
    )