from typing import List
from visual import CocoPart

def write_to_csv( frame_number: int, humans: List, width: int, height: int, csv_fp: str):
    """Save keypoint coordinates of the *first* human pose identified to a CSV file.
    Coordinates are scaled to refer the resized image.
    Columns are in order frame_no, nose.(x|y|p), (l|r)eye.(x|y|p), (l|r)ear.(x|y|p), (l|r)shoulder.(x|y|p),
    (l|r)elbow.(x|y|p), (l|r)wrist.(x|y|p), (l|r)hip.(x|y|p), (l|r)knee.(x|y|p), (l|r)ankle.(x|y|p)
    l - Left side of the identified joint
    r - Right side of the identified joint
    x - X coordinate of the identified joint
    y - Y coordinate of the identified joint
    p - Probability of the identified joint
    :param frame_number: Frame number for the video file
    :param humans: List of human poses identified
    :param width: Width of the image
    :param height: Height of the image
    :param csv_fp: Path to the CSV file
    """
    # Use only the first human identified using pose estimation
    coordinates = humans[0]['coordinates'] if len(humans) > 0 else [None for _ in range(17)]

    # Final row that will be written to the CSV file
    row = [frame_number] + ['' for _ in range(51)]  # Number of coco points * 3 -> 17 * 3 -> 51

    # Update the items in the row for every joint
    # TODO: Value for joints not identified? Currently stored as 0
    for part in CocoPart:
        if coordinates[part] is not None:
            index = 1 + 3 * part.value  # Index at which the values for this joint would start in the final row
            row[index] = coordinates[part][0] * width
            row[index + 1] = coordinates[part][1] * height
            row[index + 2] = coordinates[part][2]