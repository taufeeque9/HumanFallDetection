# HumanFallDetection

## Pose Detection Interface
Requirement: OpenPifPaf (pip3 install openpifpaf) <br>

## Usage
### fall_detector.py
#### To just see the COCO pointset on webcam: <br>
python3 fall_detector.py --coco_points
#### To run the code on a video: <br>
python3 fall_detector.py --video=path/to/video.mp4
#### To run on resnet18:
python3 fall_detector.py --checkpoint=resnet18
#### To plot the features of webcam: <br>
python3 fall_detector.py --plot_graph
#### To plot the features of a video: <br>
python3 fall_detector.py --plot_graph --video=path/to/video.mp4

### process_data.py
#### To plot the features of activity 1 of subjects 1 to 10:
python3 process_data.py --activity1 --sub_range=1,11

## Avg FPS
<TABLE>
<TR><TH></TH><TH>Resnet18</TH><TH>Resnet50</TH></TR>
<TR><TH>With CUDA</TH><TD>18</TD><TD>9</TD></TR>
<TR><TH>Without CUDA</TH><TD>8</TD><TD>4</TD></TR>
</TABLE>

## References:
https://github.com/vita-epfl/openpifpaf

https://github.com/samkit-jain/physio-pose/blob/master/physio.py

https://colab.research.google.com/drive/1H8T4ZE6wc0A9xJE4oGnhgHpUpAH5HL7W#scrollTo=6O_hZNsW7hwV
