# HumanFallDetection

## Pose Detection Interface
Requirement: OpenPifPaf (pip3 install openpifpaf) <br>

## Usage
With GPU (requires CUDA): python3 posedetector.py  <br>
With CPU : python3 posedetector.py<br>
It uses resnet50 as the default model. To use other model specify with --checkpoint <br>
Ex: python3 posedetector.py --checkpoint=resnet18 <br>

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
