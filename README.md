# untitled-aruco-project
Little whatever project using Aruco markers

[Package not released]

## Package example:
```python
from untitled_core import Marker
import cv2

marker = Marker(cv2.aruco.DICT_6X6_50, 5, 500)
cv2.imshow("Marker", marker.img)

wait_key = cv2.waitKey(0)
```

![image](https://github.com/user-attachments/assets/a75ecc27-0325-4774-b62d-9e8ec740786a)
