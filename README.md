# SURREAL Processing

Convert the SURREAL dataset to be exploited by pavlakos et al CNN https://github.com/geopavlakos/c2f-vol-train

---

**The following commands can be used to generate an annotation file for all the videos in the SURREAL train dataset**

1. Generate a list of the files to be processed

```
cd /path/to/surreal/train && find -type f -name "*.mp4" | sed 's/\.mp4//' > list.txt
```

2. Generate h5 annotation and video file list (train.h5, train_videos.txt)
```
from surrealProcessing import *
surreal2pavlakos("path/to/surreal/train", "train")
```

3. Reduce the h5 file by a factor 0.1 (if necessary)
```
from surrealProcessing import *
reduceAnnotationFile("path/to/train.h5", 0.1, "output.h5")
```
