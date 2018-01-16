# surreal_processing
Convert the SURREAL dataset to be exploited by pavlakos et al CNN https://github.com/geopavlakos/c2f-vol-train

Generate a list of the files to be processed

```
cd /path/to/surreal/train && find -type f -name "*.mp4" | sed 's/\.mp4//' > list.txt
```

Generate h5 annotation and video file list (train.h5, train_videos.txt)
```
from surrealProcessing import *
surrealProcessing("path/to/surreal/train", "train")
```

Reduce the h5 file by a factor 0.1
```
from surrealProcessing import *
reduceAnnotationFile("path/to/train.h5", 0.1, "output.h5")
```
