# Face Detection Using Haar Cascades

This is a trial to evaluate the effectiveness of Haar Cascades for face detection.

## Observation
- The result of the face detection is not very accurate.
- Depending on the value set in `scaleFactor` and `minNeighbors`, the number of faces detected can vary.
- With `scaleFactor = 1.1` and `minNeighbors = 2`.
  <image src="Public/min_neigh_2.png"/>
- For some reasons, there 2 faces that are always undetected.
  <image src="Public/undetected.png"/>