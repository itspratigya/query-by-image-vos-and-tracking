Query-by-Image Video Object Search and Tracking Using SAM 2

Owners: Pratigya Ranjan and Manju Tadi

Institute: SJSU

Course: CMPE-252 

Query-by-Image Video Object Search and Tracking is an application-level extension to Meta’s SAM 2 that automates target initialization using visual queries. Instead of requiring manual interactive prompting (points, bounding boxes, or masks) on specific video frames, the system allows users to provide a reference image of an object to automatically search, track, and segment it across long video sequences. The pipeline leverages a frozen DINOv2 backbone to map the target image and video frames into shared visual embedding space, retrieving optimal candidate frames via cosine similarity thresholding. The top-matched frame index and target bounding box are then seamlessly passed to SAM 2 for memory-based temporal propagation and masklet generation. Evaluated on the DAVIS 2017 and MOSE datasets across metrics such as Recall@K, J&F, IoU, and runtime, this repository investigates whether image-query prompting can eliminate manual video setup while preserving SAM 2's state-of-the-art segmentation performance.

Branch Strategy:
- main contain the final code and any change must never be directly pushed into main
- release contain active/current code. Any new branch should be created from release and should be kept updated. Contributors can continue working in their separate branches and should create PR request for merging changes into release. This way we account for what changes we are pushing into release and all members get to know about the changes. 
- After release is properly tested and changes are merged from all contributors, changes can be merged into main with PR.
