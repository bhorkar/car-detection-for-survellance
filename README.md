# car-detection-for-survellance
First version performs car detection on the images captured from the adoid phone running IP webcam 

The images are first fileted using a haar cascade filter to shortlist the images over which deep leaning based
algorithms will be applied. 
The screened images are pssed through Faster RCNN that produced bounding box
for cars 

The second version uses tornado. The  bounding boxed video is stream to the 5000 port for viewing on the browser 
