# FaceNet-IOT
IOT implementation for FaceNet project by David Sandberg https://github.com/davidsandberg/facenet

This code takes base FaceNet solution and builds IOT for continuous face recognition by looping through below steps 
1) Take a snap
2) Identify and align faces using MTCNN
3) Invoke Facenet to recognize face
4) Print result of recognized faces
5) Start over

It takes less than 0.5 seconds for each frame to process and recognize faces on a high end personal laptop which is expected performance for surveillance or identity and access control type of solutions

Add IOT-FaceReco.py python file to /src/align

You may want to first go through https://github.com/davidsandberg/facenet in detail to set up trained model to identify your custom set of faces and use this sample to run IOT on that. 

Please email me if you need more details / help knowing IOT with Facenet

